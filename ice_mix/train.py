import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning import seed_everything
import os
from typing import List, cast, Optional

from graphnet.utilities.logging import Logger
from graphnet.models import StandardModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.detector.icecube import IceCube86
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.training.labels import JointLabel
from graphnet.models.task.reconstruction import JointPositionandDirectionReco
from graphnet.training.loss_functions import (
    JointLoss,
    EuclideanDistanceLoss,
    VonMisesFisher3DLoss,
)
from torch.optim import AdamW

# Local imports
from src.models.transformer import IceMix
from src.utils import features, truth

# PyTorch Lightning imports
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from src.utils import CheckSamplerCallback, EpochMonitorCallback


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Set up logging
    logger = Logger()
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if "seed" in cfg:
        seed_everything(cfg.seed)

    # Create output directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    # --- Data Module Setup ---
    # Determine data paths
    data_paths = list(cfg.data.path)
    # Check for local NVMe override
    local_data_dir = os.environ.get("LOCAL_DATA_DIR")
    if local_data_dir and os.path.exists(local_data_dir):
        logger.info(f"Using local NVMe data from {local_data_dir}")
        new_data_paths = []
        for p in data_paths:
            basename = os.path.basename(p)
            local_path = os.path.join(local_data_dir, basename)
            if os.path.exists(local_path):
                new_data_paths.append(local_path)
            else:
                logger.warning(
                    f"Local file {local_path} not found. Keeping original {p}"
                )
                new_data_paths.append(p)
        data_paths = new_data_paths

    # Graph Definition
    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=256,
            z_name="sensor_pos_z",
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=features,
        columns=[0, 1, 2, 3],
    )

    # Dynamic import of selections
    try:
        from src.utils import (
            NuMu_Training_Selections,
            NuMu_Validation_Selections,
            NuE_Training_Selections,
            NuE_Validation_Selections,
        )

        train_selections = [NuMu_Training_Selections, NuE_Training_Selections]
        val_selections = [NuMu_Validation_Selections, NuE_Validation_Selections]
    except ImportError:
        logger.warning("Could not import selections from src.utils. Using None.")
        train_selections = None
        val_selections = None

    # Override selections for augmented data to use all available events
    # This prevents errors when the augmented data contains different/new event IDs
    if any("augmented" in str(p) for p in data_paths):
        logger.info(
            "Augmented data detected. Expanding training selections with augmented events."
        )

        import sqlite3

        def get_augmentation_count(db_path: str) -> int:
            """Detect number of rotations by checking max event ID."""
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT MAX(event_no) FROM truth")
                    max_id = cursor.fetchone()[0]
                    if max_id is None:
                        return 0
                    # Original IDs are small, augmented are shifted by 10**10 * rotation
                    # e.g., max_id = 50000001234 -> 5 rotations
                    return int(max_id // 10**10)
            except Exception as e:
                logger.warning(f"Could not check augmentation count for {db_path}: {e}")
                return 0

        # We must have valid initial selections to expand upon
        if train_selections is not None:
            # We need to iterate over pairs of (dataset_path, selection_list)
            # But train_selections is just a list of lists [sel_db1, sel_db2, ...]
            new_train_selections = []

            for i, p in enumerate(data_paths):
                # Get base selection for this dataset
                if i < len(train_selections):
                    base_selection = train_selections[i]
                else:
                    logger.warning(
                        f"No selection found for {p}, skipping augmentation expansion."
                    )
                    new_train_selections.append(None)
                    continue

                if base_selection is None:
                    new_train_selections.append(None)
                    continue

                # Detect rotations
                n_rotations = get_augmentation_count(p)
                logger.info(
                    f"Dataset {os.path.basename(p)}: Detected {n_rotations} rotations."
                )

                # Expand selection
                expanded_selection = list(base_selection)  # Start with copy of original
                if n_rotations > 0:
                    for rot in range(1, n_rotations + 1):
                        # Shift: id + rot * 10^10
                        shift = rot * 10**10
                        augmented_ids = [eid + shift for eid in base_selection]
                        expanded_selection.extend(augmented_ids)
                    logger.info(
                        f"  Expanded training selection from {len(base_selection)} to {len(expanded_selection)} events."
                    )

                new_train_selections.append(expanded_selection)

            train_selections = new_train_selections

        # Validation selections remain untouched (original events only)
        # val_selections = val_selections

    data_module = GraphNeTDataModulecustom(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth_table": cfg.data.truth_table,
            "pulsemaps": cfg.data.pulsemap,
            "truth": truth,
            "features": features,
            "path": data_paths,
            "graph_definition": graph_definition,
        },
        train_dataloader_kwargs={
            "batch_size": cfg.data.batch_size,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.data.pin_memory,
            "persistent_workers": cfg.data.persistent_workers,
            "prefetch_factor": cfg.data.prefetch_factor,
            "multiprocessing_context": "spawn",
        },
        train_selections=train_selections,
        val_selections=val_selections,
        test_selection=[None, None],
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth",
                zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels",
            )
        },
        train_val_split=cfg.data.train_val_split,
    )

    # --- Model Setup ---
    # Instantiate custom IceMix backbone
    backbone = IceMix(
        hidden_dim=cfg.attention.hidden_dim,
        seq_length=cfg.attention.seq_length,
        depth=cfg.attention.depth,
        head_size=cfg.attention.head_size,
        depth_rel=cfg.attention.n_rel,
        n_rel=cfg.attention.n_rel,
        scaled_emb=cfg.attention.scaled_emb,
        include_dynedge=cfg.attention.include_dynedge,
        n_features=cfg.attention.n_features,
        maha_encoder=cfg.attention.maha_encoder,
        dropout=cfg.attention.dropout,
        attn_drop=cfg.attention.attn_drop,
        proj_drop=cfg.attention.proj_drop,
        drop_path_rate=cfg.attention.drop_path_rate,
        token_drop=cfg.data.get("token_drop", cfg.attention.get("token_drop", 0.0)),
    )

    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=JointLoss(
            alpha=cfg.alpha,
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(),
        ),
    )

    # Instantiate optimizer configuration
    optimizer_kwargs = {"lr": cfg.lr, "eps": 1e-05}

    # Instantiate scheduler configuration
    scheduler_class = None
    scheduler_kwargs = None
    scheduler_config = None

    if cfg.use_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler_class = ReduceLROnPlateau
        scheduler_kwargs = {
            "mode": "min",
            "patience": cfg.scheduler.patience,
            "factor": cfg.scheduler.factor,
            "verbose": True,
        }
        scheduler_config = {"monitor": "val_loss", "frequency": 1}

    model = cast(
        StandardModel,
        StandardModel(
            graph_definition=graph_definition,
            backbone=backbone,
            tasks=[task],
            optimizer_class=AdamW,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        ),
    )

    # --- Training Setup ---
    # Loggers
    loggers = []
    # CSV Logger
    csv_logger = CSVLogger(save_dir=cfg.logs_dir, name="training_logs", version=None)
    loggers.append(csv_logger)

    # WandB Logger
    if cfg.get("wandb", False):
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            entity=cfg.get("wandb_entity", None),
            save_dir=cfg.logs_dir,
            log_model=True,
            name=cfg.run_name.replace("/", "_"),
        )
        loggers.append(wandb_logger)

    # Callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        EpochMonitorCallback(),
        CheckSamplerCallback(),
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]

    if cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", patience=cfg.early_stopping_patience, mode="min"
            )
        )

    logger.info("Starting Standard Training using Lightning Trainer...")

    # Check GPUS
    # If using torchrun, Lightning usually detects DDP automatically.
    # However, we can pass the gpus config if it's explicitly set.
    gpus = cfg.gpus
    if isinstance(gpus, (list, tuple)) and len(gpus) == 1 and gpus[0] == 0:
        # If config says [0] but we might want more, we should ideally trust the launch command (torchrun).
        # But Model.fit() arguments will override logic.
        # If we pass gpus=[0], Lightning might restrict to device 0.
        # For DDP with torchrun, passing gpus usually gets ignored or should be set to match world size.
        # We will try passing the list as is.
        pass

    # Note: StandardModel.fit signature handles constructing the Trainer.
    model.fit(
        train_dataloader=data_module.train_dataloader,
        val_dataloader=data_module.val_dataloader,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        logger=loggers,
        callbacks=callbacks,
        distribution_strategy="ddp",
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gpus=cfg.gpus,
        num_sanity_val_steps=0,
    )


if __name__ == "__main__":
    main()
