import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pytorch_lightning import seed_everything
import os
from typing import List, cast, Optional

from graphnet.utilities.logging import Logger
from graphnet.models import StandardModel
from graphnet.models.graphs import GraphDefinition
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.detector.icecube import IceCube86
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.training.labels import JointLabel
from graphnet.models.task.reconstruction import JointPositionandDirectionReco
from graphnet.training.loss_functions import (
    EuclideanDistanceLoss,
    VonMisesFisher3DLoss,
)
from torch.optim import AdamW, LBFGS

# Local imports
from src.models.lbfgs_model import (
    DistributedLBFGSStandardModel,
    freeze_for_last_blocks_fine_tuning,
)
from src.models.ema_model import EMAStandardModel
from src.models.transformer import IceMix
from src.utils import (
    CheckSamplerCallback,
    EpochMonitorCallback,
    RandomRotationCallback,
    TokenDropSeedCallback,
    features,
    get_dynamic_splits,
    load_csv_splits,
    truth,
)
from src.metrics_logging import (
    JointLossWithMetrics,
    NonFiniteLossCallback,
    PhysicsMetricsCallback,
)

# PyTorch Lightning imports
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Set up logging
    logger = Logger()
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    if "seed" in cfg:
        seed_everything(cfg.seed)

    # Create output directories - only on rank 0
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
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

    # Using GraphDefinition instead of KNNGraph avoids the persistent KNN edge build
    graph_definition = GraphDefinition(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=cfg.data.max_pulses,
            z_name="sensor_pos_z",
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=features,
    )

    split_cfg = cfg.data.split
    if split_cfg.mode == "csv":
        train_selections, val_selections, test_selections = load_csv_splits(
            data_paths=data_paths,
            train_csvs=list(split_cfg.train_csvs),
            val_csvs=list(split_cfg.val_csvs),
            test_csvs=list(split_cfg.test_csvs) if split_cfg.test_csvs else None,
        )
    elif split_cfg.mode == "random":
        train_selections, val_selections, test_selections = get_dynamic_splits(
            data_paths=data_paths,
            seed=split_cfg.seed,
            split_ratio=list(split_cfg.ratio),
        )
    else:
        raise ValueError(
            f"Unknown data.split.mode={split_cfg.mode!r}; expected 'random' or 'csv'."
        )

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

        # Validation and test selections remain untouched (original events only)
        # val_selections = val_selections
        # test_selections = test_selections

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
            # torch.DataLoader rejects persistent_workers/prefetch_factor when num_workers=0.
            **(
                {
                    "persistent_workers": cfg.data.persistent_workers,
                    "prefetch_factor": cfg.data.prefetch_factor,
                    "multiprocessing_context": "spawn",
                }
                if cfg.num_workers > 0
                else {
                    "persistent_workers": False,
                    "prefetch_factor": None,
                }
            ),
        },
        train_selections=train_selections,
        val_selections=val_selections,
        test_selection=test_selections,
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth",
                zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels",
            )
        },
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
        drop_chance=cfg.data.get("drop_chance", 1.0),
        pos_time_multiplier=cfg.attention.pos_time_multiplier,
        charge_rde_multiplier=cfg.attention.charge_rde_multiplier,
        spacetime_distance_scale=cfg.attention.spacetime_distance_scale,
        spacetime_distance_clip_min=cfg.attention.spacetime_distance_clip_min,
        spacetime_distance_clip_max=cfg.attention.spacetime_distance_clip_max,
        spacetime_distance_multiplier=cfg.attention.spacetime_distance_multiplier,
        mlp_ratio=cfg.attention.mlp_ratio,
        init_values=cfg.attention.init_values,
        n_freq=cfg.attention.n_freq,
    )

    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=JointLossWithMetrics(
            alpha=cfg.alpha,
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(),
        ),
    )

    # Instantiate optimizer configuration
    optimizer_name = str(cfg.get("optimizer", {}).get("name", "adamw")).lower()
    if optimizer_name == "adamw":
        optimizer_class = AdamW
        optimizer_kwargs = {
            "lr": cfg.lr,
            "eps": cfg.optimizer.adamw.eps,
            "weight_decay": cfg.optimizer.adamw.weight_decay,
        }
    elif optimizer_name == "lbfgs":
        optimizer_class = LBFGS
        lbfgs_cfg = cfg.optimizer.lbfgs
        optimizer_kwargs = {
            "lr": lbfgs_cfg.lr,
            "max_iter": lbfgs_cfg.max_iter,
            "history_size": lbfgs_cfg.history_size,
            "line_search_fn": lbfgs_cfg.line_search_fn,
            "tolerance_grad": lbfgs_cfg.tolerance_grad,
            "tolerance_change": lbfgs_cfg.tolerance_change,
        }
    else:
        raise ValueError(
            f"Unknown optimizer.name={optimizer_name!r}; expected 'adamw' or 'lbfgs'."
        )

    if optimizer_name == "lbfgs" and str(cfg.precision) != "32-true":
        logger.warning(
            f"LBFGS is incompatible with automatic mixed precision; "
            f"overriding precision={cfg.precision!s} with precision=32-true."
        )
        cfg.precision = "32-true"

    # Instantiate scheduler configuration
    scheduler_class = None
    scheduler_kwargs = None
    scheduler_config = None

    if optimizer_name == "lbfgs" and cfg.use_scheduler:
        logger.warning("Disabling scheduler for LBFGS fine-tuning.")
    elif cfg.use_scheduler:
        scheduler_name = str(cfg.scheduler.get("name", "plateau")).lower()
        if scheduler_name == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau

            scheduler_class = ReduceLROnPlateau
            scheduler_kwargs = {
                "mode": "min",
                "patience": cfg.scheduler.patience,
                "factor": cfg.scheduler.factor,
            }
            scheduler_config = {"monitor": "val_loss", "frequency": 1}
        elif scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler_class = CosineAnnealingLR
            scheduler_kwargs = {
                "T_max": int(cfg.max_epochs),
                "eta_min": float(cfg.lr) * float(cfg.scheduler.eta_min_factor),
            }
            scheduler_config = {"interval": "epoch", "frequency": 1}
        else:
            raise ValueError(
                f"Unknown scheduler.name={scheduler_name!r}; expected "
                "'plateau' or 'cosine'."
            )

    ema_enabled = bool(cfg.get("ema", {}).get("enabled", False))
    if ema_enabled and optimizer_name != "adamw":
        raise ValueError("EMA is supported only with optimizer.name=adamw.")
    if optimizer_name == "lbfgs":
        standard_model_class = DistributedLBFGSStandardModel
    elif ema_enabled:
        standard_model_class = EMAStandardModel
    else:
        standard_model_class = StandardModel
    model_kwargs = {}
    if ema_enabled:
        model_kwargs["ema_decay"] = float(cfg.ema.decay)
    model = cast(
        StandardModel,
        standard_model_class(
            graph_definition=graph_definition,
            backbone=backbone,
            tasks=[task],
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
            **model_kwargs,
        ),
    )
    if optimizer_name == "lbfgs":
        model.reset_lbfgs_history_each_step = bool(
            cfg.optimizer.lbfgs.reset_history_each_step
        )

    fine_tune_from_ckpt = cfg.get("fine_tune_from_ckpt")
    resume_ckpt = cfg.get("ckpt_path")
    if fine_tune_from_ckpt:
        logger.info(
            f"Loading weights for fine-tuning from checkpoint: {fine_tune_from_ckpt}"
        )
        checkpoint = torch.load(fine_tune_from_ckpt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if isinstance(model, EMAStandardModel):
            model.load_source_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    train_last_n_blocks = cfg.fine_tune.get("train_last_n_blocks")
    if train_last_n_blocks is not None:
        if not (fine_tune_from_ckpt or resume_ckpt):
            raise ValueError(
                "fine_tune.train_last_n_blocks requires fine_tune_from_ckpt "
                "or ckpt_path."
            )
        trainable, total = freeze_for_last_blocks_fine_tuning(
            model, int(train_last_n_blocks)
        )
        logger.info(
            "Fine-tuning the prediction head and final "
            f"{int(train_last_n_blocks)} transformer block(s): "
            f"{trainable:,} / {total:,} parameters trainable "
            f"({100.0 * trainable / total:.2f}%)."
        )

    # --- Training Setup ---
    # Loggers
    loggers = []
    # CSV Logger
    csv_logger = CSVLogger(save_dir=cfg.logs_dir, name="training_logs", version=None)
    loggers.append(csv_logger)

    # WandB Logger
    if cfg.get("wandb", False):
        wandb_config = OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=False,
            enum_to_str=True,
        )
        wandb_logger = WandbLogger(
            project=cfg.get("wandb_project") or cfg.project_name,
            entity=cfg.get("wandb_entity", None),
            save_dir=cfg.logs_dir,
            log_model=True,
            name=cfg.run_name.replace("/", "_"),
            group=cfg.get("wandb_group", None),
            config=wandb_config,
        )
        loggers.append(wandb_logger)

    # Callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        EpochMonitorCallback(),
        CheckSamplerCallback(),
        TokenDropSeedCallback(seed=cfg.seed),
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            filename="best-{epoch:02d}-{val_loss:.8f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
    ]

    if cfg.get("fail_on_non_finite", False):
        callbacks.append(NonFiniteLossCallback())

    callbacks.append(PhysicsMetricsCallback())

    if cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", patience=cfg.early_stopping_patience, mode="min"
            )
        )

    if cfg.data.get("augment_rotation", False):
        logger.info("Enabling On-the-Fly Random Rotation Augmentation.")
        callbacks.append(
            RandomRotationCallback(seed=cfg.data.get("rotation_seed", None))
        )

    logger.info("Starting Standard Training using Lightning Trainer...")

    num_devices = torch.cuda.device_count()
    ckpt_path = None if fine_tune_from_ckpt else resume_ckpt

    # Note: StandardModel.fit signature handles constructing the Trainer.
    model.fit(
        train_dataloader=data_module.train_dataloader,
        val_dataloader=data_module.val_dataloader,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        logger=loggers,
        callbacks=callbacks,
        ckpt_path=ckpt_path,
        distribution_strategy="ddp",
        precision=cfg.precision,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        gpus=num_devices,
        limit_train_batches=cfg.get("limit_train_batches"),
        limit_val_batches=cfg.get("limit_val_batches"),
        num_sanity_val_steps=0,
    )


if __name__ == "__main__":
    main()
