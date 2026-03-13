import os
import glob
import re
import argparse
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Optional, List, cast
import pandas as pd
from pytorch_lightning import Trainer
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
from src.models.transformer import IceMix
from src.utils import features, truth
from torch.optim import AdamW

from src.utils import get_dynamic_splits


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.exists(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "best-epoch=*-val_loss=*.ckpt")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        pattern_all = os.path.join(checkpoint_dir, "*.ckpt")
        checkpoint_files = glob.glob(pattern_all)
        if not checkpoint_files:
            return None

    best_loss = float("inf")
    best_checkpoint = None

    for checkpoint_path in checkpoint_files:
        filename = os.path.basename(checkpoint_path)
        match = re.search(r"val_loss=([0-9]+\.?[0-9]*)", filename)
        if match:
            val_loss = float(match.group(1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_path
        elif "last" in filename and best_checkpoint is None:
            if best_loss == float("inf"):
                best_checkpoint = checkpoint_path

    return best_checkpoint


def run_prediction(
    run_dir: str,
    cfg: DictConfig,
    ckpt_path: str,
    gpus: Optional[List[int]],
    drop_percentages: List[float],
    test_fraction: float = 1.0,
    use_test_split: bool = False,
) -> None:
    logger = Logger()
    logger.info(f"Processing run: {run_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")

    data_paths = list(cfg.data.path)
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
                new_data_paths.append(p)
        data_paths = new_data_paths

    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=cfg.data.max_pulses,
            z_name="sensor_pos_z",
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=features,
        columns=[0, 1, 2, 3],
    )

    split_seed = cfg.data.get("split_seed", 42)
    split_ratio = cfg.data.get("split_ratio", [0.8, 0.1, 0.1])
    
    _, val_selections, test_selections = get_dynamic_splits(
        data_paths, seed=split_seed, split_ratio=split_ratio
    )
    
    eval_selections = test_selections if use_test_split else val_selections
    
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
        train_selections=None,
        val_selections=eval_selections,
        test_selection=[None, None],
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth",
                zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels",
            )
        },
        train_val_split=split_ratio[:2],
    )

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

    # Set force_token_drop flag for eval mode
    backbone.force_token_drop = True

    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=JointLoss(
            alpha=cfg.alpha,
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(),
        ),
    )

    model = cast(
        StandardModel,
        StandardModel(
            graph_definition=graph_definition,
            backbone=backbone,
            tasks=[task],
            optimizer_class=AdamW,
            optimizer_kwargs={"lr": 1e-3},
        ),
    )

    logger.info(f"Loading state dict from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if gpus and torch.cuda.is_available():
        logger.info(f"Using GPUs: {gpus}")
    else:
        logger.info("Using CPU")

    additional_attributes = [
        "zenith",
        "azimuth",
        "position_x",
        "position_y",
        "position_z",
        "event_no",
        "energy",
        "pid",
        "interaction_type",
        "oneweight",
        "n_pulses",
    ]
    prediction_columns = [
        "pos_x_pred",
        "pos_y_pred",
        "pos_z_pred",
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "dir_kappa_pred",
    ]

    # Initialize dataloader once
    data_module.setup("fit")
    val_dataloader = data_module.val_dataloader

    if test_fraction < 1.0:
        logger.info(
            f"Running in TEST mode: using {test_fraction * 100:.2f}% of validation data."
        )
        from torch.utils.data import Subset
        import numpy as np

        loaders = (
            val_dataloader if isinstance(val_dataloader, list) else [val_dataloader]
        )
        new_loaders = []
        for loader in loaders:
            ds = loader.dataset
            num_samples = max(1, int(len(ds) * test_fraction))
            indices = np.random.choice(len(ds), num_samples, replace=False).tolist()
            subset = Subset(ds, indices)
            new_loader = type(loader)(
                subset,
                batch_size=loader.batch_size,
                num_workers=loader.num_workers,
                collate_fn=getattr(loader, "collate_fn", None),
                pin_memory=loader.pin_memory,
            )
            new_loaders.append(new_loader)

        val_dataloader = (
            new_loaders if isinstance(val_dataloader, list) else new_loaders[0]
        )

    output_path = os.path.join(run_dir, "resilience_results")
    os.makedirs(output_path, exist_ok=True)

    for drop_pct in drop_percentages:
        logger.info(f"Starting prediction loop with drop_pct = {drop_pct:.2%} ...")

        # Override token_drop directly
        backbone.token_drop = drop_pct

        results = model.predict_as_dataframe(
            val_dataloader,
            additional_attributes=additional_attributes,
            prediction_columns=prediction_columns,
            gpus=gpus,
        )

        csv_name = f"drop_{drop_pct:.2f}.csv"
        csv_path = os.path.join(output_path, csv_name)
        results.to_csv(csv_path)
        logger.info(f"Results for {drop_pct:.2%} saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run resilience predictions for IceMix models by dropping tokens."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="ice_mix/outputs",
        help="Base directory containing run outputs",
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan for runs but do not execute prediction",
    )
    parser.add_argument(
        "--drop-percentages",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50],
        help="List of token drop percentages to test",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=1.0,
        help="Fraction of validation data to evaluate on (e.g. 0.01 for 1%). Default is 1.0",
    )
    parser.add_argument(
        "--use-test-split",
        action="store_true",
        help="If set, evaluate on the 10% test split instead of the validation split.",
    )
    args = parser.parse_args()

    logger = Logger()
    base_dir = args.base_dir
    if not os.path.isabs(base_dir):
        base_dir = os.path.abspath(base_dir)

    logger.info(f"Scanning {base_dir} for runs...")

    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist.")
        return

    config_files = glob.glob(
        os.path.join(base_dir, "**", ".hydra", "config.yaml"), recursive=True
    )

    for config_file in config_files:
        run_dir = os.path.dirname(os.path.dirname(config_file))
        logger.info(f"Found run: {run_dir}")

        # Check if all resilience results already exist
        resilience_dir = os.path.join(run_dir, "resilience_results")
        all_exist = True
        for drop_pct in args.drop_percentages:
            if not os.path.exists(
                os.path.join(resilience_dir, f"drop_{drop_pct:.2f}.csv")
            ):
                all_exist = False
                break

        if all_exist and args.drop_percentages:
            logger.info(
                f"Skipping {run_dir} - All requested resilience predictions already exist."
            )
            continue

        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        best_ckpt = find_best_checkpoint(checkpoint_dir)

        if not best_ckpt:
            logger.warning(f"No valid checkpoints found in {checkpoint_dir}. Skipping.")
            continue

        if args.dry_run:
            logger.info(
                f"[Dry Run] Would predict for {run_dir} using {best_ckpt} with drops {args.drop_percentages}"
            )
            continue

        try:
            cfg = OmegaConf.load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {e}")
            continue

        gpus = None
        if torch.cuda.is_available():
            gpus = list(range(min(torch.cuda.device_count(), args.n_gpus)))

        try:
            run_prediction(
                run_dir, cfg, best_ckpt, gpus, args.drop_percentages, args.test_fraction, args.use_test_split
            )
        except Exception as e:
            logger.error(f"Failed to run resilience prediction for {run_dir}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
