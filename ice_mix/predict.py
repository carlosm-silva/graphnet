"""Run checkpoint inference for discovered IceMix output directories.

The CLI reconstructs GraphNeT objects from Hydra configuration, selects a best
checkpoint, prefers EMA weights, and writes predictions and model artifacts.
"""

import os
import glob
import re
import argparse
import torch
from omegaconf import OmegaConf, DictConfig
from typing import Optional, List, Any, cast
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
from src.models.ema_model import extract_inference_state_dict
from src.utils import (
    features,
    get_configured_splits,
    select_evaluation_split,
    truth,
    write_evaluation_manifest,
)
from torch.optim import AdamW

def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the best checkpoint below ``checkpoint_dir``.

    Returns the path whose filename contains the smallest parsed validation
    loss. ``last.ckpt`` is a fallback; ``None`` is returned when no checkpoint
    exists.
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Pattern to match checkpoint files: best-epoch=X-val_loss=Y.ckpt
    # Adjust pattern if needed based on what ModelCheckpoint actually saves
    # train.py uses: filename="best-{epoch:02d}-{val_loss:.8f}"
    pattern = os.path.join(checkpoint_dir, "best-epoch=*-val_loss=*.ckpt")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        # Fallback to checking for any .ckpt file if specific pattern fails
        pattern_all = os.path.join(checkpoint_dir, "*.ckpt")
        checkpoint_files = glob.glob(pattern_all)
        if not checkpoint_files:
            return None

    best_loss = float("inf")
    best_checkpoint = None

    for checkpoint_path in checkpoint_files:
        filename = os.path.basename(checkpoint_path)
        # Try to extract val_loss
        match = re.search(r"val_loss=([0-9]+\.?[0-9]*)", filename)
        if match:
            val_loss = float(match.group(1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_path
        elif "last" in filename and best_checkpoint is None:
            # Keep last as backup if no best- found yet, but don't prioritize it
            if best_loss == float("inf"):
                best_checkpoint = checkpoint_path

    return best_checkpoint


def run_prediction(
    run_dir: str,
    cfg: DictConfig,
    ckpt_path: str,
    gpus: Optional[List[int]],
    use_test_split: bool,
) -> None:
    """Reconstruct one run, perform inference, and write prediction artifacts.

    ``cfg`` reconstructs the data/model objects, ``ckpt_path`` supplies plain
    or EMA weights, ``gpus`` is passed to GraphNeT prediction, and
    ``use_test_split`` selects test rather than validation events. CSV/model
    artifacts are written below ``run_dir``. The function returns ``None``.
    """
    logger = Logger()
    logger.info(f"Processing run: {run_dir}")
    logger.info(f"Checkpoint: {ckpt_path}")

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
                new_data_paths.append(p)
        data_paths = new_data_paths

    # Reconstruct Graph Definition
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

    train_selections, val_selections, test_selections, train_val_split = (
        get_configured_splits(data_paths, cfg.data)
    )

    partition, eval_selections = select_evaluation_split(
        val_selections, test_selections, use_test_split
    )

    # Reconstruct Data Module (Validation only)
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
            "batch_size": cfg.data.batch_size,  # Use training batch size or could leverage larger for inference
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.data.pin_memory,
            "persistent_workers": cfg.data.persistent_workers,
            "prefetch_factor": cfg.data.prefetch_factor,
            "multiprocessing_context": "spawn",
        },
        # GraphNeT's custom data module discards caller-provided validation
        # selections when train_selections is None, so preserve both here.
        train_selections=train_selections,
        val_selections=eval_selections,
        test_selection=[None] * len(data_paths),
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth",
                zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels",
            )
        },
        train_val_split=train_val_split,
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
        loss_function=JointLoss(
            alpha=cfg.alpha,
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(),
        ),
    )

    # We need to construct the StandardModel to load state dict matching structure
    # Optimizer/Scheduler args are dummy here as we evaluate
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

    # Load Checkpoint
    logger.info(f"Loading state dict from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    inference_state = extract_inference_state_dict(ckpt)
    if any(
        key.startswith("_ema_model.module.")
        for key in ckpt.get("state_dict", {})
    ):
        logger.info("EMA checkpoint detected; using EMA weights for prediction.")
    model.load_state_dict(inference_state)
    model.eval()

    # Move to GPU if available and requested
    if gpus and torch.cuda.is_available():
        logger.info(f"Using GPUs: {gpus}")
        # device = torch.device(f"cuda:{gpus[0]}")
        # model.to(device)
    else:
        logger.info("Using CPU")

    # Prediction columns
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

    logger.info("Starting prediction loop...")

    # Use predict_as_dataframe from StandardModel
    # Note: data_module.val_dataloader might be a list if multiple datasets/selections are used
    # GraphNeTDataModulecustom creates a CombinedLoader or list of loaders?
    # Checking train.py: data_module.val_dataloader is passed to Trainer.fit
    # In GraphNeT, predict_as_dataframe expects a single DataLoader.

    # We need to instantiate the dataloader.
    data_module.setup("fit")  # Or 'validate'
    val_dataloader = data_module.val_dataloader

    results = model.predict_as_dataframe(
        val_dataloader,
        additional_attributes=additional_attributes,
        prediction_columns=prediction_columns,
        gpus=gpus,
    )

    # Save Results
    output_path = os.path.join(run_dir, "predictions")
    os.makedirs(output_path, exist_ok=True)

    csv_path = os.path.join(output_path, "results.csv")
    results.to_csv(csv_path)
    logger.info(f"Results saved to {csv_path}")

    write_evaluation_manifest(
        os.path.join(output_path, "evaluation_manifest.json"),
        data_paths=data_paths,
        event_selections=eval_selections,
        partition=partition,
        checkpoint_path=ckpt_path,
        split_config=cfg.data.get("split", cfg.data),
        use_test_split=use_test_split,
    )

    # Save model artifacts
    model.save_state_dict(f"{output_path}/state_dict.pth")
    model.save_config(f"{output_path}/model_config.yml")
    logger.info("Model artifacts saved.")


def main():
    """Discover output runs and optionally execute checkpoint inference."""
    parser = argparse.ArgumentParser(description="Run predictions for IceMix models.")
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
        "--use-test-split",
        action="store_true",
        help="If set, evaluate on the 10% test split instead of the validation split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run predictions even if they already exist.",
    )
    args = parser.parse_args()

    logger = Logger()
    base_dir = args.base_dir
    if not os.path.isabs(base_dir):
        # Assuming relative to project root / current working dir
        base_dir = os.path.abspath(base_dir)

    logger.info(f"Scanning {base_dir} for runs...")

    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist.")
        return

    # Find run directories in base_dir.
    run_dirs = []

    # Old format: base_dir/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml
    old_format_configs = glob.glob(
        os.path.join(base_dir, "*", "*", ".hydra", "config.yaml")
    )
    for config_path in old_format_configs:
        run_dirs.append(os.path.dirname(os.path.dirname(config_path)))

    # New format: directories directly under base_dir that contain a 'checkpoints' folder
    for d in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, d)
        if os.path.isdir(dir_path) and os.path.exists(
            os.path.join(dir_path, "checkpoints")
        ):
            run_dirs.append(dir_path)

    run_dirs = list(set(run_dirs))

    for run_dir in run_dirs:
        logger.info(f"Found run: {run_dir}")

        # Find hydra config for this run
        config_file = None
        old_format_config = os.path.join(run_dir, ".hydra", "config.yaml")
        if os.path.exists(old_format_config):
            config_file = old_format_config
        else:
            # New format: extract date and time from the directory name
            dir_name = os.path.basename(run_dir)
            match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", dir_name)
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                # Config is located at graphnet/outputs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml
                # base_dir is graphnet/ice_mix/outputs
                graphnet_dir = os.path.dirname(os.path.dirname(base_dir))
                potential_config = os.path.join(
                    graphnet_dir, "outputs", date_str, time_str, ".hydra", "config.yaml"
                )
                if os.path.exists(potential_config):
                    config_file = potential_config

        if not config_file:
            logger.warning(f"Could not find hydra config for {run_dir}. Skipping.")
            continue

        # Check for predictions
        pred_dir = os.path.join(run_dir, "predictions")
        if not args.force and os.path.exists(os.path.join(pred_dir, "results.csv")):
            logger.info(f"Skipping {run_dir} - Predictions already exist.")
            continue

        # Check for checkpoints
        # Config usually has output_dir, but we are *in* the output dir effectively
        # Checkpoints usually in {run_dir}/checkpoints
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        best_ckpt = find_best_checkpoint(checkpoint_dir)

        if not best_ckpt:
            logger.warning(f"No valid checkpoints found in {checkpoint_dir}. Skipping.")
            continue

        if args.dry_run:
            logger.info(f"[Dry Run] Would predict for {run_dir} using {best_ckpt}")
            continue

        # Load Config
        try:
            cfg = OmegaConf.load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {e}")
            continue

        # Setup GPUs
        gpus = None
        if torch.cuda.is_available():
            # If requesting N gpus, we create a list [0, 1, ... N-1]
            # Assumes visible devices are set correctly by SLURM
            gpus = list(range(min(torch.cuda.device_count(), args.n_gpus)))

        try:
            run_prediction(run_dir, cfg, best_ckpt, gpus, args.use_test_split)
        except Exception as e:
            logger.error(f"Failed to run prediction for {run_dir}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
