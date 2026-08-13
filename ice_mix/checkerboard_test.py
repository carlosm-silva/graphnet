"""Evaluate IceMix checkpoints on complementary deterministic pulse halves.

The CLI reconstructs GraphNeT datasets and models from run metadata, applies a
seeded random partition by padded sequence position, and writes perturbed
prediction tables below each run. The historical ``checkerboard`` filename is a
misnomer retained for command and artifact compatibility: this code does not use
detector coordinates and must not be interpreted as a spatial-efficiency study.
"""

import os
import glob
import re
import argparse
import random
import torch
import torch.nn as nn
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

from src.utils import (
    get_configured_splits,
    select_evaluation_split,
    write_evaluation_manifest,
)

# We use this to monkey patch IceMix forward logic for a historical deterministic
# complementary half-partition without modifying the model definition.
_original_ice_mix_forward = IceMix.forward


def _checkerboard_forward(self, data):
    """Encode one seeded complementary half of each padded pulse sequence.

    Despite the historical function name, the selection is random by sequence
    position and independent of detector coordinates. For fixed seed and batch
    construction, halves 1 and 2 are disjoint and cover every valid pulse.
    """
    from graphnet.models.utils import array_to_sequence
    from torch_geometric.utils import to_dense_batch

    x0, mask, seq_length = array_to_sequence(data.x, data.batch, padding_value=0)
    x = self.fourier_ext(x0, seq_length)
    rel_pos_bias = self.rel_pos(x0)
    batch_size = mask.shape[0]

    if self.include_dynedge:
        graph = self.dyn_edge(data)
        graph, _ = to_dense_batch(graph, data.batch)
        x = torch.cat([x, graph], 2)

    # Historical "checkerboard" patch: deterministic random complementary halves.
    if hasattr(self, "checkerboard_half"):
        B, N = mask.shape
        device = mask.device

        # We need a random but deterministic disjoint split.
        # We'll use a local random generator seeded with 42 to shuffle indices.
        # To do this per-batch in a compatible way, we can construct drop masks using random numbers
        # but maintaining the same seed.

        gen = torch.Generator(device=device)
        gen.manual_seed(getattr(self, "complementary_half_seed", 42))

        # Generate a unified random permutation score for tokens to partition them
        rand_scores = torch.rand(mask.shape, generator=gen, device=device)

        # To make it exactly 50/50, find the median rank threshold per sequence length (approx by N/2)
        # Actually, mask indicates valid sequence length. We should partition the *valid* tokens per batch item.
        valid_counts = mask.sum(dim=1)  # shape (B,)

        # We want half the valid tokens.
        # An easy way to partition exactly in half is to assign random scores to invalid tokens as Inf,
        # so they stay at the end.
        scores_for_partition = rand_scores.clone()
        scores_for_partition[~mask] = float("inf")

        ranks = torch.argsort(scores_for_partition, dim=1)  # indices of sorted elements
        # Create an empty mask of what's selected
        selection_mask = torch.zeros_like(mask, dtype=torch.bool)

        for b in range(B):
            v_len = valid_counts[b].item()
            half_len = v_len // 2

            # The indices for the first half
            half1_indices = ranks[b, :half_len]
            # The indices for the second half
            half2_indices = ranks[b, half_len:v_len]

            if self.checkerboard_half == 1:
                selection_mask[b, half1_indices] = True
            elif self.checkerboard_half == 2:
                selection_mask[b, half2_indices] = True

        mask = mask & selection_mask

    attn_mask = torch.zeros(mask.shape, device=mask.device)
    attn_mask[~mask] = -torch.inf

    for i, blk in enumerate(self.sandwich):
        x = blk(x, attn_mask, rel_pos_bias)
        if i + 1 == self.n_rel:
            rel_pos_bias = None

    mask = torch.cat(
        [
            torch.ones(batch_size, 1, dtype=mask.dtype, device=mask.device),
            mask,
        ],
        1,
    )
    attn_mask = torch.zeros(mask.shape, device=mask.device)
    attn_mask[~mask] = -torch.inf
    cls_token = self.cls_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
    x = torch.cat([cls_token, x], 1)

    for blk in self.blocks:
        x = blk(x, None, attn_mask)

    return x[:, 0]


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the best checkpoint below ``checkpoint_dir``, or ``None``.

    The smallest validation loss parsed from a filename wins; ``last.ckpt`` is
    retained only as a fallback.
    """
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
    test_fraction: float = 1.0,
    use_test_split: bool = False,
    seed: int = 42,
) -> None:
    """Run complementary-half inference and write artifacts below a run.

    ``cfg`` supplies data/model settings and ``ckpt_path`` supplies weights.
    ``test_fraction`` selects a seeded random subset of validation events, or test events
    when ``use_test_split`` is true; ``gpus`` is passed to GraphNeT. Two CSV
    files are written below ``run_dir``. The function returns ``None``.
    """
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
            max_pulses=256,
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
        token_drop=0.0,  # Disable standard token drop
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

    evaluated_selections = [list(selection) for selection in eval_selections]
    if test_fraction < 1.0:
        logger.info(
            f"Running in TEST mode: using {test_fraction * 100:.2f}% of validation data."
        )
        from torch.utils.data import Subset
        import numpy as np

        random_generator = np.random.default_rng(seed)

        loaders = (
            val_dataloader if isinstance(val_dataloader, list) else [val_dataloader]
        )
        if len(loaders) != len(evaluated_selections):
            raise RuntimeError(
                "Cannot map subsampled validation loaders back to per-database "
                "event selections for the evaluation manifest."
            )
        new_loaders = []
        for loader_index, loader in enumerate(loaders):
            ds = loader.dataset
            num_samples = max(1, int(len(ds) * test_fraction))
            indices = random_generator.choice(
                len(ds), num_samples, replace=False
            ).tolist()
            evaluated_selections[loader_index] = [
                evaluated_selections[loader_index][index] for index in indices
            ]
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

    output_path = os.path.join(run_dir, "checkerboard_results")
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    write_evaluation_manifest(
        os.path.join(output_path, "evaluation_manifest.json"),
        data_paths=data_paths,
        event_selections=evaluated_selections,
        partition=partition,
        checkpoint_path=ckpt_path,
        split_config=cfg.data.get("split", cfg.data),
        seed=seed,
        test_fraction=test_fraction,
        perturbation=(
            "deterministic random complementary half-split by padded sequence "
            "position; not spatial"
        ),
    )

    # Apply the historical monkey patch for complementary pulse halves.
    IceMix.forward = _checkerboard_forward

    try:
        for half in [1, 2]:
            logger.info(
                f"Starting complementary-half prediction loop for half {half} ..."
            )

            backbone.checkerboard_half = half
            backbone.complementary_half_seed = seed

            results = model.predict_as_dataframe(
                val_dataloader,
                additional_attributes=additional_attributes,
                prediction_columns=prediction_columns,
                gpus=gpus,
            )

            csv_name = f"checkerboard_half_{half}.csv"
            csv_path = os.path.join(output_path, csv_name)
            results.to_csv(csv_path)
            logger.info(f"Results for half {half} saved to {csv_path}")
    finally:
        # Restore original forward method just in case
        IceMix.forward = _original_ice_mix_forward


def main():
    """Parse CLI options, select a named run, and run checkerboard inference."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the historical 'checkerboard' evaluation, which is actually a "
            "seeded random complementary half-split of pulse sequence positions."
        )
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="The Hydra project_name (model config name) to test (e.g., IceMix-Drop-Augmented-Rotation)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="ice_mix/outputs",
        help="Base directory containing run outputs",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional exact run directory; avoids automatic best-run selection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected run and checkpoint without evaluating it.",
    )
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of GPUs to use")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for event subsampling and complementary pulse halves.",
    )
    args = parser.parse_args()

    logger = Logger()
    base_dir = args.base_dir
    if not os.path.isabs(base_dir):
        base_dir = os.path.abspath(base_dir)

    logger.info(
        f"Scanning {base_dir} for runs matching model config '{args.model_config}'..."
    )

    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist.")
        return

    if args.run_dir is not None:
        requested_run = os.path.abspath(args.run_dir)
        requested_config = os.path.join(requested_run, ".hydra", "config.yaml")
        if not os.path.isfile(requested_config):
            raise FileNotFoundError(
                f"Requested run has no .hydra/config.yaml: {requested_run}"
            )
        config_files = [requested_config]
    else:
        config_files = glob.glob(
            os.path.join(base_dir, "**", ".hydra", "config.yaml"), recursive=True
        )

    matching_runs = []

    for config_file in config_files:
        run_dir = os.path.dirname(os.path.dirname(config_file))

        try:
            cfg = OmegaConf.load(config_file)
        except Exception as e:
            logger.error(f"Failed to load config {config_file}: {e}")
            continue

        if cfg.get("project_name") == args.model_config:
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            best_ckpt = find_best_checkpoint(checkpoint_dir)

            if best_ckpt:
                # Find its validation loss so we can pick the absolute best one
                filename = os.path.basename(best_ckpt)
                match = re.search(r"val_loss=([0-9]+\.?[0-9]*)", filename)
                loss = float("inf")
                if match:
                    loss = float(match.group(1))
                matching_runs.append((run_dir, cfg, best_ckpt, loss))

    if not matching_runs:
        logger.error(
            f"No valid checkpoints found for model config '{args.model_config}'."
        )
        return

    # Sort runs by loss (lowest first)
    matching_runs.sort(key=lambda x: x[3])

    best_run = matching_runs[0]
    best_run_dir, best_cfg, best_ckpt, best_loss = best_run

    logger.info(f"Selected best run for '{args.model_config}': {best_run_dir}")
    logger.info(f"Best Validation Loss: {best_loss}")
    logger.info(f"Checkpoint Path: {best_ckpt}")

    if args.dry_run:
        logger.info("Dry run complete; no evaluation artifacts were written.")
        return

    # Check if results exist
    output_path = os.path.join(best_run_dir, "checkerboard_results")
    if (
        os.path.exists(output_path)
        and os.path.exists(os.path.join(output_path, "checkerboard_half_1.csv"))
        and os.path.exists(os.path.join(output_path, "checkerboard_half_2.csv"))
    ):
        logger.info(
            f"Skipping {best_run_dir} - All checkerboard predictions already exist."
        )
        return

    gpus = None
    if torch.cuda.is_available():
        gpus = list(range(min(torch.cuda.device_count(), args.n_gpus)))

    try:
        run_prediction(
            best_run_dir,
            best_cfg,
            best_ckpt,
            gpus,
            args.test_fraction,
            args.use_test_split,
            args.seed,
        )
    except Exception as e:
        logger.error(f"Failed to run checkerboard prediction for {best_run_dir}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
