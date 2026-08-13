"""Evaluate linear interpolation between two rotation-model checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Union

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.data import Data

from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graphs import GraphDefinition
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.task.reconstruction import JointPositionandDirectionReco
from graphnet.training.labels import JointLabel
from graphnet.training.loss_functions import (
    EuclideanDistanceLoss,
    JointLoss,
    VonMisesFisher3DLoss,
)

from src.models.transformer import IceMix
from src.utils import features, get_dynamic_splits, truth


Batch = Union[Data, List[Data]]
RESULT_FIELDS = (
    "lambda",
    "val_loss",
    "val_loss_direction",
    "val_loss_position",
    "position_error_m_mean",
    "angular_error_deg_mean",
    "kappa_mean",
    "num_events",
    "num_batches",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate weight interpolation on the rotation validation split."
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Optional batch limit for a smoke test; omit for the full validation split.",
    )
    return parser.parse_args()


def _load_config(batch_size: int, num_workers: int) -> DictConfig:
    config_dir = str((Path(__file__).resolve().parent / "conf").resolve())
    overrides = [
        "data=augmented_rotation",
        f"data.batch_size={batch_size}",
        f"num_workers={num_workers}",
        "attention.dropout=0.0",
        "attention.attn_drop=0.0",
        "attention.proj_drop=0.0",
        "attention.drop_path_rate=0.0",
    ]
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def _local_data_paths(cfg: DictConfig) -> List[str]:
    data_paths = [str(path) for path in cfg.data.path]
    local_data_dir = os.environ.get("LOCAL_DATA_DIR")
    if not local_data_dir:
        return data_paths

    local_paths = []
    for path in data_paths:
        candidate = os.path.join(local_data_dir, os.path.basename(path))
        local_paths.append(candidate if os.path.exists(candidate) else path)
    return local_paths


def _build_validation_loader(cfg: DictConfig):
    data_paths = _local_data_paths(cfg)
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

    split_ratio = list(cfg.data.split.ratio)
    train_selections, val_selections, _ = get_dynamic_splits(
        data_paths,
        seed=int(cfg.data.split.seed),
        split_ratio=split_ratio,
    )
    loader_kwargs = {
        "batch_size": int(cfg.data.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.data.pin_memory),
    }
    if cfg.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": bool(cfg.data.persistent_workers),
                "prefetch_factor": int(cfg.data.prefetch_factor),
                "multiprocessing_context": "spawn",
            }
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
        train_dataloader_kwargs=loader_kwargs,
        train_selections=train_selections,
        val_selections=val_selections,
        test_selection=[None] * len(data_paths),
        train_val_split=split_ratio[:2],
        split_seed=int(cfg.data.split.seed),
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth",
                zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels",
            )
        },
    )
    return graph_definition, data_module.val_dataloader


def _build_model(cfg: DictConfig, graph_definition: GraphDefinition) -> StandardModel:
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
    return StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": 1e-3},
    )


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no state_dict: {path}")
    return checkpoint


def _interpolate_state_dict(
    state_a: Mapping[str, Tensor], state_b: Mapping[str, Tensor], weight_b: float
) -> MutableMapping[str, Tensor]:
    if state_a.keys() != state_b.keys():
        missing_a = sorted(state_b.keys() - state_a.keys())
        missing_b = sorted(state_a.keys() - state_b.keys())
        raise ValueError(
            f"Checkpoint state keys differ; missing from A={missing_a}, "
            f"missing from B={missing_b}."
        )

    result: MutableMapping[str, Tensor] = {}
    for name, value_a in state_a.items():
        value_b = state_b[name]
        if value_a.shape != value_b.shape or value_a.dtype != value_b.dtype:
            raise ValueError(f"Incompatible tensor for {name}")
        if weight_b == 0.0:
            result[name] = value_a
        elif weight_b == 1.0:
            result[name] = value_b
        elif value_a.is_floating_point() or value_a.is_complex():
            result[name] = torch.lerp(value_a, value_b, weight_b)
        elif torch.equal(value_a, value_b):
            result[name] = value_a
        else:
            raise ValueError(f"Non-floating state differs for {name}")
    return result


def _move_batch(batch: Batch, device: torch.device) -> List[Data]:
    data = batch if isinstance(batch, list) else [batch]
    return [item.to(device, non_blocking=True) for item in data]


def _evaluate(
    model: StandardModel,
    loader: Iterable[Batch],
    alpha: float,
    device: torch.device,
    limit_batches: Optional[int],
) -> Dict[str, float]:
    position_loss_fn = EuclideanDistanceLoss()
    direction_loss_fn = VonMisesFisher3DLoss()
    totals = {
        "position": 0.0,
        "direction": 0.0,
        "angle": 0.0,
        "kappa": 0.0,
    }
    num_events = 0
    num_batches = 0

    model.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            data = _move_batch(batch, device)
            prediction = model(data)[0]
            target = torch.cat([item["joint_labels"] for item in data], dim=0)
            if target.dim() == 3:
                target = target.squeeze(1)

            position = position_loss_fn(
                prediction[:, :3], target[:, :3], return_elements=True
            )
            direction = direction_loss_fn(
                prediction[:, 3:7], target[:, 3:6], return_elements=True
            )
            direction_cosine = (
                prediction[:, 3:6] * target[:, 3:6]
            ).sum(dim=1).clamp(-1.0, 1.0)
            angle = torch.rad2deg(torch.arccos(direction_cosine))

            tensors = (prediction, target, position, direction, angle)
            if not all(torch.isfinite(value).all().item() for value in tensors):
                raise FloatingPointError(
                    f"Non-finite evaluation value at validation batch {batch_idx}."
                )

            batch_events = int(target.shape[0])
            totals["position"] += float(position.sum().item())
            totals["direction"] += float(direction.sum().item())
            totals["angle"] += float(angle.sum().item())
            totals["kappa"] += float(prediction[:, 6].sum().item())
            num_events += batch_events
            num_batches += 1

            if num_batches % 500 == 0:
                print(
                    f"  evaluated {num_batches} batches / {num_events} events",
                    flush=True,
                )

    if num_events == 0:
        raise RuntimeError("Validation loader produced no events.")

    position_mean = totals["position"] / num_events
    direction_mean = totals["direction"] / num_events
    return {
        "val_loss": alpha * position_mean + direction_mean,
        "val_loss_direction": direction_mean,
        "val_loss_position": alpha * position_mean,
        "position_error_m_mean": position_mean,
        "angular_error_deg_mean": totals["angle"] / num_events,
        "kappa_mean": totals["kappa"] / num_events,
        "num_events": num_events,
        "num_batches": num_batches,
    }


def _write_results(path: Path, results: List[Dict[str, float]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This full-validation test requires a CUDA GPU.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative.")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in args.lambdas):
        raise ValueError("All interpolation lambdas must be finite and in [0, 1].")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_a = _load_checkpoint(args.checkpoint_a)
    checkpoint_b = _load_checkpoint(args.checkpoint_b)
    state_a = checkpoint_a["state_dict"]
    state_b = checkpoint_b["state_dict"]

    cfg = _load_config(args.batch_size, args.num_workers)
    torch.manual_seed(int(cfg.seed))
    graph_definition, val_loader = _build_validation_loader(cfg)
    device = torch.device("cuda:0")
    model = _build_model(cfg, graph_definition).to(device)

    metadata = {
        "checkpoint_a": str(args.checkpoint_a.resolve()),
        "checkpoint_b": str(args.checkpoint_b.resolve()),
        "checkpoint_a_epoch": checkpoint_a.get("epoch"),
        "checkpoint_b_epoch": checkpoint_b.get("epoch"),
        "checkpoint_a_global_step": checkpoint_a.get("global_step"),
        "checkpoint_b_global_step": checkpoint_b.get("global_step"),
        "lambdas": args.lambdas,
        "alpha": float(cfg.alpha),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "limit_val_batches": args.limit_val_batches,
    }
    with (args.output_dir / "metadata.json").open("w") as output:
        json.dump(metadata, output, indent=2)

    results: List[Dict[str, float]] = []
    for weight_b in args.lambdas:
        print(f"Evaluating lambda={weight_b:.4f}", flush=True)
        interpolated = _interpolate_state_dict(state_a, state_b, weight_b)
        model.load_state_dict(interpolated, strict=True)
        metrics = _evaluate(
            model,
            val_loader,
            alpha=float(cfg.alpha),
            device=device,
            limit_batches=args.limit_val_batches,
        )
        row = {"lambda": weight_b, **metrics}
        results.append(row)
        _write_results(args.output_dir / "results.csv", results)
        print(json.dumps(row, sort_keys=True), flush=True)

    best = min(results, key=lambda row: row["val_loss"])
    print(f"Best interpolation: {json.dumps(best, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
