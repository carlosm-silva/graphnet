"""
This file contains utility functions for the icemix model training.
"""

from typing import List, Optional
import pandas as pd
from graphnet.data.constants import FEATURES, TRUTH
from pytorch_lightning import Callback
import torch
import torch.distributed as dist
import os
import glob
import re
import numpy as np


# Auxiliary functions
def load_list_from_csv(csv_file_path: str) -> List[int]:
    """Load event numbers from a CSV file.

    Args:
        csv_file_path: Path to the CSV file containing event numbers.

    Returns:
        List of event numbers as integers.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        KeyError: If the 'event_no' column is not found in the CSV.
        ValueError: If the CSV file cannot be parsed or contains invalid data.
    """
    df = pd.read_csv(csv_file_path, dtype={"event_no": int})
    event_list = df["event_no"].tolist()
    return event_list


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the checkpoint file with the smallest validation loss.

    Args:
        checkpoint_dir: Directory containing checkpoint files

    Returns:
        Path to the best checkpoint file, or None if no valid checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    # Pattern to match checkpoint files: best-epoch=X-val_loss=Y.ckpt
    pattern = os.path.join(checkpoint_dir, "best-epoch=*-val_loss=*.ckpt")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        return None

    best_loss = float("inf")
    best_checkpoint = None

    # Extract validation loss from filename and find the minimum
    for checkpoint_path in checkpoint_files:
        filename = os.path.basename(checkpoint_path)
        # Use regex to extract val_loss value
        match = re.search(r"val_loss=([0-9]+\.?[0-9]*)", filename)
        if match:
            val_loss = float(match.group(1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_path

    return best_checkpoint


class CheckSamplerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if (
            trainer.train_dataloader is not None
            and trainer.train_dataloader.sampler is not None
        ):
            sampler = trainer.train_dataloader.sampler
            # get this rank's indices
            local_indices = list(iter(sampler))
            # print first & last few to verify non-overlap
            print(
                f"[PID {os.getpid():5d}] ▶ rank={dist.get_rank() if dist.is_initialized() else 0}/"
                f"{dist.get_world_size() if dist.is_initialized() else 1}  "
                f"sample indices head: {local_indices[:5]}  tail: {local_indices[-5:]}"
            )
        else:
            print(
                "Warning: train_dataloader or sampler is None in CheckSamplerCallback"
            )


class EpochMonitorCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if (
            trainer.train_dataloader is not None
            and trainer.train_dataloader.sampler is not None
        ):
            sampler = trainer.train_dataloader.sampler
            local_indices = list(iter(sampler))

            if dist.is_available() and dist.is_initialized():
                # monitor rank, world_size, per-rank batches, total batches
                w = dist.get_world_size()
                r = dist.get_rank()
                per_rank = len(local_indices)
                total = per_rank * w
                dev = torch.cuda.current_device()
                print(
                    f"[PID {os.getpid():5d}] ▶ rank={r}/{w}  device={dev}  "
                    f"batches_per_rank={per_rank}  total_batches≈{total}"
                )

                # gather all indices from each rank
                all_indices = [None] * w
                dist.all_gather_object(all_indices, local_indices)
                if r == 0:
                    # verify full coverage without overlap
                    merged = []
                    for sub in all_indices:
                        if isinstance(sub, list):
                            merged.extend(sub)
                    merged = sorted(set(merged))
                    dataset_size = len(sampler.dataset)
                    assert (
                        len(merged) == dataset_size
                    ), f"Sample coverage error: got {len(merged)}/{dataset_size}"
                    print(f"[Rank 0] ✔ all {dataset_size} samples covered by {w} ranks")
            else:
                # single-process mode
                per_rank = len(local_indices)
                print(
                    f"[PID {os.getpid():5d}] ▶ single-process mode  total_batches={per_rank}"
                )
        else:
            print(
                "Warning: train_dataloader or sampler is None in EpochMonitorCallback"
            )


class TokenDropSeedCallback(Callback):
    """Keep token-drop masks stable across repeated optimizer closures."""

    _MAX_SEED = 2**63 - 1

    def __init__(self, seed: int) -> None:
        super().__init__()
        self._seed = seed

    def _batch_seed(self, epoch: int, batch_idx: int, rank: int) -> int:
        return (
            self._seed
            + epoch * 1_000_000_007
            + batch_idx * 1_000_003
            + rank * 10_007
        ) % self._MAX_SEED

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        rank = int(getattr(trainer, "global_rank", 0))
        seed = self._batch_seed(trainer.current_epoch, batch_idx, rank)
        pl_module.backbone.set_token_drop_seed(seed)


class RandomRotationCallback(Callback):
    """
    Applies a random 2D rotation in the xy-plane to each event in the batch.
    Rotates DOM positions (x, y) and truth labels (position_x, position_y, azimuth).
    """
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.generator = torch.Generator(device='cpu')
        if seed is not None:
            self.generator.manual_seed(seed)
        else:
            self.generator.seed()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._apply_rotation(batch)

    def _apply_rotation(self, batch):
        device = batch.x.device

        # Get number of unique graphs in the batch (batch_size)
        num_graphs = batch.batch.max().item() + 1

        # Generate a random angle for each graph in [0, 2pi) using CPU generator,
        # then move to the batch's device to ensure determinism/reproducibility across devices.
        angles = torch.rand(num_graphs, generator=self.generator, dtype=torch.float32) * 2 * np.pi
        angles = angles.to(device)

        # 1. Rotate node features (batch.x)
        # Assuming features: ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area']
        # -> x is at index 0, y is at index 1
        x = batch.x[:, 0]
        y = batch.x[:, 1]

        # Expand angles to per-node basis
        node_angles = angles[batch.batch]
        cos_a = torch.cos(node_angles)
        sin_a = torch.sin(node_angles)

        # Apply 2D rotation to node positions
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a

        batch.x[:, 0] = new_x
        batch.x[:, 1] = new_y

        # 2. Rotate truth labels (batch.joint_labels)
        if hasattr(batch, 'joint_labels') and batch.joint_labels is not None:
            # joint_labels is [batch_size, 6] -> [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
            pos_x = batch.joint_labels[:, 0]
            pos_y = batch.joint_labels[:, 1]

            dir_x = batch.joint_labels[:, 3]
            dir_y = batch.joint_labels[:, 4]

            cos_graph = torch.cos(angles)
            sin_graph = torch.sin(angles)

            # Rotate positions
            new_pos_x = pos_x * cos_graph - pos_y * sin_graph
            new_pos_y = pos_x * sin_graph + pos_y * cos_graph

            # Rotate direction vectors
            new_dir_x = dir_x * cos_graph - dir_y * sin_graph
            new_dir_y = dir_x * sin_graph + dir_y * cos_graph

            batch.joint_labels[:, 0] = new_pos_x
            batch.joint_labels[:, 1] = new_pos_y
            batch.joint_labels[:, 3] = new_dir_x
            batch.joint_labels[:, 4] = new_dir_y

        if hasattr(batch, 'azimuth') and batch.azimuth is not None:
            batch.azimuth = (batch.azimuth + angles) % (2 * np.pi)

        if hasattr(batch, 'position_x') and hasattr(batch, 'position_y'):
            if batch.position_x is not None and batch.position_y is not None:
                px = batch.position_x
                py = batch.position_y
                batch.position_x = px * cos_graph - py * sin_graph
                batch.position_y = px * sin_graph + py * cos_graph


# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")


def load_csv_splits(
    data_paths: List[str],
    train_csvs: List[str],
    val_csvs: List[str],
    test_csvs: Optional[List[str]] = None,
):
    """Load train/val/test event selections from per-dataset CSVs (one CSV
    per entry in ``data_paths``, matched positionally). ``test_csvs=None``
    returns a list of ``None``s, which signals "no test data" downstream."""
    n = len(data_paths)
    if len(train_csvs) != n or len(val_csvs) != n:
        raise ValueError(
            f"train_csvs ({len(train_csvs)}) and val_csvs ({len(val_csvs)}) "
            f"must both have length {n} to match data_paths."
        )
    if test_csvs is not None and len(test_csvs) != n:
        raise ValueError(
            f"test_csvs ({len(test_csvs)}) must have length {n} to match data_paths."
        )

    train_selections = [load_list_from_csv(p) for p in train_csvs]
    val_selections = [load_list_from_csv(p) for p in val_csvs]
    test_selections = (
        [load_list_from_csv(p) for p in test_csvs] if test_csvs else [None] * n
    )
    return train_selections, val_selections, test_selections


def get_dynamic_splits(
    data_paths: List[str],
    seed: int = 42,
    split_ratio: Optional[List[float]] = None,
):
    """
    Generate dynamic train, validation, and test splits for the given SQLite databases.

    Args:
        data_paths: List of file paths to the SQLite databases.
        seed: Random seed for deterministic shuffling.
        split_ratio: List containing [train_fraction, val_fraction, test_fraction] which should sum to 1.0.

    Returns:
        tuple: (train_selections, val_selections, test_selections), where each is a list of lists of event numbers.
    """
    import sqlite3
    import random
    import pandas as pd

    if split_ratio is None:
        split_ratio = [0.8, 0.1, 0.1]

    train_selections = []
    val_selections = []
    test_selections = []

    for db_path in data_paths:
        file_name = os.path.basename(db_path).lower()
        print(f"Generating dynamic splits for {file_name} with seed {seed} and ratio {split_ratio}...")
        try:
            with sqlite3.connect(db_path) as conn:
                # Using pandas read_sql_query is much faster than cursor.fetchall() for millions of rows
                df = pd.read_sql_query("SELECT event_no FROM truth", conn)
                events = df['event_no'].tolist()
        except Exception as e:
            print(f"Error reading from {db_path}: {e}")
            train_selections.append([])
            val_selections.append([])
            test_selections.append([])
            continue

        # Deterministically shuffle
        rnd = random.Random(seed)
        rnd.shuffle(events)

        n_events = len(events)
        n_train = int(n_events * split_ratio[0])
        n_val = int(n_events * split_ratio[1])
        # test takes the rest

        train_events = events[:n_train]
        val_events = events[n_train:n_train+n_val]
        test_events = events[n_train+n_val:]

        train_selections.append(train_events)
        val_selections.append(val_events)
        test_selections.append(test_events)

    return train_selections, val_selections, test_selections
