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
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


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


# Callbacks
def get_callbacks(checkpoint_dir: str) -> List[Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    progress_bar_callback = TQDMProgressBar()
    return [
        checkpoint_callback,
        progress_bar_callback,
        EpochMonitorCallback(),
        CheckSamplerCallback(),
    ]


# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")

# Selection paths
NumuValidation = "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_numu_database_part_1_validation_selection.csv"
NumuTraining = "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_numu_database_part_1_training_selection.csv"
NueValidation = "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nue_database_part_1_validation_selection.csv"
NueTraining = "/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/Divided_training/data/P14_nugen_nue_database_part_1_training_selection.csv"

# Pre-load selections to be available for import
# Note: Ideally these should be loaded on demand or configured, but for compatibility we load them here.
try:
    NuMu_Training_Selections = load_list_from_csv(NumuTraining)
    NuMu_Validation_Selections = load_list_from_csv(NumuValidation)
    NuE_Training_Selections = load_list_from_csv(NueTraining)
    NuE_Validation_Selections = load_list_from_csv(NueValidation)
except FileNotFoundError as e:
    print(f"Warning: Could not load selection files: {e}")
    NuMu_Training_Selections = []
    NuMu_Validation_Selections = []
    NuE_Training_Selections = []
    NuE_Validation_Selections = []
