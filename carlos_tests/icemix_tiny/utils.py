"""
This file contains utility functions for the icemix model training.
"""

from typing import List
import pandas as pd
from graphnet.data.constants import FEATURES, TRUTH
from pytorch_lightning import Callback
import torch
import torch.distributed as dist
import os
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


class CheckSamplerCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.train_dataloader is not None and trainer.train_dataloader.sampler is not None:
            sampler = trainer.train_dataloader.sampler
            # get this rank's indices
            local_indices = list(iter(sampler))
            # print first & last few to verify non-overlap
            print(f"[PID {os.getpid():5d}] ▶ rank={dist.get_rank() if dist.is_initialized() else 0}/"
                  f"{dist.get_world_size() if dist.is_initialized() else 1}  "
                  f"sample indices head: {local_indices[:5]}  tail: {local_indices[-5:]}")
        else:
            print("Warning: train_dataloader or sampler is None in CheckSamplerCallback")

class EpochMonitorCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.train_dataloader is not None and trainer.train_dataloader.sampler is not None:
            sampler = trainer.train_dataloader.sampler
            local_indices = list(iter(sampler))

            if dist.is_available() and dist.is_initialized():
                # monitor rank, world_size, per-rank batches, total batches
                w = dist.get_world_size()
                r = dist.get_rank()
                per_rank = len(local_indices)
                total = per_rank * w
                dev = torch.cuda.current_device()
                print(f"[PID {os.getpid():5d}] ▶ rank={r}/{w}  device={dev}  "
                      f"batches_per_rank={per_rank}  total_batches≈{total}")

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
                    assert len(merged) == dataset_size, (
                        f"Sample coverage error: got {len(merged)}/{dataset_size}"
                    )
                    print(f"[Rank 0] ✔ all {dataset_size} samples covered by {w} ranks")
            else:
                # single-process mode
                per_rank = len(local_indices)
                print(f"[PID {os.getpid():5d}] ▶ single-process mode  total_batches={per_rank}")
        else:
            print("Warning: train_dataloader or sampler is None in EpochMonitorCallback")

checkpoint_callback = ModelCheckpoint(
    dirpath="/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix_tiny/checkpoints",                  # Specify save directory
    filename="best-{epoch:02d}-{val_loss:.4f}",  # Filename format
    monitor="val_loss",                         # Monitor validation loss
    mode="min",
    save_top_k=1,                               # Save only the best checkpoint
    save_last=True                              # Also save the last checkpoint
)


progress_bar_callback = TQDMProgressBar()

custom_callbacks = [checkpoint_callback, progress_bar_callback, EpochMonitorCallback(), CheckSamplerCallback()]


# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")

NumuValidation = "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_validation_selection.csv"
NumuTraining = "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_training_selection.csv"
NueValidation = "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnue_validation_selection.csv"
NueTraining = "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnue_training_selection.csv"


# Load full selections
NuMu_Training_Selections_full = load_list_from_csv(NumuTraining)
NuMu_Validation_Selections_full = load_list_from_csv(NumuValidation)
NuE_Training_Selections_full = load_list_from_csv(NueTraining)
NuE_Validation_Selections_full = load_list_from_csv(NueValidation)

# For quick testing, use only 10% of the data
# Set this to False to use full dataset
USE_10_PERCENT = False  # Changed to False for L40S run

if USE_10_PERCENT:
    # Take first 10% of each selection
    NuMu_Training_Selections = NuMu_Training_Selections_full[:len(NuMu_Training_Selections_full)//10]
    NuMu_Validation_Selections = NuMu_Validation_Selections_full[:len(NuMu_Validation_Selections_full)//10]
    NuE_Training_Selections = NuE_Training_Selections_full[:len(NuE_Training_Selections_full)//10]
    NuE_Validation_Selections = NuE_Validation_Selections_full[:len(NuE_Validation_Selections_full)//10]
    print(f"Using 10% of dataset: {len(NuMu_Training_Selections)} numu training, {len(NuE_Training_Selections)} nue training events")
else:
    # Use full dataset
    NuMu_Training_Selections = NuMu_Training_Selections_full
    NuMu_Validation_Selections = NuMu_Validation_Selections_full
    NuE_Training_Selections = NuE_Training_Selections_full
    NuE_Validation_Selections = NuE_Validation_Selections_full
    print(f"Using full dataset: {len(NuMu_Training_Selections)} numu training, {len(NuE_Training_Selections)} nue training events")
