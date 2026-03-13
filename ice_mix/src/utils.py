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
        
        # Also update raw azimuth and position properties if they exist
        if hasattr(batch, 'azimuth') and batch.azimuth is not None:
            batch.azimuth = (batch.azimuth + angles) % (2 * np.pi)
            
        if hasattr(batch, 'position_x') and hasattr(batch, 'position_y'):
            if batch.position_x is not None and batch.position_y is not None:
                px = batch.position_x
                py = batch.position_y
                batch.position_x = px * cos_graph - py * sin_graph
                batch.position_y = px * sin_graph + py * cos_graph


# Callbacks
def get_callbacks(checkpoint_dir: str, augment_rotation: bool = False, rotation_seed: Optional[int] = None) -> List[Callback]:
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    progress_bar_callback = TQDMProgressBar()
    cb_list = [
        checkpoint_callback,
        progress_bar_callback,
        EpochMonitorCallback(),
        CheckSamplerCallback(),
    ]
    if augment_rotation:
        cb_list.append(RandomRotationCallback(seed=rotation_seed))
        
    return cb_list


# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")

# Selection paths (legacy, kept for backwards compatibility with augmented data)
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

def get_dynamic_splits(data_paths: List[str], seed: int = 42, split_ratio: List[float] = [0.8, 0.1, 0.1]):
    """
    Generate dynamic train, validation, and test splits for the given SQLite databases.
    If 'augmented' is in the dataset path, falls back to legacy CSV splits for that dataset.
    
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
    
    train_selections = []
    val_selections = []
    test_selections = []
    
    # Pre-calculated legacy splits mapping by looking for 'numu' or 'nue' in filename
    legacy_map = {
        'numu': (NuMu_Training_Selections, NuMu_Validation_Selections, []), # legacy didn't have test
        'nue': (NuE_Training_Selections, NuE_Validation_Selections, [])
    }
    
    for db_path in data_paths:
        file_name = os.path.basename(db_path).lower()
        
        # Fallback to legacy static splits for augmented datasets
        # Note: In the future, data will be augmented on the fly during training, so this fallback can eventually be removed.
        if "augmented" in file_name:
            print(f"Warning: 'augmented' found in dataset {file_name}. Falling back to legacy CSV splits.")
            matched = False
            for key, (tr, va, te) in legacy_map.items():
                if key in file_name:
                    train_selections.append(tr)
                    val_selections.append(va)
                    test_selections.append(te)
                    matched = True
                    break
            if not matched:
                print(f"Error: Could not match augmented dataset {file_name} to legacy 'numu' or 'nue' splits.")
                train_selections.append([])
                val_selections.append([])
                test_selections.append([])
            continue
            
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
