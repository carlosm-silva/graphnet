"""Inspect GraphNeT joint-label layout with a historical database."""

import sys
import torch
import numpy as np
import pytorch_lightning as pl
from graphnet.data.dataset import SQLiteDataset
from graphnet.training.labels import JointLabel
from src.utils import get_dynamic_splits
from train import GraphNeTDataModulecustom
import time

data_paths = ['/storage/home/hcoda1/4/jliao74/r-itaboada3-0/jliao74/my_numu_database_part_1 (1).db']
train, val, test = get_dynamic_splits(data_paths, 42, [0.8, 0.1, 0.1])

# Just take a subset of 10 events for testing
train_subset = [train[0][:10]]

datamodule = GraphNeTDataModulecustom(
    dataset_reference=SQLiteDataset,
    dataset_args={
        "path": data_paths,
        "features": ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area'],
        "truth": ['position_x', 'position_y', 'position_z', 'azimuth', 'zenith'],
    },
    train_selections=train_subset,
    val_selections=train_subset,
    test_selection=train_subset,
    train_dataloader_kwargs={"batch_size": 2, "num_workers": 1, "shuffle": False},
    validation_dataloader_kwargs={"batch_size": 2, "num_workers": 1},
    test_dataloader_kwargs={"batch_size": 2, "num_workers": 1},
    train_val_split=[0.9, 0.1],
    split_seed=42,
    labels={"joint_labels": JointLabel(key="joint_labels")}
)

datamodule.setup("fit")
train_loader = datamodule.train_dataloader()
batch = next(iter(train_loader))

print(f"Batch node shape: {batch.x.shape}")
print(f"Batch label shape: {batch.joint_labels.shape}")

# Print out the truth labels for one event to identify the columns
print("Truth dictionary mapping:")
for k, v in batch.to_dict().items():
    if type(v) == torch.Tensor and v.shape[0] == 2:
        print(f"{k}: {v}")
