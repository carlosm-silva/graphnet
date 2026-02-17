from graphnet.utilities.argparse import ArgumentParser
from typing import Optional, List, Dict, Any, cast
from graphnet.utilities.logging import Logger
import os
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.loggers.logger import Logger as PLLogger
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import IceMixNodes
from graphnet.models.detector.icecube import IceCube86
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.training.labels import JointLabel
from graphnet.models.gnn import DeepIce
from graphnet.models.task.reconstruction import JointPositionandDirectionReco
from graphnet.training.loss_functions import (
    JointLoss,
    EuclideanDistanceLoss,
    VonMisesFisher3DLoss,
)
from graphnet.models import StandardModel
from torch.optim import AdamW, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

from typing import List
import pandas as pd
from graphnet.data.constants import FEATURES, TRUTH
from pytorch_lightning import Callback
import torch
import torch.distributed as dist
import os
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import re
import glob

# Import from utils.py
from utils import (
    load_list_from_csv,
    CheckSamplerCallback,
    EpochMonitorCallback,
    checkpoint_callback,
    progress_bar_callback,
    custom_callbacks,
    features,
    truth,
    NuMu_Training_Selections,
    NuMu_Validation_Selections,
    NuE_Training_Selections,
    NuE_Validation_Selections,
)


data_module = GraphNeTDataModulecustom(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth_table": "truth",
            "pulsemaps": config["pulsemap"],
            "truth": truth,
            "features": features,
            "path": data_paths,
            "graph_definition": graph_definition,
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": min(config["num_workers"], 4),  # Fewer workers but keep performance features
            "pin_memory": False,  # Keep for performance
            "persistent_workers": False,  # Keep for performance  
            "prefetch_factor": 2,  # Reduce prefetch to save memory
            "multiprocessing_context": "spawn",  # Use spawn instead of fork for better memory handling
        },
        train_selections=[
            NuMu_Training_Selections,
            NuE_Training_Selections,
            # , NuGen_Training_Selections[:10000]
        ],
        val_selections=[NuMu_Validation_Selections, NuE_Validation_Selections],
        test_selection=[None, None],
        labels={
            "joint_labels": JointLabel(
                azimuth_key="azimuth",
                zenith_key="zenith",
                position_keys=("position_x", "position_y", "position_z"),
                key="joint_labels",
            )
        },
        train_val_split=[0.2, 0.8],
    )

training_dataloader = data_module.train_dataloader
validation_dataloader = data_module.val_dataloader