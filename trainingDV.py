"""Example of training Model."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.gnn import DynEdgeTITO
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.reconstruction import DirectionReconstructionWithKappadecoupled
from graphnet.models.task.reconstruction import DirectionReconstructiondecoupledforKing
from graphnet.models.task.reconstruction import DirectionReconstructiondecoupledforNormal
from graphnet.models.task.reconstruction import JointPositionandDirectionReco
from graphnet.training.labels import Direction
from graphnet.training.labels import JointLabel
from graphnet.training.callbacks import ProgressBar
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.loss_functions import KingLossSoftClamp
from graphnet.training.loss_functions import NormalDistributionLoss
from graphnet.training.loss_functions import JointLoss
from graphnet.training.loss_functions import EuclideanDistanceLoss
from graphnet.training.utils import make_train_validation_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.data.datamodule import GraphNeTDataModule
import pandas as pd
import sqlite3
import numpy as np
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
import math
import random
import torch
import time



seed = int(time.time())
print(f"Random Seed is: {seed}")


random.seed(seed)


np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




checkpoint_callback = ModelCheckpoint(
    dirpath="/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/VertexReco/Vertex/LargeTC0.004_LRNEW/",                  # checkpoint save dir
    filename="best-{epoch:02d}-{val_loss:.4f}",   
    monitor="val_loss",                          
    mode="min",
    save_top_k=1,                               # save one best model
    save_last=True                              # save latest checkpoint
)



progress_bar_callback = ProgressBar()

custom_callbacks = [checkpoint_callback, progress_bar_callback]



# Constants
features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
truth.append("oneweight")

def load_list_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, dtype={'event_no': int}) 
    event_list = df['event_no'].tolist()
    return event_list


NumuValidation = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_validation_selection.csv'
NumuTraining = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnumu_training_selection.csv'
NueValidation = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnue_validation_selection.csv'
NueTraining = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnue_training_selection.csv'
NutauValidation = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnutau_validation_selection.csv'
NutauTraining = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/resultstitoclassnutau_training_selection.csv'
NugenTraining = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/nugen_numu_20878_part_1_training_selection.csv'
NugenValidation = '/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/Divided_training/data/nugen_numu_20878_part_1_validation_selection.csv'


NuMu_Training_Selections = load_list_from_csv(NumuTraining)
NuMu_Validation_Selections = load_list_from_csv(NumuValidation)
NuE_Training_Selections = load_list_from_csv(NueTraining)
NuE_Validation_Selections = load_list_from_csv(NueValidation)
NuTau_Training_Selections = load_list_from_csv(NutauTraining)
NuTau_Validation_Selections = load_list_from_csv(NutauValidation)
NuGen_Training_Selections = load_list_from_csv(NugenTraining)
NuGen_Validation_Selections = load_list_from_csv(NugenValidation)


def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    graph_definition = KNNGraph(detector=IceCube86(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
        input_feature_names=features,)
   
    archive = os.path.join('/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/VertexReco/Vertex/LargeTC0.004_LRNEW', "Iit1") #Save results

    run_name = "dynedgeTITO_{}_example".format(config["target"])
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)



    data_module = GraphNeTDataModulecustom(dataset_reference=SQLiteDataset,
                                     dataset_args = {
                                    "truth_table": truth_table,
                                    "pulsemaps": config["pulsemap"],
                                    "truth": truth,
                                    "features": features,
                                    "path": ["/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_numu_database_part_1 (1).db", "/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_nue_database_part_1 (1).db"],
                                    "graph_definition": graph_definition},
                                    train_dataloader_kwargs={"batch_size": config["batch_size"],
                                                             "num_workers": config["num_workers"],
                                                             },
                                    train_selections=[NuMu_Training_Selections, NuE_Training_Selections
                                      
                                       ],
                                    val_selections=[NuMu_Validation_Selections, NuE_Validation_Selections],
                                    test_selection = [None,None],
                                    labels={
                                             "joint_labels": JointLabel(
                                                                 azimuth_key="azimuth", zenith_key="zenith",
                                                                 position_keys=("position_x", "position_y", "position_z"),
                                                                 key="joint_labels"
                                                                     )
                                            },
                                    train_val_split = [0.2, 0.8],
                                )
    
    training_dataloader = data_module.train_dataloader
    validation_dataloader = data_module.val_dataloader



    # Building model
    backbone = DynEdgeTITO(
        nb_inputs=graph_definition.nb_outputs,
        features_subset=[0, 1, 2, 3],
        dyntrans_layer_sizes=[(256, 256), (256, 256), (256, 256), (256, 256)],
        global_pooling_schemes=["max"],
        use_global_features=True,
        use_post_processing_layers=True,
    )
    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=JointLoss(alpha=0.004,position_loss=EuclideanDistanceLoss(),direction_loss=VonMisesFisher3DLoss()),
        )
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience":6,
            "factor":0.5
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )



    optim_conf = model.configure_optimizers()
    lr_conf = optim_conf.get("lr_scheduler", optim_conf.get("lr_schedulers"))
    scheduler = lr_conf["scheduler"]

    print("patience =", scheduler.patience)
    print("factor =", scheduler.factor)
    print("threshold =", scheduler.threshold)
    print("mode =", scheduler.mode)


    # Training model

    model.fit(
        training_dataloader,
        validation_dataloader,
        logger=wandb_logger if wandb else None,
        accumulate_grad_batches=10,
        **config["fit"],
        callbacks=custom_callbacks,
        ckpt_path=None
    )
   





    best_ckpt = checkpoint_callback.best_model_path
    print(f"*** Best ckpt: {best_ckpt} ***")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()


    # Get predictions
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

    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        validation_dataloader,
        additional_attributes=additional_attributes,
        prediction_columns=prediction_columns,
        gpus=[0], # 1 GPU for prediction
    )

    # Save predictions and model to file
    db_name = path.split("/")[-1].split(".")[0]
    path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {path}")
    os.makedirs(path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{path}/results.csv")

    # Save full model (including weights) to .pth file - Not version proof
    model.save(f"{path}/model.pth")

    # Save model config and state dict - Version safe save method.
    model.save_state_dict(f"{path}/state_dict.pth")
    model.save_config(f"{path}/model_config.yml")



if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model without the use of config files.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default="/storage/home/hcoda1/4/jliao74/p-itaboada3-0/jliao74/my_numu_database_part_1 (1).db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="'SRTInIcePulses'",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default="direction",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth",
    )

    parser.with_standard_arguments(
        ("gpus",[0,1]), #Using 2 GPUs for Training
        ("max-epochs", 1),
        ("early-stopping-patience", 2),
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args, unknown = parser.parse_known_args()



    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.wandb,
    )
