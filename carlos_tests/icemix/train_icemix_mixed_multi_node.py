from graphnet.utilities.argparse import ArgumentParser
from typing import Optional, List, Dict, Any, cast
from graphnet.utilities.logging import Logger
import os
from pytorch_lightning.loggers import WandbLogger
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# Enable Tensor Core utilization for L40S GPUs
torch.set_float32_matmul_precision('high')

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
    ckpt_path: Optional[str] = None,
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
            "distribution_strategy": "ddp_find_unused_parameters_true",
            "num_nodes": int(os.environ.get("SLURM_NNODES", 1)),  # Auto-detect number of nodes
        },
    }

    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=256,
            z_name="sensor_pos_z",  # Likely wrong, but since `add_ice_properties` is False, it doesn't matter
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=features,
        columns=[0, 1, 2, 3],
    )

    archive = os.path.join(
        "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/graphnet/carlos_tests/icemix/",
        "results",
    )

    run_name = f"dynedgeTITO_{config['target']}_example"

    data_module = GraphNeTDataModulecustom(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth_table": truth_table,
            "pulsemaps": config["pulsemap"],
            "truth": truth,
            "features": features,
            "path": [
                "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/tango_data/my_numu_database_part_1 (1).db",
                "/storage/home/hcoda1/8/cfilho3/p-itaboada3-0/tango_data/my_nue_database_part_1 (1).db",
            ],
            "graph_definition": graph_definition,
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
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

    # Building model
    backbone = cast(
        DeepIce,
        DeepIce(
            hidden_dim=768,
            seq_length=256,
            depth=12,
            head_size=32,
            n_rel=4,
            scaled_emb=True,
            include_dynedge=False,
            n_features=len(features),
            maha_encoder=True,
        ),
    )

    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=JointLoss(
            alpha=0.01,
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(),
        ),
    )

    model = cast(StandardModel, StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 6, "factor": 0.5},
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    ))

    optim_conf = model.configure_optimizers()
    lr_conf = cast(Dict[str, Any], optim_conf.get("lr_scheduler", optim_conf.get("lr_schedulers")))
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
        precision="16-mixed",
        **config["fit"],
        callbacks=custom_callbacks,  # Re-enabled checkpoint callback
        ckpt_path=ckpt_path
    )

    
    # Load best checkpoint if available
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
        best_ckpt = checkpoint_callback.best_model_path
        print(f"*** Best ckpt: {best_ckpt} ***")
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
    else:
        print("*** No best checkpoint available, using current model state ***")
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
        gpus=[0],
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
        description="""Train GNN model without the use of config files."""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default="/storage/coda1/p-itaboada3/0/cfilho3/tango_data/my_numu_database_part_1 (1).db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="'SRTInIcePulses'",
    )

    parser.add_argument(
        "--target",
        help=("Name of feature to use as regression target (default: " "%(default)s)"),
        default="direction",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth",
    )

    parser.with_standard_arguments(
        ("gpus", list(range(torch.cuda.device_count()))),
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

    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from. If not set, training starts from scratch.",
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
        args.ckpt_path,
    ) 