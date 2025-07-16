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
    EuclideanDistanceLoss,
    VonMisesFisher3DLoss,
)
# 🎯 NEW: Import SimplexMultiLoss instead of problematic JointLoss
from graphnet.training.simplex_loss import JointPositionDirectionSimplexLoss
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

# Enable Tensor Core utilization for L40S GPUs
torch.set_float32_matmul_precision('high')

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
    
    best_loss = float('inf')
    best_checkpoint = None
    
    # Extract validation loss from filename and find the minimum
    for checkpoint_path in checkpoint_files:
        filename = os.path.basename(checkpoint_path)
        # Use regex to extract val_loss value
        match = re.search(r'val_loss=([0-9]+\.?[0-9]*)', filename)
        if match:
            val_loss = float(match.group(1))
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = checkpoint_path
    
    return best_checkpoint

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
    pin_memory: bool = False,
    persistent_workers: bool = False,
    accumulate_grad_batches: int = 1,
    mode: str = "train",
    # 🎯 NEW: SimplexMultiLoss parameters
    alpha: float = 0.3,
    balance_method: str = "running_mean",
    momentum: float = 0.95,
) -> None:
    """Run training with SimplexMultiLoss for improved convergence."""
    # Construct Logger
    logger = Logger()
    
    logger.info("🎯 Using SimplexMultiLoss for improved joint training!")
    logger.info(f"   → Alpha (position weight): {alpha:.3f}")
    logger.info(f"   → Direction weight: {1-alpha:.3f}")
    logger.info(f"   → Balance method: {balance_method}")
    logger.info(f"   → Momentum: {momentum}")

    # Setup CSV Logger with different name to distinguish from original
    csv_log_dir = "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/logs"
    os.makedirs(csv_log_dir, exist_ok=True)
    csv_logger = CSVLogger(
        save_dir=csv_log_dir,
        name="simplex_training_logs",  # Different name
        version=None
    )

    # Initialise Weights & Biases (W&B) run
    loggers: List[PLLogger] = [csv_logger]
    if wandb:
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="simplex-joint-reconstruction",  # Different project
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
            tags=["simplex-loss", "joint-reconstruction", "improved-convergence"]
        )
        loggers.append(wandb_logger)

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
        "alpha": alpha,
        "balance_method": balance_method,
        "momentum": momentum,
        "loss_type": "SimplexMultiLoss",
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    graph_definition = KNNGraph(
        detector=IceCube86(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=128,
            z_name="sensor_pos_z",
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=features,
        columns=[0, 1, 2, 3],
    )

    archive = os.path.join(
        "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/",
        "results_simplex",  # Different results directory
    )

    run_name = f"dynedgeTITO_simplex_{config['target']}_alpha{alpha}"

    # Use local NVMe data if available, otherwise fallback to network storage
    local_data_dir = os.environ.get('LOCAL_DATA_DIR')
    if local_data_dir and os.path.exists(local_data_dir):
        logger.info(f"Using local NVMe data from {local_data_dir}")
        data_paths = [
            os.path.join(local_data_dir, "my_numu_database_part_1 (1).db"),
            os.path.join(local_data_dir, "my_nue_database_part_1 (1).db"),
        ]
    else:
        logger.info("Using network storage data")
        data_paths = [
            "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/tango_data/my_numu_database_part_1 (1).db",
            "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/tango_data/my_nue_database_part_1 (1).db",
        ]

    data_module = GraphNeTDataModulecustom(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth_table": truth_table,
            "pulsemaps": config["pulsemap"],
            "truth": truth,
            "features": features,
            "path": data_paths,
            "graph_definition": graph_definition,
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
            "prefetch_factor": 4,
        },
        train_selections=[
            NuMu_Training_Selections,
            NuE_Training_Selections,
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

    # Building model (same as original)
    backbone = cast(
        DeepIce,
        DeepIce(
            hidden_dim=384,
            seq_length=128,
            depth=12,
            head_size=32,
            n_rel=4,
            scaled_emb=True,
            include_dynedge=False,
            n_features=len(features),
            maha_encoder=False,
            dropout=0.1,
            attn_drop=0.05,
            proj_drop=0.1,
            drop_path_rate=0.2,
        ),
    )

    # 🎯 KEY CHANGE: Use SimplexMultiLoss instead of problematic JointLoss
    logger.info("Creating SimplexMultiLoss for balanced joint training...")
    simplex_loss = JointPositionDirectionSimplexLoss(
        position_loss=EuclideanDistanceLoss(),
        direction_loss=VonMisesFisher3DLoss(),
        alpha=alpha,  # Much cleaner interpretation than alpha=0.04
        balance_method=balance_method,
        momentum=momentum,
    )
    
    logger.info(f"✅ SimplexMultiLoss created with weights: {simplex_loss.simplex_weights}")

    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=simplex_loss,  # Using SimplexMultiLoss
    )

    model = cast(StandardModel, StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": 1e-03/16, "eps": 1e-05},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={"patience": 6, "factor": 0.5},
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    ))

    if mode == "train":
        logger.info("Starting training mode with SimplexMultiLoss...")
        
        # Use different checkpoint directory for simplex version
        if ckpt_path is None:
            checkpoint_dir = "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/checkpoints_simplex"
            
            best_ckpt_path = find_best_checkpoint(checkpoint_dir)
            if best_ckpt_path is not None:
                ckpt_path = best_ckpt_path
                logger.info(f"Auto-loading best checkpoint from: {ckpt_path}")
            else:
                auto_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
                if os.path.exists(auto_ckpt_path):
                    ckpt_path = auto_ckpt_path
                    logger.info(f"Auto-loading last checkpoint from: {ckpt_path}")
                else:
                    logger.info("No checkpoint found, starting training from scratch")

        optim_conf = model.configure_optimizers()
        lr_conf = cast(Dict[str, Any], optim_conf.get("lr_scheduler", optim_conf.get("lr_schedulers")))
        scheduler = lr_conf["scheduler"]

        print("patience =", scheduler.patience)
        print("factor =", scheduler.factor)
        print("threshold =", scheduler.threshold)
        print("mode =", scheduler.mode)

        # Log initial loss statistics
        logger.info("📊 Initial SimplexMultiLoss configuration:")
        stats = simplex_loss.get_loss_statistics()
        logger.info(f"   → Task weights: {dict(zip(stats['loss_names'], stats['simplex_weights']))}")
        logger.info(f"   → Balance method: {balance_method}")
        logger.info(f"   → Momentum: {momentum}")

        # Training model
        primary_logger = loggers[0] if not wandb else loggers[1]
        model.fit(
            training_dataloader,
            validation_dataloader,
            logger=primary_logger,
            accumulate_grad_batches=accumulate_grad_batches,
            precision="16-mixed",
            **config["fit"],
            callbacks=custom_callbacks,
            ckpt_path=ckpt_path
        )
        
        # Log final loss statistics
        logger.info("📊 Final SimplexMultiLoss statistics:")
        final_stats = simplex_loss.get_loss_statistics()
        logger.info(f"   → Final task weights: {dict(zip(final_stats['loss_names'], final_stats['simplex_weights']))}")
        logger.info(f"   → Running means: {dict(zip(final_stats['loss_names'], final_stats['running_means']))}")
        logger.info(f"   → Total updates: {final_stats['num_updates']}")
        
        logger.info("🎉 Training completed successfully with SimplexMultiLoss!")
        
    elif mode == "predict":
        # Prediction mode (same as original but with simplex checkpoint directory)
        logger.info("Starting prediction mode...")
        
        if ckpt_path is None:
            checkpoint_dir = "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix_tiny/checkpoints_simplex"
            best_ckpt_path = find_best_checkpoint(checkpoint_dir)
            
            if best_ckpt_path is not None:
                ckpt_path = best_ckpt_path
                logger.info(f"Using best checkpoint: {ckpt_path}")
            else:
                auto_ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")
                if os.path.exists(auto_ckpt_path):
                    ckpt_path = auto_ckpt_path
                    logger.info(f"Using last checkpoint: {ckpt_path}")
                else:
                    raise FileNotFoundError("No trained model checkpoint found.")
        
        # Load and run prediction (same as original)
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        additional_attributes = [
            "zenith", "azimuth", "position_x", "position_y", "position_z",
            "event_no", "energy", "pid", "interaction_type", "oneweight",
        ]
        prediction_columns = [
             "pos_x_pred", "pos_y_pred", "pos_z_pred",
             "dir_x_pred", "dir_y_pred", "dir_z_pred", "dir_kappa_pred",
        ]

        logger.info("Starting prediction...")
        results = model.predict_as_dataframe(
            validation_dataloader,
            additional_attributes=additional_attributes,
            prediction_columns=prediction_columns,
            gpus=gpus,
        )

        # Save results
        db_name = path.split("/")[-1].split(".")[0]
        output_path = os.path.join(archive, db_name, run_name)
        logger.info(f"Writing results to {output_path}")
        os.makedirs(output_path, exist_ok=True)

        results.to_csv(f"{output_path}/results.csv")
        model.save(f"{output_path}/model.pth")
        model.save_state_dict(f"{output_path}/state_dict.pth")
        model.save_config(f"{output_path}/model_config.yml")
        
        logger.info("Prediction completed successfully!")
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'predict'.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""Train GNN model with SimplexMultiLoss for improved convergence."""
    )

    # Standard arguments (same as original)
    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default="/storage/coda1/r-itaboada3/0/cfilho3/tango_data/my_numu_database_part_1 (1).db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="'SRTInIcePulses'",
    )

    parser.add_argument(
        "--target",
        help=("Name of feature to use as regression target (default: %(default)s)"),
        default="direction",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth",
    )

    # 🎯 NEW: SimplexMultiLoss specific arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Position weight in simplex loss, alpha ∈ [0,1] (default: %(default)s)",
    )

    parser.add_argument(
        "--balance-method",
        choices=["running_mean", "batch_mean", "none"],
        default="running_mean",
        help="Loss balancing method (default: %(default)s)",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.95,
        help="Momentum for running mean statistics (default: %(default)s)",
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
        help="Path to a checkpoint file to resume training from.",
    )

    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="If True, pin_memory is used for data loading.",
    )

    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="If True, persistent_workers is used for data loading.",
    )

    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "predict"],
        help="Mode of operation (default: %(default)s)",
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
        args.pin_memory,
        args.persistent_workers,
        args.accumulate_grad_batches,
        args.mode,
        args.alpha,
        args.balance_method,
        args.momentum,
    )