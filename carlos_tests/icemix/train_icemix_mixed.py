from graphnet.utilities.argparse import ArgumentParser
from typing import Optional, List, Dict, Any, cast, Tuple
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
import numpy as np

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

def custom_train_loop(
    model: StandardModel,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    epochs: int,
    stage: int,
    checkpoint_dir: str,
    swa_model: Optional[AveragedModel] = None,
    accumulate_grad_batches: int = 1,
) -> Tuple[str, float]:
    """
    Custom training loop with OneCycleLR scheduler and optional SWA support.
    
    Returns:
        Tuple of (best_checkpoint_path, best_val_loss)
    """
    logger = Logger()
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Stage {stage} - Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            loss_components = model(batch)
            loss = (
                sum(loss_components)
                if isinstance(loss_components, list)
                else loss_components
            )
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / accumulate_grad_batches
            
            # Backward pass
            loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Step the scheduler after each optimizer step
                scheduler.step()
                
                # Update SWA model if provided (only for stage 2+)
                if swa_model is not None:
                    swa_model.update_parameters(model)
            
            train_loss += loss.item() * accumulate_grad_batches
            train_steps += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{train_loss/train_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validation"):
                batch = batch.to(device)
                loss_components = model(batch)
                loss = (
                    sum(loss_components)
                    if isinstance(loss_components, list)
                    else loss_components
                )
                if loss.dim() > 0:
                    loss = loss.mean()
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        logger.info(f"Stage {stage} - Epoch {epoch+1}: Train Loss = {train_loss/train_steps:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"stage_{stage}_best-epoch={epoch+1}-val_loss={avg_val_loss:.4f}.ckpt"
            )
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'stage': stage,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            best_checkpoint_path = checkpoint_path
            logger.info(f"Saved new best checkpoint: {checkpoint_path}")
        
        model.train()
    
    return best_checkpoint_path, best_val_loss

def multi_stage_training(
    model: StandardModel,
    train_dataloader,
    val_dataloader,
    base_lr: float,
    checkpoint_dir: str,
    epochs_per_stage: List[int],
    stage_lr_factors: List[float],
    device,
    accumulate_grad_batches: int = 1,
) -> StandardModel:
    """
    Implements multi-stage training with OneCycleLR and SWA.
    
    Args:
        model: The model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        base_lr: Base learning rate (1e-4)
        checkpoint_dir: Directory to save checkpoints
        epochs_per_stage: List of epochs for each stage
        stage_lr_factors: Learning rate multipliers for each stage
        device: Device to train on
        accumulate_grad_batches: Gradient accumulation steps
    
    Returns:
        The final trained model (with SWA if applicable)
    """
    logger = Logger()
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Calculate batches per epoch
    batches_per_epoch = len(train_dataloader)
    
    # Initialize SWA model (will be used from stage 2 onwards)
    swa_model = None
    
    # Track best checkpoint across all stages
    overall_best_checkpoint = None
    overall_best_loss = float('inf')
    
    for stage_idx, (epochs, lr_factor) in enumerate(zip(epochs_per_stage, stage_lr_factors)):
        stage = stage_idx + 1
        max_lr = base_lr * lr_factor
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Stage {stage} - Max LR: {max_lr}, Epochs: {epochs}")
        logger.info(f"{'='*50}")
        
        # Load checkpoint from previous stage if not stage 1
        if stage > 1 and overall_best_checkpoint:
            logger.info(f"Loading checkpoint from previous stage: {overall_best_checkpoint}")
            checkpoint = torch.load(overall_best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            
            # Initialize SWA model starting from stage 2
            if stage == 2:
                logger.info("Initializing SWA model for stage 2+")
                swa_model = AveragedModel(model)
        
        # Create new optimizer for this stage
        optimizer = AdamW(model.parameters(), lr=max_lr/25.0, eps=1e-05)
        
        # Calculate total steps for this stage
        total_steps = epochs * batches_per_epoch // accumulate_grad_batches
        
        # Create OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            anneal_strategy='cos',
            pct_start=0.01,
            div_factor=25.0,
            final_div_factor=25.0
        )
        
        # Train for this stage
        best_checkpoint, best_loss = custom_train_loop(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=epochs,
            stage=stage,
            checkpoint_dir=checkpoint_dir,
            swa_model=swa_model if stage > 1 else None,
            accumulate_grad_batches=accumulate_grad_batches
        )
        
        # Update overall best
        if best_loss < overall_best_loss:
            overall_best_loss = best_loss
            overall_best_checkpoint = best_checkpoint
        
        logger.info(f"Stage {stage} completed. Best loss: {best_loss:.4f}")
    
    # After all stages, finalize SWA model if used
    if swa_model is not None:
        logger.info("Updating batch normalization statistics for SWA model...")
        swa_model.to(device)
        update_bn(train_dataloader, swa_model, device=device)
        
        # Save final SWA model
        swa_checkpoint_path = os.path.join(checkpoint_dir, "final_swa_model.ckpt")
        torch.save({
            'state_dict': swa_model.module.state_dict(),
            'swa_state_dict': swa_model.state_dict(),
        }, swa_checkpoint_path)
        logger.info(f"Saved final SWA model to: {swa_checkpoint_path}")
        
        # Return the SWA model's base model
        return swa_model.module
    else:
        return model

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
    num_stages: int = 5,
    stage_epochs: Optional[List[int]] = None,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Setup CSV Logger for easy plotting (always enabled)
    csv_log_dir = "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix/logs"
    os.makedirs(csv_log_dir, exist_ok=True)
    csv_logger = CSVLogger(
        save_dir=csv_log_dir,
        name="training_logs",
        version=None  # This will auto-increment version numbers
    )

    # Initialise Weights & Biases (W&B) run
    loggers: List[PLLogger] = [csv_logger]  # Start with CSV logger
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
        # Add W&B logger to the list
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
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
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
        "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix/",
        "results",
    )

    run_name = f"dynedgeTITO_{config['target']}_example"

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
            "prefetch_factor": 4,  # Increase prefetch buffer for L40S
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
            seq_length=512,
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

    task = JointPositionandDirectionReco(
        hidden_size=backbone.nb_outputs,
        target_labels=["joint_labels"],
        loss_function=JointLoss(
            alpha=0,
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(),
        ),
    )

    model = cast(StandardModel, StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=AdamW,
        optimizer_kwargs={"lr": 1e-04, "eps": 1e-05},  # Base learning rate for stage 1
        scheduler_class=None,  # We'll handle scheduling manually
        scheduler_kwargs={},
        scheduler_config={},
    ))

    if mode == "train":
        logger.info("Starting training mode...")
        
        # Setup device
        if gpus:
            device = torch.device(f"cuda:{gpus[0]}")
        else:
            device = torch.device("cpu")
        
        # Define training stages configuration
        base_lr = 1e-4  # Base learning rate
        
        # Define epochs per stage
        if stage_epochs is not None:
            # Use custom epochs per stage
            if len(stage_epochs) != num_stages:
                raise ValueError(f"Length of stage_epochs ({len(stage_epochs)}) must match num_stages ({num_stages})")
            epochs_per_stage = stage_epochs
        else:
            # Default to 8 epochs per stage, but respect max_epochs if it's smaller
            default_epochs_per_stage = 8
            if max_epochs < default_epochs_per_stage * num_stages:
                # If max_epochs is too small, divide equally
                epochs_per_stage = [max_epochs // num_stages] * num_stages
                remaining_epochs = max_epochs - sum(epochs_per_stage)
                if remaining_epochs > 0:
                    epochs_per_stage[-1] += remaining_epochs
            else:
                # Use default 8 epochs per stage
                epochs_per_stage = [default_epochs_per_stage] * num_stages
        
        # Define learning rate multipliers based on number of stages
        if num_stages == 5:
            # Original 5-stage configuration
            stage_lr_factors = [
                1.0,      # Stage 1: 1e-4
                0.1,      # Stage 2: 1e-5
                0.05,     # Stage 3: 0.5e-5
                0.035,    # Stage 4: 0.35e-5
                0.01,     # Stage 5: 1e-6
            ]
        else:
            # For custom number of stages, use exponential decay
            stage_lr_factors = []
            for i in range(num_stages):
                # Exponentially decay from 1.0 to 0.01
                factor = 1.0 * (0.01 ** (i / (num_stages - 1)))
                stage_lr_factors.append(factor)
        
        logger.info(f"Multi-stage training configuration:")
        logger.info(f"Number of stages: {num_stages}")
        logger.info(f"Epochs per stage: {epochs_per_stage}")
        logger.info(f"Learning rate factors: {[f'{f:.3e}' for f in stage_lr_factors]}")
        
        checkpoint_dir = "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix/checkpoints"
        
        # Perform multi-stage training
        trained_model = multi_stage_training(
            model=model,
            train_dataloader=training_dataloader,
            val_dataloader=validation_dataloader,
            base_lr=base_lr,
            checkpoint_dir=checkpoint_dir,
            epochs_per_stage=epochs_per_stage,
            stage_lr_factors=stage_lr_factors,
            device=device,
            accumulate_grad_batches=accumulate_grad_batches
        )
        
        # Update the model reference to the trained model (with SWA if applicable)
        model = trained_model
        
        logger.info("Multi-stage training completed successfully!")
        
    elif mode == "predict":
        logger.info("Starting prediction mode...")
        
        # Setup device
        if gpus:
            device = torch.device(f"cuda:{gpus[0]}")
        else:
            device = torch.device("cpu")
        
        # For prediction mode, we need a trained model checkpoint
        checkpoint_dir = "/storage/home/hcoda1/8/cfilho3/r-itaboada3-0/graphnet/carlos_tests/icemix/checkpoints"
        
        if ckpt_path is None:
            # First, check for SWA model
            swa_checkpoint_path = os.path.join(checkpoint_dir, "final_swa_model.ckpt")
            if os.path.exists(swa_checkpoint_path):
                ckpt_path = swa_checkpoint_path
                logger.info(f"Using SWA checkpoint: {ckpt_path}")
            else:
                # Try to find the best regular checkpoint
                best_ckpt_path = find_best_checkpoint(checkpoint_dir)
                
                if best_ckpt_path is not None:
                    ckpt_path = best_ckpt_path
                    logger.info(f"Using best checkpoint: {ckpt_path}")
                else:
                    raise FileNotFoundError("No trained model checkpoint found. Please provide --ckpt-path or train a model first.")
        
        # Load the trained model
        logger.info(f"Loading model from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # Check if it's an SWA checkpoint
        if "swa_state_dict" in ckpt:
            logger.info("Loading SWA model weights")
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt["state_dict"])
        
        model.to(device)
        model.eval()
        logger.info(f"Model loaded from checkpoint: {ckpt_path}")

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
            "n_pulses",
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

        logger.info("Starting prediction...")

        results = model.predict_as_dataframe(
            validation_dataloader,
            additional_attributes=additional_attributes,
            prediction_columns=prediction_columns,
            gpus=gpus,
        )

        logger.info("Prediction completed successfully!")

        # Save predictions and model to file
        db_name = path.split("/")[-1].split(".")[0]
        output_path = os.path.join(archive, db_name, run_name)
        logger.info(f"Writing results to {output_path}")
        os.makedirs(output_path, exist_ok=True)

        # Save results as .csv
        results.to_csv(f"{output_path}/results.csv")

        logger.info("Results saved to csv file")

        logger.info("Saving model to file...")
        # Save full model (including weights) to .pth file - Not version proof
        model.save(f"{output_path}/model.pth")

        logger.info("Saving model config and state dict...")
        # Save model config and state dict - Version safe save method.
        model.save_state_dict(f"{output_path}/state_dict.pth")
        logger.info("Model config and state dict saved to file")
        model.save_config(f"{output_path}/model_config.yml")
        logger.info("Model config saved to file")
        
        logger.info("Prediction completed successfully!")
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'predict'.")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""Train GNN model without the use of config files."""
    )

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
        ("batch-size", 8),
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
        default=2,
        help="Number of gradient accumulation steps (default: 2 to maintain effective batch size of 16 with batch_size=8)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "predict"],
        help="Mode of operation: 'train' to train the model, 'predict' to run inference on validation data (default: %(default)s)",
    )
    
    parser.add_argument(
        "--num-stages",
        type=int,
        default=5,
        help="Number of training stages for multi-stage training (default: %(default)s)",
    )
    
    parser.add_argument(
        "--stage-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Custom epochs per stage. If not provided, max_epochs will be divided equally among stages.",
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
        args.num_stages,
        args.stage_epochs,
    )
