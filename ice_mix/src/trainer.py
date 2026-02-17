import os
import torch
from typing import List, Tuple, Optional
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
from graphnet.utilities.logging import Logger
from graphnet.models import StandardModel


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
    scratch_dir: str,
    swa_model: Optional[AveragedModel] = None,
    accumulate_grad_batches: int = 1,
) -> Tuple[str, float]:
    """
    Custom training loop with OneCycleLR scheduler and optional SWA support.

    Returns:
        Tuple of (best_checkpoint_path, best_val_loss)
    """
    logger = Logger()
    best_val_loss = float("inf")
    best_checkpoint_path = None

    # Ensure directories exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(
            train_dataloader, desc=f"Stage {stage} - Epoch {epoch+1}/{epochs}"
        )

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
            try:
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(
                    {"loss": f"{train_loss/train_steps:.4f}", "lr": f"{current_lr:.2e}"}
                )
            except Exception:
                pass

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
        logger.info(
            f"Stage {stage} - Epoch {epoch+1}: Train Loss = {train_loss/train_steps:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        # Create checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss,
        }

        # Save checkpoint to scratch (all epochs)
        scratch_path = os.path.join(
            scratch_dir,
            f"stage_{stage}_epoch={epoch+1}_val_loss={avg_val_loss:.4f}.ckpt",
        )
        torch.save(checkpoint, scratch_path)

        # Save checkpoint if best (to Main dir)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(
                checkpoint_dir,
                f"stage_{stage}_best-epoch={epoch+1}-val_loss={avg_val_loss:.4f}.ckpt",
            )
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"Saved new best checkpoint: {best_checkpoint_path}")

        model.train()

    return best_checkpoint_path, best_val_loss


def multi_stage_training(
    model: StandardModel,
    train_dataloader,
    val_dataloader,
    base_lr: float,
    checkpoint_dir: str,
    scratch_dir: str,
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
        scratch_dir: Directory to save intermediate checkpoints
        epochs_per_stage: List of epochs for each stage
        stage_lr_factors: Learning rate multipliers for each stage
        device: Device to train on
        accumulate_grad_batches: Gradient accumulation steps

    Returns:
        The final trained model (with SWA if applicable)
    """
    logger = Logger()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    # Calculate batches per epoch
    batches_per_epoch = len(train_dataloader)

    # Initialize SWA model (will be used from stage 2 onwards)
    swa_model = None

    # Track best checkpoint across all stages
    overall_best_checkpoint = None
    overall_best_loss = float("inf")

    for stage_idx, (epochs, lr_factor) in enumerate(
        zip(epochs_per_stage, stage_lr_factors)
    ):
        stage = stage_idx + 1
        max_lr = base_lr * lr_factor

        logger.info(f"\n{'='*50}")
        logger.info(f"Starting Stage {stage} - Max LR: {max_lr}, Epochs: {epochs}")
        logger.info(f"{'='*50}")

        # Load checkpoint from previous stage if not stage 1
        if stage > 1 and overall_best_checkpoint:
            logger.info(
                f"Loading checkpoint from previous stage: {overall_best_checkpoint}"
            )
            checkpoint = torch.load(overall_best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])

            # Initialize SWA model starting from stage 2
            if stage == 2:
                logger.info("Initializing SWA model for stage 2+")
                swa_model = AveragedModel(model)

        # Create new optimizer for this stage
        optimizer = AdamW(model.parameters(), lr=max_lr / 25.0, eps=1e-05)

        # Calculate total steps for this stage
        total_steps = epochs * batches_per_epoch // accumulate_grad_batches

        # Create OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            anneal_strategy="cos",
            pct_start=0.01,
            div_factor=25.0,
            final_div_factor=25.0,
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
            scratch_dir=scratch_dir,
            swa_model=swa_model if stage > 1 else None,
            accumulate_grad_batches=accumulate_grad_batches,
        )

        # Update overall best
        if best_loss < overall_best_loss:
            overall_best_loss = best_loss
            overall_best_checkpoint = best_checkpoint
            overall_best_loss = float(overall_best_loss)  # Ensure it is a float

        logger.info(f"Stage {stage} completed. Best loss: {best_loss:.4f}")

    # After all stages, finalize SWA model if used
    if swa_model is not None:
        logger.info("Updating batch normalization statistics for SWA model...")
        swa_model.to(device)
        update_bn(train_dataloader, swa_model, device=device)

        # Save final SWA model
        swa_checkpoint_path = os.path.join(checkpoint_dir, "final_swa_model.ckpt")
        torch.save(
            {
                "state_dict": swa_model.module.state_dict(),
                "swa_state_dict": swa_model.state_dict(),
            },
            swa_checkpoint_path,
        )
        logger.info(f"Saved final SWA model to: {swa_checkpoint_path}")

        # Return the SWA model's base model
        return swa_model.module
    else:
        return model


def standard_train_loop(
    model: StandardModel,
    train_dataloader,
    val_dataloader,
    checkpoint_dir: str,
    scratch_dir: str,
    optimizer,
    scheduler,
    device,
    max_epochs: int,
    accumulate_grad_batches: int = 1,
) -> StandardModel:
    """
    Standard training loop with ReduceLROnPlateau support.
    """
    logger = Logger()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_checkpoint_path = None

    logger.info(f"Starting standard training for {max_epochs} epochs")

    model.to(device)
    model.train()

    for epoch in range(max_epochs):
        # Training phase
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
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

            # Optimizer step
            if (batch_idx + 1) % accumulate_grad_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulate_grad_batches
            train_steps += 1

            # Update progress bar
            try:
                # Get usage from first param group
                current_lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    {"loss": f"{train_loss/train_steps:.4f}", "lr": f"{current_lr:.2e}"}
                )
            except Exception:
                pass

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
        logger.info(
            f"Epoch {epoch+1}: Train Loss = {train_loss/train_steps:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        # Step scheduler (ReduceLROnPlateau steps on metrics)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Create checkpoint
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": avg_val_loss,
        }

        # Save to scratch
        scratch_path = os.path.join(
            scratch_dir, f"epoch={epoch+1}_val_loss={avg_val_loss:.4f}.ckpt"
        )
        torch.save(checkpoint, scratch_path)

        # Save best to main
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(
                checkpoint_dir, f"best_epoch={epoch+1}_val_loss={avg_val_loss:.4f}.ckpt"
            )
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"Saved new best checkpoint: {best_checkpoint_path}")

        model.train()

    return model
