"""Demonstration of evaluation loss for proper model selection with adaptive loss weighting.

This script demonstrates the solution to a critical problem: when using adaptive
loss weighting methods (uncertainty weighting, GradNorm, MGDA, SimplexMultiLoss),
the training loss changes during training, making it unsuitable for model selection
callbacks like ModelCheckpoint and EarlyStopping.

The solution is to use a fixed evaluation loss that provides a consistent
"yardstick" for model comparison across epochs.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from graphnet.training.loss_functions import EuclideanDistanceLoss, VonMisesFisher3DLoss
from graphnet.training.simplex_loss import TwoLossSimplexLoss


def simulate_training_with_evaluation_loss():
    """Simulate training to demonstrate the difference between adaptive training loss
    and fixed evaluation loss for model selection."""
    
    print("=" * 80)
    print("EVALUATION LOSS FOR MODEL SELECTION DEMO")
    print("=" * 80)
    
    # Create loss function with both adaptive training and fixed evaluation
    loss_function = TwoLossSimplexLoss(
        loss1=EuclideanDistanceLoss(),
        loss2=VonMisesFisher3DLoss(),
        loss1_name="position",
        loss2_name="direction",
        alpha=0.3,  # 30% position weight for training (ADAPTIVE)
        evaluation_alpha=0.01,  # 1% position weight for evaluation (FIXED)
        balance_method="running_mean",
        momentum=0.95,
        prediction_slices=[slice(0, 3), slice(3, 7)]  # pos: [0:3], dir: [3:7] (4D)
    )
    
    print(f"Training alpha (adaptive): {loss_function.alpha:.3f}")
    print(f"Evaluation alpha (fixed): 0.01")
    print(f"Has evaluation loss: {loss_function.has_evaluation_loss()}")
    print()
    
    # Create sample data: 7D predictions (3D position + 4D direction with kappa)
    batch_size = 64
    predictions_base = torch.randn(batch_size, 7)  # [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, kappa]
    targets_base = torch.randn(batch_size, 7)
    
    # Normalize direction components to unit vectors (first 3 components of direction part)
    predictions_base[:, 3:6] = torch.nn.functional.normalize(predictions_base[:, 3:6], dim=1)
    targets_base[:, 3:6] = torch.nn.functional.normalize(targets_base[:, 3:6], dim=1)
    
    # Set reasonable kappa values (concentration parameter)
    predictions_base[:, 6] = torch.abs(predictions_base[:, 6]) + 0.1  # Ensure positive
    targets_base[:, 6] = torch.abs(targets_base[:, 6]) + 0.1
    
    # Simulate data that shows convergence over epochs
    epochs = 20
    
    # Storage for metrics
    training_losses = []
    evaluation_losses = []
    adaptive_weights = []
    individual_losses = {"position": [], "direction": []}
    
    print("Epoch | Train Loss | Eval Loss  | Alpha | Pos Loss | Dir Loss | Status")
    print("-" * 75)
    
    best_eval_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        # Simulate improving predictions over time
        # Position loss decreases faster than direction loss (different convergence rates)
        pos_noise = max(0.1, 2.0 - epoch * 0.1)  # Fast decrease
        dir_noise = max(0.05, 1.0 - epoch * 0.03)  # Slower decrease
        
        # Create batch data with improving predictions
        predictions = predictions_base.clone()
        targets = targets_base.clone()
        
        # Add noise that decreases over epochs (simulating learning)
        predictions[:, :3] += pos_noise * torch.randn(batch_size, 3)  # Position
        predictions[:, 3:6] = torch.nn.functional.normalize(
            predictions[:, 3:6] + dir_noise * torch.randn(batch_size, 3), dim=1
        )  # Direction
        targets[:, 3:6] = torch.nn.functional.normalize(targets[:, 3:6], dim=1)
        
        # Forward pass
        loss_function.train()
        training_loss = loss_function(predictions, targets)
        mean_training_loss = training_loss.mean().item()
        
        # Get evaluation loss and statistics
        evaluation_loss = loss_function.get_evaluation_loss()
        stats = loss_function.get_loss_statistics()
        
        # Store metrics
        training_losses.append(mean_training_loss)
        evaluation_losses.append(evaluation_loss)
        adaptive_weights.append(stats['simplex_weights'][0])  # Position weight
        individual_losses["position"].append(stats['loss_position'])
        individual_losses["direction"].append(stats['loss_direction'])
        
        # Model selection logic (what ModelCheckpoint would do)
        status = ""
        if evaluation_loss < best_eval_loss:
            best_eval_loss = evaluation_loss
            best_epoch = epoch
            patience_counter = 0
            status = "✓ SAVED"
        else:
            patience_counter += 1
            if patience_counter >= patience:
                status = "✗ EARLY STOP"
            else:
                status = f"({patience_counter}/{patience})"
        
        print(f"{epoch+1:5d} | {mean_training_loss:10.4f} | {evaluation_loss:9.4f} | "
              f"{adaptive_weights[-1]:5.3f} | {individual_losses['position'][-1]:8.4f} | "
              f"{individual_losses['direction'][-1]:8.4f} | {status}")
        
        if status == "✗ EARLY STOP":
            print(f"\nEarly stopping triggered! Best model from epoch {best_epoch + 1}")
            break
    
    print()
    print("=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    
    print(f"📊 Training loss changed from {training_losses[0]:.4f} to {training_losses[-1]:.4f}")
    print(f"📊 Evaluation loss changed from {evaluation_losses[0]:.4f} to {evaluation_losses[-1]:.4f}")
    print(f"📊 Alpha changed from {adaptive_weights[0]:.3f} to {adaptive_weights[-1]:.3f}")
    print()
    
    print("✅ EVALUATION LOSS ADVANTAGES:")
    print("   • Consistent comparison across epochs (same loss function)")
    print("   • Reflects true task performance, not loss scaling artifacts")
    print("   • Enables proper model selection with ModelCheckpoint")
    print("   • Supports early stopping based on actual convergence")
    print()
    
    print("❌ TRAINING LOSS PROBLEMS:")
    print("   • Changes meaning as weights adapt (different loss functions)")
    print("   • May improve due to balancing, not better predictions")
    print("   • Unsuitable for ModelCheckpoint/EarlyStopping")
    print("   • Can mislead about actual model performance")
    
    return {
        'training_losses': training_losses,
        'evaluation_losses': evaluation_losses,
        'adaptive_weights': adaptive_weights,
        'individual_losses': individual_losses,
        'best_epoch': best_epoch
    }


def pytorch_lightning_integration_example():
    """Show how to integrate evaluation loss with PyTorch Lightning."""
    
    print("\n" + "=" * 80)
    print("PYTORCH LIGHTNING INTEGRATION")
    print("=" * 80)
    
    integration_code = '''
class JointReconstructionModule(LightningModule):
    """Example Lightning module with proper evaluation loss integration."""
    
    def __init__(self):
        super().__init__()
        
                 # 🔥 Create loss with BOTH adaptive training and fixed evaluation
         self.loss_fn = TwoLossSimplexLoss(
             loss1=EuclideanDistanceLoss(),
             loss2=VonMisesFisher3DLoss(),
             loss1_name="position",
             loss2_name="direction",
            alpha=0.3,  # Adaptive training weight (30% position)
            evaluation_alpha=0.01,  # FIXED evaluation weight (1% position)
            balance_method="running_mean",
            momentum=0.95
        )
        
        # Log the configuration
        print(f"Training alpha: {self.loss_fn.alpha}")
        print(f"Evaluation alpha: 0.01 (FIXED)")
        print(f"Has evaluation loss: {self.loss_fn.has_evaluation_loss()}")
    
    def training_step(self, batch, batch_idx):
        predictions = self.model(batch)
        
        # Use adaptive training loss for gradient updates
        loss = self.loss_fn(predictions, batch.y)
        train_loss = loss.mean()
        
        # Log training metrics
        self.log("train_loss", train_loss)
        
        # Optional: Log loss statistics for monitoring
        if batch_idx % 100 == 0:  # Log every 100 batches
            stats = self.loss_fn.get_loss_statistics()
            self.log("train_alpha", stats['simplex_weights'][0])
            self.log("train_pos_loss", stats['loss_position'])
            self.log("train_dir_loss", stats['loss_direction'])
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        predictions = self.model(batch)
        
        # Compute both losses
        _ = self.loss_fn(predictions, batch.y)
        
        # 🎯 KEY: Use FIXED evaluation loss for model selection
        eval_loss = self.loss_fn.get_evaluation_loss()
        self.log("val_loss", eval_loss)  # ← ModelCheckpoint monitors this!
        
        # Optional: Also log adaptive training loss for comparison
        train_loss = self.loss_fn.last_training_loss
        self.log("val_train_loss", train_loss)
        
        # Log individual losses
        stats = self.loss_fn.get_loss_statistics()
        self.log("val_pos_loss", stats['loss_position'])
        self.log("val_dir_loss", stats['loss_direction'])
        
        return eval_loss
    
    def on_validation_epoch_end(self):
        """Print evaluation loss after each validation epoch."""
        # Get the logged evaluation loss for this epoch
        eval_loss = self.trainer.logged_metrics.get("val_loss", None)
        train_loss = self.trainer.logged_metrics.get("val_train_loss", None)
        
        current_epoch = self.current_epoch
        
        if eval_loss is not None:
            print(f"Epoch {current_epoch + 1}: Evaluation Loss = {eval_loss:.6f}")
            if train_loss is not None:
                print(f"Epoch {current_epoch + 1}: Training Loss = {train_loss:.6f} "
                      f"(adaptive, for comparison)")
            
            # Also print individual loss components for detailed monitoring
            pos_loss = self.trainer.logged_metrics.get("val_pos_loss", None)
            dir_loss = self.trainer.logged_metrics.get("val_dir_loss", None)
            
            if pos_loss is not None and dir_loss is not None:
                print(f"Epoch {current_epoch + 1}: Position Loss = {pos_loss:.6f}, "
                      f"Direction Loss = {dir_loss:.6f}")
            print("-" * 60)
    
    def configure_callbacks(self):
        return [
            # 🎯 ModelCheckpoint uses FIXED evaluation loss
            ModelCheckpoint(
                monitor="val_loss",  # Uses evaluation_alpha=0.01 (FIXED)
                mode="min",
                save_top_k=3,
                filename="best-{epoch:02d}-{val_loss:.4f}"
            ),
            
            # 🎯 EarlyStopping uses FIXED evaluation loss  
            EarlyStopping(
                monitor="val_loss",  # Uses evaluation_alpha=0.01 (FIXED)
                patience=10,
                mode="min",
                verbose=True
            ),
            
            # Optional: Monitor training loss separately
            ModelCheckpoint(
                monitor="val_train_loss",  # Uses adaptive training loss
                mode="min", 
                save_top_k=1,
                filename="best-train-{epoch:02d}-{val_train_loss:.4f}"
            )
        ]
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3)


# Usage:
trainer = Trainer(
    max_epochs=100,
    callbacks=model.configure_callbacks(),
    logger=WandbLogger(project="joint-reconstruction")
)

trainer.fit(model, datamodule)
    '''
    
    print(integration_code)
    
    print("\n🔧 CONFIGURATION SUMMARY:")
    print("• training_step(): Uses adaptive loss for gradient updates")
    print("• validation_step(): Logs FIXED evaluation loss as 'val_loss'")
    print("• ModelCheckpoint: Monitors 'val_loss' (evaluation_alpha=0.01)")
    print("• EarlyStopping: Monitors 'val_loss' (evaluation_alpha=0.01)")
    print("• Result: Consistent model selection across training!")


def comparison_with_naive_approach():
    """Compare proper evaluation loss vs naive adaptive loss for model selection."""
    
    print("\n" + "=" * 80)
    print("COMPARISON: NAIVE vs PROPER MODEL SELECTION")
    print("=" * 80)
    
    # Create two identical loss functions
    naive_loss = TwoLossSimplexLoss(
        loss1=EuclideanDistanceLoss(),
        loss2=VonMisesFisher3DLoss(),
        loss1_name="position",
        loss2_name="direction",
        alpha=0.3,
        balance_method="running_mean",
        prediction_slices=[slice(0, 3), slice(3, 7)]
        # No evaluation_alpha = uses adaptive loss for model selection
    )
    
    proper_loss = TwoLossSimplexLoss(
        loss1=EuclideanDistanceLoss(),
        loss2=VonMisesFisher3DLoss(),
        loss1_name="position",
        loss2_name="direction",
        alpha=0.3,
        evaluation_alpha=0.01,  # Fixed evaluation loss
        balance_method="running_mean",
        prediction_slices=[slice(0, 3), slice(3, 7)]
    )
    
    print("NAIVE APPROACH (adaptive loss for model selection):")
    print("❌ Problem: Loss function changes meaning during training")
    print("❌ Cannot compare loss values across epochs reliably")
    print("❌ ModelCheckpoint may save suboptimal models")
    print("❌ EarlyStopping may trigger prematurely or never")
    print()
    
    print("PROPER APPROACH (fixed evaluation loss for model selection):")
    print("✅ Solution: Fixed loss function provides consistent yardstick")
    print("✅ Can compare evaluation loss values across all epochs")
    print("✅ ModelCheckpoint saves truly best models") 
    print("✅ EarlyStopping triggers based on actual convergence")
    print()
    
    # Simulate a problematic scenario
    batch_size = 32
    predictions = torch.randn(batch_size, 7)  # 7D: pos(3) + dir(3) + kappa(1)
    targets = torch.randn(batch_size, 7)
    
    # Normalize directions and set positive kappa
    predictions[:, 3:6] = torch.nn.functional.normalize(predictions[:, 3:6], dim=1)
    targets[:, 3:6] = torch.nn.functional.normalize(targets[:, 3:6], dim=1)
    predictions[:, 6] = torch.abs(predictions[:, 6]) + 0.1
    targets[:, 6] = torch.abs(targets[:, 6]) + 0.1
    
    print("SIMULATION: Same predictions, different epochs")
    
    for epoch in [1, 5, 10]:
        # Simulate running statistics changing over time
        for _ in range(epoch * 3):  # More updates = more adaptation
            _ = naive_loss(predictions, targets)
            _ = proper_loss(predictions, targets)
        
        naive_loss_val = naive_loss(predictions, targets).mean().item()
        proper_eval_loss = proper_loss.get_evaluation_loss()
        
        naive_stats = naive_loss.get_loss_statistics()
        proper_stats = proper_loss.get_loss_statistics()
        
        print(f"Epoch {epoch:2d}:")
        print(f"  Naive (adaptive): {naive_loss_val:.4f} " 
              f"(alpha={naive_stats['simplex_weights'][0]:.3f})")
        print(f"  Proper (fixed):   {proper_eval_loss:.4f} (alpha=0.01 FIXED)")
        print()
    
    print("🔍 OBSERVATION:")
    print("   Naive loss values change due to weight adaptation, not prediction quality")
    print("   Proper evaluation loss changes only due to actual model improvement")


if __name__ == "__main__":
    # Run demonstration
    results = simulate_training_with_evaluation_loss()
    pytorch_lightning_integration_example()
    comparison_with_naive_approach()
    
    print("\n" + "=" * 80)
    print("🎉 CONCLUSION")
    print("=" * 80)
    print("The evaluation loss feature solves a fundamental problem with adaptive")
    print("loss weighting: it provides a fixed yardstick for model selection while")
    print("still allowing adaptive balancing during training.")
    print()
    print("📋 IMPLEMENTATION CHECKLIST:")
    print("✅ Add evaluation_alpha parameter to loss function")
    print("✅ Log evaluation loss as 'val_loss' in validation_step()")
    print("✅ Configure ModelCheckpoint to monitor 'val_loss'")
    print("✅ Configure EarlyStopping to monitor 'val_loss'")
    print("✅ Optionally log adaptive training loss separately")
    print()
    print("Result: Proper model selection + adaptive loss balancing! 🚀") 