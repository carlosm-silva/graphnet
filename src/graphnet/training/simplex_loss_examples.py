"""Usage examples for SimplexMultiLoss classes.

This file demonstrates different ways to use the simplex-constrained loss functions
for various multi-task learning scenarios.
"""

import torch
from graphnet.training.loss_functions import EuclideanDistanceLoss, VonMisesFisher3DLoss, LossFunction
from graphnet.training.simplex_loss import (
    SimplexMultiLoss, 
    TwoLossSimplexLoss, 
    JointPositionDirectionSimplexLoss
)


def example_1_basic_two_loss():
    """Example 1: Basic two-loss combination with fixed weights."""
    print("=== Example 1: Basic Two-Loss Combination ===")
    
    # Create loss functions
    pos_loss = EuclideanDistanceLoss()
    dir_loss = VonMisesFisher3DLoss()
    
    # Create simplex loss with 30% position, 70% direction
    loss_fn = TwoLossSimplexLoss(
        loss1=pos_loss,
        loss2=dir_loss,
        alpha=0.3,
        balance_method="running_mean"
    )
    
    # Example forward pass
    batch_size = 16
    prediction = torch.randn(batch_size, 7)  # 3 pos + 4 dir (including kappa)
    target = torch.randn(batch_size, 6)      # 3 pos + 3 dir
    
    loss = loss_fn(prediction, target)
    print(f"Combined loss: {loss.mean():.4f}")
    print(f"Current alpha: {loss_fn.alpha:.3f}")
    print(f"Simplex weights: {loss_fn.simplex_weights}")
    
    # Change alpha during training
    loss_fn.set_alpha(0.5)
    print(f"After setting alpha=0.5: {loss_fn.simplex_weights}")


def example_2_learnable_weights():
    """Example 2: Learnable simplex weights."""
    print("\n=== Example 2: Learnable Weights ===")
    
    pos_loss = EuclideanDistanceLoss()
    dir_loss = VonMisesFisher3DLoss()
    
    # Create loss with learnable weights
    loss_fn = TwoLossSimplexLoss(
        loss1=pos_loss,
        loss2=dir_loss,
        alpha=0.3,
        learnable_weights=True,
        balance_method="running_mean"
    )
    
    print(f"Initial weights: {loss_fn.simplex_weights}")
    
    # Simulate training step
    batch_size = 16
    prediction = torch.randn(batch_size, 7, requires_grad=True)
    target = torch.randn(batch_size, 6)
    
    loss = loss_fn(prediction, target).mean()
    loss.backward()
    
    print(f"Gradients on raw weights: {loss_fn.raw_weights.grad}")


def example_3_joint_position_direction():
    """Example 3: Drop-in replacement for current JointLoss."""
    print("\n=== Example 3: Joint Position-Direction (Drop-in Replacement) ===")
    
    pos_loss = EuclideanDistanceLoss()
    dir_loss = VonMisesFisher3DLoss()
    
    # Drop-in replacement for JointLoss
    loss_fn = JointPositionDirectionSimplexLoss(
        position_loss=pos_loss,
        direction_loss=dir_loss,
        alpha=0.3,  # Much cleaner than old alpha=0.04!
        balance_method="running_mean"
    )
    
    # Same interface as original JointLoss
    batch_size = 16
    prediction = torch.randn(batch_size, 7)  # 3 pos + 4 dir
    target = torch.randn(batch_size, 6)      # 3 pos + 3 dir
    
    loss = loss_fn(prediction, target)
    print(f"Joint loss: {loss.mean():.4f}")
    print(f"Position weight: {loss_fn.alpha:.3f}")
    print(f"Direction weight: {1 - loss_fn.alpha:.3f}")


def example_4_loss_statistics():
    """Example 4: Monitoring loss statistics."""
    print("\n=== Example 4: Loss Statistics and Debugging ===")
    
    pos_loss = EuclideanDistanceLoss()
    dir_loss = VonMisesFisher3DLoss()
    
    loss_fn = TwoLossSimplexLoss(
        loss1=pos_loss,
        loss2=dir_loss,
        alpha=0.3,
        balance_method="running_mean",
        momentum=0.9
    )
    
    # Simulate several training steps
    for step in range(5):
        batch_size = 16
        prediction = torch.randn(batch_size, 7)
        target = torch.randn(batch_size, 6)
        
        loss = loss_fn(prediction, target)
        
        if step % 2 == 0:
            stats = loss_fn.get_loss_statistics()
            print(f"Step {step}:")
            print(f"  Running means: {[f'{x:.3f}' for x in stats['running_means']]}")
            print(f"  Simplex weights: {[f'{x:.3f}' for x in stats['simplex_weights']]}")
            print(f"  Updates: {stats['num_updates']}")


def example_5_curriculum_learning():
    """Example 5: Curriculum learning with changing weights."""
    print("\n=== Example 5: Curriculum Learning ===")
    
    pos_loss = EuclideanDistanceLoss()
    dir_loss = VonMisesFisher3DLoss()
    
    loss_fn = TwoLossSimplexLoss(
        loss1=pos_loss,
        loss2=dir_loss,
        alpha=0.8,  # Start position-heavy
        balance_method="running_mean"
    )
    
    # Simulate curriculum: gradually shift from position to direction focus
    total_epochs = 10
    for epoch in range(total_epochs):
        # Linear interpolation: 0.8 → 0.3 over training
        progress = epoch / (total_epochs - 1)
        alpha = 0.8 * (1 - progress) + 0.3 * progress
        loss_fn.set_alpha(alpha)
        
        print(f"Epoch {epoch}: alpha={alpha:.3f}, weights={loss_fn.simplex_weights.tolist()}")


def example_6_comparison_with_original():
    """Example 6: Side-by-side comparison with problematic original approach."""
    print("\n=== Example 6: Comparison with Original JointLoss ===")
    
    pos_loss = EuclideanDistanceLoss()
    dir_loss = VonMisesFisher3DLoss()
    
    # Original approach (problematic)
    from graphnet.training.loss_functions import JointLoss
    original_loss = JointLoss(
        position_loss=pos_loss,
        direction_loss=dir_loss,
        alpha=0.04  # This was problematic!
    )
    
    # New simplex approach
    simplex_loss = JointPositionDirectionSimplexLoss(
        position_loss=pos_loss,
        direction_loss=dir_loss,
        alpha=0.04,  # Same value but with proper scaling
        balance_method="running_mean"
    )
    
    # Test with same data
    batch_size = 16
    prediction = torch.randn(batch_size, 7)
    target = torch.randn(batch_size, 6)
    
    # Compare losses
    original_result = original_loss(prediction, target)
    simplex_result = simplex_loss(prediction, target)
    
    print(f"Original JointLoss: {original_result.mean():.4f}")
    print(f"Simplex JointLoss: {simplex_result.mean():.4f}")
    print(f"Simplex weights: {simplex_loss.simplex_weights}")
    
    # Show statistics from simplex loss
    stats = simplex_loss.get_loss_statistics()
    print(f"Running means: {[f'{x:.3f}' for x in stats['running_means']]}")


def example_evaluation_loss_for_model_selection():
    """Example: Using separate evaluation loss for proper model selection.
    
    This example demonstrates the key insight that adaptive loss weighting 
    (uncertainty weighting, GradNorm, MGDA, etc.) changes the loss function
    during training, making it unsuitable for model selection. We need a 
    fixed "yardstick" for callbacks like ModelCheckpoint and EarlyStopping.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from graphnet.training.loss_functions import EuclideanDistanceLoss, VonMisesFisher3DLoss
    from graphnet.training.simplex_loss import JointPositionDirectionSimplexLoss
    
    print("=== Evaluation Loss for Model Selection ===")
    
    # Create sample data: 6D predictions (3D position + 3D direction)
    batch_size = 32
    predictions = torch.randn(batch_size, 6)  # [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
    targets = torch.randn(batch_size, 6)
    
    # Normalize direction components to unit vectors
    predictions[:, 3:] = torch.nn.functional.normalize(predictions[:, 3:], dim=1)
    targets[:, 3:] = torch.nn.functional.normalize(targets[:, 3:], dim=1)
    
    # ================================================================
    # APPROACH 1: Adaptive training loss + Fixed evaluation loss
    # ================================================================
    print("\nTraining with adaptive loss but fixed evaluation for model selection...")
    
    # Create loss with different weights for training vs. evaluation
    loss_function = JointPositionDirectionSimplexLoss(
        position_loss=EuclideanDistanceLoss(),
        direction_loss=VonMisesFisher3DLoss(),
        alpha=0.3,  # 30% position weight for training (adaptive)
        evaluation_alpha=0.01,  # 1% position weight for evaluation (FIXED)
        balance_method="running_mean",  # Enable adaptive balancing
        momentum=0.95
    )
    
    print(f"Has evaluation loss: {loss_function.has_evaluation_loss()}")
    print(f"Training alpha: {loss_function.alpha:.3f}")
    
    # Simulate training over multiple epochs
    for epoch in range(5):
        loss_function.train()  # Set to training mode
        
        # Forward pass - computes both training and evaluation loss
        training_loss = loss_function(predictions, targets)
        mean_training_loss = training_loss.mean()
        
        # Get the fixed evaluation loss for model selection
        evaluation_loss = loss_function.get_evaluation_loss()
        
        # Get detailed statistics
        stats = loss_function.get_loss_statistics()
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Training Loss: {mean_training_loss:.4f} (adaptive weights)")
        print(f"  Evaluation Loss: {evaluation_loss:.4f} (FIXED alpha=0.01)")
        print(f"  Individual Losses: pos={stats['loss_position']:.4f}, dir={stats['loss_direction']:.4f}")
        print(f"  Current Adaptive Weights: pos={stats['simplex_weights'][0]:.3f}, dir={stats['simplex_weights'][1]:.3f}")
        print(f"  Running Means: pos={stats['running_means'][0]:.4f}, dir={stats['running_means'][1]:.4f}")
        
        # =====================================================
        # This is the KEY INSIGHT for model selection:
        # =====================================================
        # Use evaluation_loss for ModelCheckpoint/EarlyStopping
        # Use mean_training_loss for gradient updates
        
        # Simulate PyTorch Lightning callback behavior:
        # ModelCheckpoint would use evaluation_loss to save best model
        # EarlyStopping would use evaluation_loss to detect convergence
        
        # Example callback logic:
        if epoch == 0:
            best_eval_loss = evaluation_loss
            print(f"  → New best model saved (eval_loss: {evaluation_loss:.4f})")
        elif evaluation_loss < best_eval_loss:
            best_eval_loss = evaluation_loss
            print(f"  → New best model saved (eval_loss: {evaluation_loss:.4f})")
        else:
            print(f"  → No improvement (best: {best_eval_loss:.4f})")
    
    # ================================================================
    # WHY THIS MATTERS: Comparison with naive approach
    # ================================================================
    print("\n" + "="*60)
    print("WHY FIXED EVALUATION LOSS IS ESSENTIAL:")
    print("="*60)
    
    print("\nPROBLEM: Using adaptive training loss for model selection")
    print("- Epoch 1: training_loss = 0.8 (alpha=0.3)")  
    print("- Epoch 2: training_loss = 0.7 (alpha=0.35) ← weights changed!")
    print("- Epoch 3: training_loss = 0.75 (alpha=0.28) ← weights changed!")
    print("❌ Cannot compare 0.8 vs 0.7 vs 0.75 - different loss functions!")
    
    print("\nSOLUTION: Using fixed evaluation loss for model selection")
    print("- Epoch 1: evaluation_loss = 0.85 (alpha=0.01 FIXED)")
    print("- Epoch 2: evaluation_loss = 0.82 (alpha=0.01 FIXED)")  
    print("- Epoch 3: evaluation_loss = 0.79 (alpha=0.01 FIXED)")
    print("✅ Can compare 0.85 > 0.82 > 0.79 - same loss function!")
    
    # ================================================================
    # INTEGRATION WITH PYTORCH LIGHTNING
    # ================================================================
    print("\n" + "="*60)
    print("PYTORCH LIGHTNING INTEGRATION:")
    print("="*60)
    
    example_integration_code = '''
    # In your LightningModule:
    
    def __init__(self):
        self.loss_fn = JointPositionDirectionSimplexLoss(
            position_loss=EuclideanDistanceLoss(),
            direction_loss=VonMisesFisher3DLoss(), 
            alpha=0.3,  # Adaptive training weight
            evaluation_alpha=0.01,  # FIXED evaluation weight for callbacks
            balance_method="running_mean"
        )
    
    def training_step(self, batch, batch_idx):
        predictions = self.model(batch)
        loss = self.loss_fn(predictions, batch.y)  # Adaptive training loss
        self.log("train_loss", loss.mean())
        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        predictions = self.model(batch)
        _ = self.loss_fn(predictions, batch.y)  # Compute both losses
        
        # Log FIXED evaluation loss for callbacks
        eval_loss = self.loss_fn.get_evaluation_loss()
        self.log("val_loss", eval_loss)  # ← ModelCheckpoint uses this!
        
        # Optionally log adaptive training loss for monitoring
        train_loss = self.loss_fn.last_training_loss
        self.log("val_train_loss", train_loss)
        
        return eval_loss
    
    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                monitor="val_loss",  # ← Uses FIXED evaluation loss
                mode="min",
                save_top_k=1
            ),
            EarlyStopping(
                monitor="val_loss",  # ← Uses FIXED evaluation loss  
                patience=10,
                mode="min"
            )
        ]
    '''
    
    print(example_integration_code)
    
    print("\n" + "="*60)
    print("MATHEMATICAL GUARANTEE:")
    print("="*60)
    print("• Training Loss: Σᵢ wᵢ(t) * lossᵢ * scaleᵢ(t)  [adaptive weights & scales]")
    print("• Evaluation Loss: Σᵢ wᵢ_fixed * lossᵢ * scaleᵢ(t)  [FIXED weights, adaptive scales]")
    print("• Evaluation loss provides consistent comparison across epochs")
    print("• Model selection based on true task performance, not loss dynamics")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_two_loss()
    example_2_learnable_weights()
    example_3_joint_position_direction()
    example_4_loss_statistics()
    example_5_curriculum_learning()
    example_6_comparison_with_original()
    example_evaluation_loss_for_model_selection()
    
    print("\n🎉 All SimplexMultiLoss examples completed successfully!")
    print("\nKey Benefits:")
    print("✅ Mathematically principled simplex constraints")
    print("✅ Automatic loss scale balancing")
    print("✅ Clean alpha ∈ [0,1] parameter interpretation")
    print("✅ Drop-in replacement for JointLoss")
    print("✅ Extensible to k>2 tasks")
    print("✅ Supports learnable task weights")
    print("✅ Built-in curriculum learning support") 