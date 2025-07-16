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


if __name__ == "__main__":
    # Run all working examples
    example_1_basic_two_loss()
    example_2_learnable_weights()
    example_3_joint_position_direction()
    example_4_loss_statistics()
    example_5_curriculum_learning()
    example_6_comparison_with_original()
    
    print("\n🎉 All SimplexMultiLoss examples completed successfully!")
    print("\nKey Benefits:")
    print("✅ Mathematically principled simplex constraints")
    print("✅ Automatic loss scale balancing")
    print("✅ Clean alpha ∈ [0,1] parameter interpretation")
    print("✅ Drop-in replacement for JointLoss")
    print("✅ Extensible to k>2 tasks")
    print("✅ Supports learnable task weights")
    print("✅ Built-in curriculum learning support") 