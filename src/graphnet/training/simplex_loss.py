"""Simplex-constrained multi-loss functions for balanced multi-task learning.

This module provides loss functions that combine multiple losses with weights
constrained to lie on a (k-1)-simplex, ensuring they sum to 1 and are non-negative.
This provides a principled mathematical foundation for multi-task learning with
automatic loss scale balancing.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphnet.training.loss_functions import LossFunction


class SimplexMultiLoss(LossFunction):
    """K-simplex constrained multi-loss with adaptive balancing.
    
    Combines k losses with weights constrained to a (k-1)-simplex:
    - Weights sum to 1: Σᵢ wᵢ = 1  
    - Weights non-negative: wᵢ ≥ 0
    - Automatic scale balancing for each loss component
    
    The final loss is computed as:
    combined_loss = Σᵢ wᵢ * (lossᵢ * scaleᵢ)
    
    Where scaleᵢ are normalization factors that bring all losses to similar magnitudes.
    """
    
    def __init__(
        self,
        loss_functions: List[LossFunction],
        loss_names: List[str],
        weights: Optional[List[float]] = None,
        balance_method: str = "running_mean",
        momentum: float = 0.95,
        min_samples: int = 10,
        learnable_weights: bool = False,
        prediction_slices: Optional[List[slice]] = None,
        evaluation_weights: Optional[List[float]] = None,
    ):
        """Initialize SimplexMultiLoss.
        
        Args:
            loss_functions: List of k loss functions to combine
            loss_names: Names for each loss (for debugging and target extraction)
            weights: Initial simplex weights [w₁, w₂, ..., wₖ] where Σwᵢ=1
                    If None, uses uniform: [1/k, 1/k, ..., 1/k]
            balance_method: Method for automatic loss balancing:
                - "running_mean": Use running mean normalization (default)
                - "batch_mean": Use current batch mean normalization  
                - "none": No automatic balancing (use raw losses)
            momentum: EMA momentum for running_mean method (0.95 = slow adaptation)
            min_samples: Minimum samples before applying running mean balancing
            learnable_weights: If True, make simplex weights learnable parameters
            prediction_slices: Optional slices for extracting predictions from tensor
            evaluation_weights: Fixed weights for model selection evaluation loss.
                               If provided, a separate evaluation loss will be computed
                               using these fixed weights for ModelCheckpoint/EarlyStopping.
                               Must sum to 1 and be >= 0 (simplex constraint).
        """
        super().__init__()
        
        self.k = len(loss_functions)
        if len(loss_names) != self.k:
            raise ValueError(f"Number of loss_names ({len(loss_names)}) must match loss_functions ({self.k})")
        
        self.loss_functions = nn.ModuleList(loss_functions)
        self.loss_names = loss_names
        self.balance_method = balance_method
        self.momentum = momentum
        self.min_samples = min_samples
        self.prediction_slices = prediction_slices
        
        # Initialize adaptive training weights (simplex constraint)
        if weights is None:
            weights = [1.0 / self.k] * self.k  # Uniform
        else:
            weights = self._normalize_to_simplex(weights)
        
        if learnable_weights:
            # Use softmax parameterization for automatic simplex constraint
            self._raw_weights = nn.Parameter(torch.zeros(self.k))
        else:
            self.register_buffer('_fixed_weights', torch.tensor(weights, dtype=torch.float32))
        
        self.learnable_weights = learnable_weights
        
        # Initialize evaluation weights for model selection
        self.evaluation_weights = None
        if evaluation_weights is not None:
            evaluation_weights = self._normalize_to_simplex(evaluation_weights)
            self.register_buffer('_evaluation_weights', torch.tensor(evaluation_weights, dtype=torch.float32))
            self.evaluation_weights = self._evaluation_weights
        
        # Running statistics for automatic balancing
        self.register_buffer('running_means', torch.ones(self.k))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.long))
        
        # Store last computed losses for logging
        self.last_individual_losses = None
        self.last_training_loss = None  
        self.last_evaluation_loss = None
    
    def _normalize_to_simplex(self, weights: List[float]) -> List[float]:
        """Ensure weights sum to 1 and are non-negative."""
        weights = [max(0.0, w) for w in weights]  # Ensure non-negative
        total = sum(weights)
        if total == 0:
            return [1.0 / len(weights)] * len(weights)  # Fallback to uniform
        return [w / total for w in weights]
    
    @property
    def simplex_weights(self) -> Tensor:
        """Get current simplex weights (adaptive training weights)."""
        if self.learnable_weights:
            return F.softmax(self._raw_weights, dim=0)
        else:
            return self._fixed_weights
    
    def _extract_prediction_component(self, prediction: Tensor, loss_idx: int) -> Tensor:
        """Extract the prediction component for a specific loss."""
        if self.prediction_slices is not None:
            return prediction[:, self.prediction_slices[loss_idx]]
        else:
            # Default: assume prediction tensor should be split equally
            pred_dim = prediction.shape[1]
            component_size = pred_dim // self.k
            start_idx = loss_idx * component_size
            if loss_idx == self.k - 1:  # Last component gets remaining dimensions
                return prediction[:, start_idx:]
            else:
                return prediction[:, start_idx:start_idx + component_size]
    
    def _update_running_stats(self, batch_means: Tensor):
        """Update running averages of loss magnitudes."""
        if self.num_updates < self.min_samples:
            # Initialize with current batch statistics
            self.running_means.copy_(batch_means)
        else:
            # Exponential moving average
            self.running_means.mul_(self.momentum).add_(batch_means, alpha=1 - self.momentum)
        self.num_updates += 1

    def _get_normalization_scales(self, batch_means: Tensor) -> Tensor:
        """Compute normalization scales for each loss."""
        if self.balance_method == "none":
            return torch.ones(self.k, device=batch_means.device, dtype=batch_means.dtype)
        elif self.balance_method == "batch_mean":
            return 1.0 / (batch_means + 1e-8)
        else:  # running_mean
            return 1.0 / (self.running_means + 1e-8)
    
    def _forward(self, prediction: Tensor, target: Union[Tensor, Dict[str, Tensor]]) -> Tensor:
        """Compute simplex-weighted combination of losses.
        
        Args:
            prediction: Model predictions. Can be:
                - Single tensor that will be split among losses
                - Dict mapping loss names to prediction tensors
            target: Targets. Can be:
                - Single tensor that will be split among losses (same as prediction)
                - Dict mapping loss names to target tensors
                
        Returns:
            Combined training loss with simplex constraint and adaptive balancing.
            If evaluation_weights provided, also stores evaluation loss for callbacks.
        """
        # Handle different input formats
        if isinstance(target, dict):
            # Dictionary format: extract components by name
            target_dict = target
            if isinstance(prediction, dict):
                prediction_dict = prediction
            else:
                # Split prediction tensor for dict targets
                prediction_dict = {}
                for i, name in enumerate(self.loss_names):
                    prediction_dict[name] = self._extract_prediction_component(prediction, i)
        else:
            # Tensor format: split both prediction and target
            if isinstance(prediction, dict):
                raise ValueError("If prediction is dict, target must also be dict")
            
            target_dict = {}
            prediction_dict = {}
            
            # Split target tensor among losses
            target_dim = target.shape[-1] if target.dim() > 1 else target.shape[0]
            component_size = target_dim // self.k
            
            for i, name in enumerate(self.loss_names):
                prediction_dict[name] = self._extract_prediction_component(prediction, i)
                
                # Extract target component
                start_idx = i * component_size
                if i == self.k - 1:  # Last component gets remaining dimensions
                    target_dict[name] = target[..., start_idx:]
                else:
                    target_dict[name] = target[..., start_idx:start_idx + component_size]
        
        # Compute individual losses
        individual_losses = []
        batch_means = []
        
        for i, (loss_fn, name) in enumerate(zip(self.loss_functions, self.loss_names)):
            if name not in target_dict:
                raise ValueError(f"Target for loss '{name}' not found in targets")
            if name not in prediction_dict:
                raise ValueError(f"Prediction for loss '{name}' not found in predictions")
                
            # Compute loss - handle both GraphNet LossFunction and PyTorch loss functions
            if hasattr(loss_fn, 'forward') and 'return_elements' in loss_fn.forward.__code__.co_varnames:
                # GraphNet LossFunction - supports return_elements
                loss_i = loss_fn.forward(
                    prediction_dict[name], 
                    target_dict[name], 
                    return_elements=True
                )
            else:
                # Standard PyTorch loss function
                loss_i = loss_fn(prediction_dict[name], target_dict[name])
                # Ensure it's per-sample loss (not reduced)
                if loss_i.dim() == 0:  # Scalar loss, expand to per-sample
                    batch_size = prediction_dict[name].shape[0]
                    loss_i = loss_i.unsqueeze(0).expand(batch_size)
                elif loss_i.dim() > 1:  # Multi-dimensional loss, reduce to per-sample
                    loss_i = loss_i.view(loss_i.shape[0], -1).mean(dim=1)
            individual_losses.append(loss_i)
            batch_means.append(loss_i.detach().mean())
        
        # Store individual losses for logging
        self.last_individual_losses = [loss.detach().mean() for loss in individual_losses]
        
        # Update running statistics for balancing
        batch_means = torch.stack(batch_means)
        if self.training and self.balance_method != "none":
            self._update_running_stats(batch_means)
        
        # Get normalization scales
        scales = self._get_normalization_scales(batch_means)
        
        # Compute adaptive training loss
        training_weights = self.simplex_weights
        training_loss = torch.zeros_like(individual_losses[0])
        
        for i, (loss_i, weight_i, scale_i) in enumerate(zip(individual_losses, training_weights, scales)):
            normalized_loss_i = loss_i * scale_i
            training_loss += weight_i * normalized_loss_i
        
        # Store training loss for logging
        self.last_training_loss = training_loss.detach().mean()
        
        # Compute fixed evaluation loss if evaluation weights provided
        if self.evaluation_weights is not None:
            evaluation_loss = torch.zeros_like(individual_losses[0])
            for i, (loss_i, eval_weight_i, scale_i) in enumerate(zip(individual_losses, self.evaluation_weights, scales)):
                normalized_loss_i = loss_i * scale_i
                evaluation_loss += eval_weight_i * normalized_loss_i
            
            # Store evaluation loss for logging and callback access
            self.last_evaluation_loss = evaluation_loss.detach().mean()
        
        return training_loss
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get current loss statistics for debugging."""
        stats = {
            "simplex_weights": self.simplex_weights.detach().cpu().tolist(),
            "running_means": self.running_means.detach().cpu().tolist(),
            "num_updates": int(self.num_updates.item()) if hasattr(self.num_updates, 'item') else int(self.num_updates),
            "loss_names": self.loss_names,
        }
        
        if self.learnable_weights and hasattr(self, '_raw_weights'):
            stats["raw_weights"] = self._raw_weights.detach().cpu().tolist()
            
        # Add individual loss values if available
        if self.last_individual_losses is not None:
            for i, (name, loss_val) in enumerate(zip(self.loss_names, self.last_individual_losses)):
                stats[f"loss_{name}"] = float(loss_val.item()) if hasattr(loss_val, 'item') else float(loss_val)
                
        # Add training and evaluation loss values
        if self.last_training_loss is not None:
            stats["training_loss"] = float(self.last_training_loss.item()) if hasattr(self.last_training_loss, 'item') else float(self.last_training_loss)
            
        if self.last_evaluation_loss is not None:
            stats["evaluation_loss"] = float(self.last_evaluation_loss.item()) if hasattr(self.last_evaluation_loss, 'item') else float(self.last_evaluation_loss)
            
        return stats
    
    def get_evaluation_loss(self) -> Optional[float]:
        """Get the last computed evaluation loss for model selection callbacks.
        
        Returns:
            The evaluation loss value if evaluation_weights were provided and 
            a forward pass has been computed, None otherwise.
        """
        if self.last_evaluation_loss is not None:
            return float(self.last_evaluation_loss.item()) if hasattr(self.last_evaluation_loss, 'item') else float(self.last_evaluation_loss)
        return None
    
    def has_evaluation_loss(self) -> bool:
        """Check if this loss function computes evaluation loss for model selection."""
        return self.evaluation_weights is not None
    
    def set_weights(self, new_weights: List[float]):
        """Update the simplex weights (only for non-learnable weights)."""
        if hasattr(self, 'raw_weights'):
            raise RuntimeError("Cannot set weights when learnable_weights=True. Use gradient descent instead.")
            
        new_weights = self._normalize_to_simplex(new_weights)
        self.weights.copy_(torch.tensor(new_weights, dtype=self.weights.dtype))


class TwoLossSimplexLoss(SimplexMultiLoss):
    """Simplified interface for 2-loss case with single alpha parameter.
    
    This class provides a drop-in replacement for binary loss combinations
    with the mathematical guarantee that weights sum to 1.
    
    Usage:
        loss = TwoLossSimplexLoss(
            loss1=EuclideanDistanceLoss(),
            loss2=VonMisesFisher3DLoss(),
            alpha=0.3  # 30% loss1, 70% loss2
        )
    """
    
    def __init__(
        self, 
        loss1: LossFunction,
        loss2: LossFunction, 
        alpha: float = 0.5,
        loss1_name: str = "loss1",
        loss2_name: str = "loss2",
        evaluation_alpha: Optional[float] = None,
        **kwargs
    ):
        """Initialize two-loss simplex combination.
        
        Args:
            loss1: First loss function
            loss2: Second loss function  
            alpha: Weight for first loss, alpha ∈ [0,1]. Second loss gets (1-alpha)
            loss1_name: Name for first loss (for debugging)
            loss2_name: Name for second loss (for debugging)
            evaluation_alpha: Fixed alpha for model selection evaluation loss.
                             If provided, enables separate evaluation loss for callbacks.
                             Must be in [0,1]. If None, no evaluation loss computed.
            **kwargs: Additional arguments passed to SimplexMultiLoss
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
            
        weights = [alpha, 1.0 - alpha]
        
        # Handle evaluation weights
        evaluation_weights = None
        if evaluation_alpha is not None:
            if not 0.0 <= evaluation_alpha <= 1.0:
                raise ValueError(f"evaluation_alpha must be in [0, 1], got {evaluation_alpha}")
            evaluation_weights = [evaluation_alpha, 1.0 - evaluation_alpha]
        
        super().__init__(
            loss_functions=[loss1, loss2],
            loss_names=[loss1_name, loss2_name], 
            weights=weights,
            evaluation_weights=evaluation_weights,
            **kwargs
        )
        
    def set_alpha(self, alpha: float):
        """Update alpha parameter (only for non-learnable weights).
        
        Args:
            alpha: New weight for first loss, alpha ∈ [0,1]
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
            
        self.set_weights([alpha, 1.0 - alpha])
    
    @property 
    def alpha(self) -> float:
        """Current alpha value (weight of first loss)."""
        return self.simplex_weights[0].item()


class JointPositionDirectionSimplexLoss(TwoLossSimplexLoss):
    """Specialized simplex loss for joint position-direction reconstruction.
    
    This class provides a direct replacement for the problematic JointLoss
    with proper simplex constraints and adaptive balancing.
    """
    
    def __init__(
        self,
        position_loss: LossFunction,
        direction_loss: LossFunction,
        alpha: float = 0.3,
        evaluation_alpha: Optional[float] = None,
        **kwargs
    ):
        """Initialize joint position-direction loss.
        
        Args:
            position_loss: Loss function for position prediction (e.g., EuclideanDistanceLoss)
            direction_loss: Loss function for direction prediction (e.g., VonMisesFisher3DLoss)  
            alpha: Weight for position loss, alpha ∈ [0,1]. Direction gets (1-alpha)
            evaluation_alpha: Fixed alpha for model selection evaluation loss.
                             If provided, enables separate evaluation loss for callbacks.
                             Must be in [0,1]. If None, uses alpha value.
            **kwargs: Additional arguments passed to TwoLossSimplexLoss
        """
        super().__init__(
            loss1=position_loss,
            loss2=direction_loss,
            alpha=alpha,
            loss1_name="position",
            loss2_name="direction",
            evaluation_alpha=evaluation_alpha,
            prediction_slices=[slice(0, 3), slice(3, None)],  # pos: [:3], dir: [3:]
            **kwargs
        )
    
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Forward pass specialized for joint position-direction targets.
        
        Args:
            prediction: Model prediction with shape [N, 7] where:
                - [:, :3] are position predictions  
                - [:, 3:] are direction predictions (including kappa)
            target: Target tensor with shape [N, 6] where:
                - [:, :3] are position targets
                - [:, 3:] are direction targets
                
        Returns:
            Combined loss with simplex weighting and adaptive balancing.
        """
        # Check dimensions
        if target.dim() == 3:
            target = target.squeeze(1)  # Remove singleton dimension if present
            
        # Split target into position and direction components
        target_dict = {
            "position": target[:, :3],
            "direction": target[:, 3:] 
        }
        
        # Call parent with properly formatted targets
        return super()._forward(prediction, target_dict) 