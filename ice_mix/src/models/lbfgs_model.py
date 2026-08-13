"""DDP-safe model wrapper for deterministic LBFGS closures."""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch_geometric.data import Data

from graphnet.models import StandardModel


@contextmanager
def seeded_model_rng(seed: int, device: torch.device) -> Iterator[None]:
    """Run a closure with repeatable randomness without changing global state."""
    cuda_devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def synchronize_loss_value(loss: Tensor) -> Tensor:
    """Give every DDP rank one loss value while preserving local gradients."""
    if not (dist.is_available() and dist.is_initialized()):
        return loss

    mean_loss = loss.detach().clone()
    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
    mean_loss /= dist.get_world_size()
    return loss + (mean_loss - loss.detach())


class DistributedLBFGSStandardModel(StandardModel):
    """Make repeated LBFGS closures deterministic and DDP rank-consistent."""

    def configure_optimizers(self) -> Dict[str, Any]:
        """Construct LBFGS from trainable parameters only.

        LBFGS flattens every parameter passed to it, including frozen parameters.
        Filtering here therefore avoids allocating curvature history for the frozen
        part of a fine-tuned model.
        """
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("LBFGS requires at least one trainable parameter.")

        optimizer = self._optimizer_class(
            trainable_parameters, **self._optimizer_kwargs
        )
        config: Dict[str, Any] = {"optimizer": optimizer}
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(optimizer, **self._scheduler_kwargs)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                **self._scheduler_config,
            }
        return config

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Discard curvature estimated from previous mini-batches."""
        if getattr(self, "reset_lbfgs_history_each_step", True):
            raw_optimizer = getattr(optimizer, "optimizer", optimizer)
            raw_optimizer.state.clear()
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        seed = getattr(self.backbone, "token_drop_seed", None)
        if seed is None:
            raise RuntimeError(
                "LBFGS requires TokenDropSeedCallback to set a closure seed."
            )

        with seeded_model_rng(seed, self.device):
            loss = super().training_step(train_batch, batch_idx)
        return synchronize_loss_value(loss)


def freeze_for_last_blocks_fine_tuning(
    model: StandardModel, train_last_n_blocks: int
) -> Tuple[int, int]:
    """Train only the final transformer blocks and prediction task(s)."""
    blocks = getattr(model.backbone, "blocks", None)
    if blocks is None:
        raise ValueError("Fine-tuning requires a backbone with a 'blocks' sequence.")
    if not 0 <= train_last_n_blocks <= len(blocks):
        raise ValueError(
            "train_last_n_blocks must be between 0 and "
            f"{len(blocks)}, received {train_last_n_blocks}."
        )

    model.requires_grad_(False)
    if train_last_n_blocks:
        for block in blocks[-train_last_n_blocks:]:
            block.requires_grad_(True)
    for task in model._tasks:
        task.requires_grad_(True)

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("_ema_model.")
    )
    return trainable, total
