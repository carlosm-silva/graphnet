"""AdamW model wrapper with an FP32 exponential moving average for IceMix."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import torch
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel
from torch_geometric.data import Data

from graphnet.models import StandardModel


EMA_STATE_PREFIX = "_ema_model.module."


def extract_inference_state_dict(
    checkpoint_or_state: Mapping[str, Any],
) -> Dict[str, Tensor]:
    """Return ordinary-model weights, preferring EMA weights when present.

    Parameters
    ----------
    checkpoint_or_state : mapping
        Lightning checkpoint containing ``state_dict`` or a bare state mapping.

    Returns
    -------
    dict of str to torch.Tensor
        Keys compatible with an ordinary GraphNeT ``StandardModel``.

    Raises
    ------
    TypeError
        If the resolved state dictionary is not a mapping.
    """
    candidate = checkpoint_or_state.get("state_dict", checkpoint_or_state)
    if not isinstance(candidate, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping.")

    ema_state = {
        key[len(EMA_STATE_PREFIX) :]: value
        for key, value in candidate.items()
        if key.startswith(EMA_STATE_PREFIX) and isinstance(value, Tensor)
    }
    if ema_state:
        return ema_state
    return {
        key: value
        for key, value in candidate.items()
        if isinstance(value, Tensor)
        and not key.startswith("_ema_model.")
        and key != "_ema_updates"
    }


class EMAStandardModel(StandardModel):
    """Train online weights with AdamW and validate using an FP32 EMA copy."""

    def __init__(self, *args: Any, ema_decay: float = 0.999, **kwargs: Any) -> None:
        """Construct the online GraphNeT model and its non-trainable FP32 EMA copy.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :class:`graphnet.models.StandardModel`.
        ema_decay : float
            Previous-average weight in ``[0, 1)`` for each optimizer-step
            exponential moving-average update.
        """
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), received {ema_decay}.")
        self.ema_decay = float(ema_decay)
        super().__init__(*args, **kwargs)

        # AveragedModel deep-copies self before _ema_model is assigned, avoiding
        # recursive EMA copies while retaining the exact StandardModel layout.
        self._ema_model = AveragedModel(self, use_buffers=True)
        self._ema_model.float().requires_grad_(False).eval()
        self.register_buffer("_ema_updates", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True) -> "EMAStandardModel":
        """Set online modules to training/evaluation mode and keep EMA in eval."""
        super().train(mode)
        if hasattr(self, "_ema_model"):
            self._ema_model.eval()
        return self

    def online_named_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        """Yield named parameters excluding the non-trainable EMA copy."""
        return (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if not name.startswith("_ema_model.")
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build the configured optimizer and scheduler for online weights only."""
        parameters = [
            parameter
            for _, parameter in self.online_named_parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError(
                "AdamW requires at least one trainable online parameter."
            )
        optimizer = self._optimizer_class(parameters, **self._optimizer_kwargs)
        config: Dict[str, Any] = {"optimizer": optimizer}
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(optimizer, **self._scheduler_kwargs)
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                **self._scheduler_config,
            }
        return config

    @torch.no_grad()
    def copy_online_to_ema(self) -> None:
        """Initialize EMA tensors from online tensors in FP32."""
        online = dict(self.named_parameters())
        online_buffers = dict(self.named_buffers())
        for name, ema_parameter in self._ema_model.module.named_parameters():
            ema_parameter.copy_(online[name].detach().to(dtype=ema_parameter.dtype))
        for name, ema_buffer in self._ema_model.module.named_buffers():
            source = online_buffers[name].detach().to(dtype=ema_buffer.dtype)
            ema_buffer.copy_(source)
        self._ema_updates.zero_()
        self._ema_model.n_averaged.zero_()

    @torch.no_grad()
    def update_ema(self) -> None:
        """Apply one EMA update after an optimizer step."""
        online = dict(self.named_parameters())
        online_buffers = dict(self.named_buffers())
        decay = self.ema_decay
        for name, ema_parameter in self._ema_model.module.named_parameters():
            source = online[name].detach().to(dtype=ema_parameter.dtype)
            ema_parameter.mul_(decay).add_(source, alpha=1.0 - decay)
        for name, ema_buffer in self._ema_model.module.named_buffers():
            source = online_buffers[name].detach().to(dtype=ema_buffer.dtype)
            if torch.is_floating_point(ema_buffer):
                ema_buffer.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                ema_buffer.copy_(source)
        self._ema_updates.add_(1)
        self._ema_model.n_averaged.copy_(self._ema_updates)

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        """Perform one online optimizer step, update EMA, and log EMA state."""
        super().optimizer_step(*args, **kwargs)
        self.update_ema()
        self.log("ema_decay", self.ema_decay, on_step=True, on_epoch=False)
        self.log(
            "ema_updates",
            self._ema_updates.to(dtype=torch.float32),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist the EMA hyperparameter alongside Lightning's full state."""
        checkpoint["icemix_ema"] = {
            "decay": self.ema_decay,
            "updates": int(self._ema_updates.item()),
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore the decay used by the run instead of trusting new config."""
        ema_metadata = checkpoint.get("icemix_ema", {})
        if "decay" in ema_metadata:
            self.ema_decay = float(ema_metadata["decay"])

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Compute and log checkpoint-selection loss with EMA weights.

        Returns the scalar task loss and logs epoch-level ``val_loss`` with DDP
        synchronization. Online weights are not modified.
        """
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        ema_module = self._ema_model.module
        ema_module.eval()
        loss = ema_module.shared_step(val_batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def metric_tasks_for_phase(self, phase: str) -> nn.ModuleList:
        """Return EMA tasks for validation and online tasks otherwise."""
        if phase == "val":
            return self._ema_model.module._tasks
        return self._tasks

    def load_source_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        """Load plain weights into the online model and initialize EMA.

        Raises ``RuntimeError`` when non-EMA keys are missing or unexpected.
        Optimizer and scheduler state are intentionally not loaded.
        """
        incompatible = self.load_state_dict(dict(state_dict), strict=False)
        missing_online = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("_ema_model.") and key != "_ema_updates"
        ]
        if missing_online or incompatible.unexpected_keys:
            raise RuntimeError(
                "Source checkpoint is incompatible with the online model: "
                f"missing={missing_online}, unexpected={incompatible.unexpected_keys}"
            )
        self.copy_online_to_ema()
