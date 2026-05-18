"""Per-event physics metric tracking for IceMix joint reconstruction."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from pytorch_lightning import Callback
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from graphnet.training.loss_functions import JointLoss


_PER_EVENT_KEYS: Tuple[str, ...] = (
    "pos_err_m",
    "ang_err_deg",
    "kappa",
    "vmf_calibration",
)
_PERCENTILES: Tuple[float, ...] = (0.1, 0.5, 0.9)
_PERCENTILE_SUFFIXES: Tuple[str, ...] = ("p10", "median", "p90")
_EXPECTED_PREDICTION_DIM = 7
_EXPECTED_TARGET_DIM = 6


class JointLossWithMetrics(JointLoss):
    """:class:`JointLoss` that publishes per-event tensors and per-batch
    scalars to ``last_per_event`` / ``last_loss_components`` for
    :class:`PhysicsMetricsCallback`. The ``pos_err_m`` units assume
    ``position_loss`` is a Euclidean distance in meters."""

    def __init__(self, position_loss, direction_loss, alpha: float = 0.01):
        super().__init__(
            position_loss=position_loss,
            direction_loss=direction_loss,
            alpha=alpha,
        )
        self.last_loss_components: Dict[str, Tensor] = {}
        self.last_per_event: Dict[str, Tensor] = {}

    @staticmethod
    def _validate_joint_shapes(prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.dim() != 2 or prediction.size(1) != _EXPECTED_PREDICTION_DIM:
            raise ValueError(
                "JointLossWithMetrics expects prediction shape [N, 7] laid out as "
                "[pos_x, pos_y, pos_z, dir_x, dir_y, dir_z, kappa]. "
                f"Received {tuple(prediction.shape)}."
            )

        if target.dim() == 3:
            if target.size(1) != 1:
                raise ValueError(
                    "JointLossWithMetrics expects target shape [N, 1, 6] or [N, 6]. "
                    f"Received {tuple(target.shape)}."
                )
            target = target.squeeze(1)

        if target.dim() != 2 or target.size(1) != _EXPECTED_TARGET_DIM:
            raise ValueError(
                "JointLossWithMetrics expects target shape [N, 6] laid out as "
                "[pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]. "
                f"Received {tuple(target.shape)}."
            )

        if prediction.size(0) != target.size(0):
            raise ValueError(
                "Prediction and target batch dimensions must agree. "
                f"Received {prediction.size(0)} and {target.size(0)}."
            )

        return target

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        self.last_loss_components = {}
        self.last_per_event = {}

        target = self._validate_joint_shapes(prediction, target)

        position_pred = prediction[:, :3]
        direction_pred_full = prediction[:, 3:7]
        unit_dir = prediction[:, 3:6]
        kappa = prediction[:, 6]
        position_target = target[:, :3]
        direction_target = target[:, 3:]

        position_loss = self.position_loss(
            position_pred, position_target, return_elements=True
        )
        direction_loss = self.direction_loss(
            direction_pred_full, direction_target, return_elements=True
        )

        if position_loss.dim() == 0 or direction_loss.dim() == 0:
            raise ValueError(
                "[ERROR] Either `position_loss` or `direction_loss` returned a scalar. "
                "They must return per-sample losses with shape [N,]."
            )

        combined_loss = self.alpha * position_loss + direction_loss

        with torch.no_grad():
            pos_err_m = position_loss.detach()
            kappa_d = kappa.detach()
            cos_angle = (
                (unit_dir.detach() * direction_target.detach())
                .sum(dim=1)
                .clamp(-1.0, 1.0)
            )
            ang_err_rad = torch.arccos(cos_angle)

            self.last_loss_components = {
                "loss/position": (self.alpha * pos_err_m).mean(),
                "loss/direction": direction_loss.detach().mean(),
            }
            self.last_per_event = {
                "pos_err_m": pos_err_m,
                "ang_err_deg": torch.rad2deg(ang_err_rad),
                "kappa": kappa_d,
                "vmf_calibration": ang_err_rad * kappa_d.clamp_min(0).sqrt(),
            }

        return combined_loss


def _resolve_loss_fn(pl_module, task_index: int) -> JointLossWithMetrics:
    tasks = getattr(pl_module, "_tasks", None)
    if not tasks:
        raise RuntimeError(
            "PhysicsMetricsCallback requires pl_module._tasks to be populated."
        )
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(
            f"PhysicsMetricsCallback task_index={task_index} is out of range for "
            f"{len(tasks)} configured task(s)."
        )

    loss_fn = getattr(tasks[task_index], "_loss_function", None)
    if not isinstance(loss_fn, JointLossWithMetrics):
        raise TypeError(
            "PhysicsMetricsCallback requires the selected task to use "
            "JointLossWithMetrics. "
            f"Received {type(loss_fn).__name__!s} at task index {task_index}."
        )
    return loss_fn


def _is_rank_zero() -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank() == 0


class PerEventStats(Metric):
    """Per-event metric emitting ``[mean, p10, median, p90]``."""

    full_state_update: bool = False
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value: Tensor) -> None:
        if value is None or value.numel() == 0:
            return
        self.values.append(
            value.detach().flatten().to(device=self.device, dtype=torch.float32)
        )

    def compute(self) -> Tensor:
        nan_out = torch.full(
            (1 + len(_PERCENTILES),),
            float("nan"),
            device=self.device,
            dtype=torch.float32,
        )
        state = self.values
        if isinstance(state, list):
            if not state:
                return nan_out
            cat = dim_zero_cat(state)
        elif isinstance(state, Tensor):
            cat = state
        else:
            return nan_out

        if cat.numel() == 0:
            return nan_out

        cat = cat.to(dtype=torch.float32)
        mean = cat.mean()
        qs = torch.quantile(
            cat,
            torch.tensor(_PERCENTILES, device=cat.device, dtype=cat.dtype),
        )
        return torch.cat([mean.unsqueeze(0), qs])


def _stats_to_metric_dict(phase: str, key: str, stats: Tensor) -> Dict[str, float]:
    mean, p10, p50, p90 = (float(stats[i].item()) for i in range(4))
    return {
        f"{phase}/{key}_mean": mean,
        f"{phase}/{key}_p10": p10,
        f"{phase}/{key}_median": p50,
        f"{phase}/{key}_p90": p90,
    }


def _iter_loggers(pl_module) -> Iterable[object]:
    loggers = getattr(pl_module, "loggers", None)
    if loggers is not None:
        return loggers

    logger = getattr(pl_module, "logger", None)
    if logger is None:
        return ()
    return (logger,)


def _log_epoch_metrics_direct(pl_module, metrics: Dict[str, float]) -> None:
    step = getattr(pl_module, "global_step", None)
    for logger in _iter_loggers(pl_module):
        log_metrics = getattr(logger, "log_metrics", None)
        if callable(log_metrics):
            log_metrics(metrics, step)


class PhysicsMetricsCallback(Callback):
    """Log per-event physics metrics each epoch."""

    def __init__(self, task_index: int = 0) -> None:
        super().__init__()
        self._task_index = task_index
        self._metrics: Dict[str, Dict[str, PerEventStats]] = {
            phase: {key: PerEventStats() for key in _PER_EVENT_KEYS}
            for phase in ("train", "val")
        }

    def _phase_metrics(self, phase: str) -> Dict[str, PerEventStats]:
        return self._metrics[phase]

    @staticmethod
    def _batch_size(pl_module, batch) -> int:
        return pl_module._get_batch_size(
            batch if isinstance(batch, list) else [batch]
        )

    def _ensure_metrics_on_device(self, pl_module, phase: str) -> None:
        device = getattr(pl_module, "device", torch.device("cpu"))
        for metric in self._phase_metrics(phase).values():
            if metric.device != device:
                metric.to(device)

    def _update(self, pl_module, phase: str, batch) -> None:
        loss_fn = _resolve_loss_fn(pl_module, self._task_index)
        bs = self._batch_size(pl_module, batch)
        on_step = phase == "train"

        for k, v in loss_fn.last_loss_components.items():
            pl_module.log(
                f"{phase}/{k}",
                v,
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
                batch_size=bs,
            )

        self._ensure_metrics_on_device(pl_module, phase)
        per_event = loss_fn.last_per_event
        for k in _PER_EVENT_KEYS:
            metric_tensor = per_event.get(k)
            self._phase_metrics(phase)[k].update(metric_tensor)
            if on_step and metric_tensor is not None and metric_tensor.numel() > 0:
                pl_module.log(
                    f"{phase}/{k}_mean_step",
                    metric_tensor.detach()
                    .to(device=pl_module.device, dtype=torch.float32)
                    .mean(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=False,
                    rank_zero_only=True,
                    batch_size=bs,
                )

    @staticmethod
    def _prewarm_lightning_metric_cache(trainer) -> None:
        """Touch Lightning's metric cache on every rank.

        The progress bar reads ``trainer.progress_bar_metrics`` only on
        rank zero, which fires ``sync_dist=True`` collectives that would
        otherwise deadlock against this callback's gather.
        """
        if trainer is None:
            return
        try:
            _ = trainer.progress_bar_metrics
            _ = trainer.callback_metrics
        except Exception:
            pass

    def _flush(self, trainer, pl_module, phase: str) -> None:
        self._prewarm_lightning_metric_cache(trainer)

        epoch_metrics: Dict[str, float] = {}
        for k in _PER_EVENT_KEYS:
            metric = self._phase_metrics(phase)[k]
            stats = metric.compute()
            epoch_metrics.update(_stats_to_metric_dict(phase, k, stats))
            metric.reset()

        if epoch_metrics and _is_rank_zero():
            _log_epoch_metrics_direct(pl_module, epoch_metrics)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        for metric in self._phase_metrics("train").values():
            metric.reset()

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        for metric in self._phase_metrics("val").values():
            metric.reset()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        self._update(pl_module, "train", batch)
        if isinstance(outputs, Tensor):
            bs = self._batch_size(pl_module, batch)
            pl_module.log(
                "train/loss_step",
                outputs.detach(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                rank_zero_only=True,
                batch_size=bs,
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._flush(trainer, pl_module, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        self._update(pl_module, "val", batch)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._flush(trainer, pl_module, "val")
