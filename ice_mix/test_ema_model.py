"""CPU tests for the IceMix-local AdamW+EMA model wrapper."""

from copy import deepcopy
from types import MethodType

import pytest
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel

from ice_mix.src.models.ema_model import (
    EMAStandardModel,
    extract_inference_state_dict,
)


def _fixture(decay: float = 0.5) -> EMAStandardModel:
    model = object.__new__(EMAStandardModel)
    nn.Module.__init__(model)
    model.online = nn.Linear(2, 1)
    model.frozen = nn.Linear(2, 1)
    model.frozen.requires_grad_(False)
    model._tasks = nn.ModuleList([nn.Linear(1, 1)])
    model.ema_decay = decay
    model._optimizer_class = AdamW
    model._optimizer_kwargs = {"lr": 0.1, "eps": 1e-5, "weight_decay": 0.01}
    model._scheduler_class = None
    model._scheduler_kwargs = {}
    model._scheduler_config = {}
    model._ema_model = AveragedModel(deepcopy(model), use_buffers=True)
    model._ema_model.float().requires_grad_(False).eval()
    model.register_buffer("_ema_updates", torch.zeros((), dtype=torch.long))
    return model


def _plain_state(model: EMAStandardModel):
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("_ema_model.") and key != "_ema_updates"
    }


def test_ema_initialization_is_fp32_and_matches_online() -> None:
    """EMA initialization copies online weights in FP32 with zero updates."""
    model = _fixture()
    model.copy_online_to_ema()
    online = dict(model.named_parameters())
    for name, ema_parameter in model._ema_model.module.named_parameters():
        assert ema_parameter.dtype == torch.float32
        assert torch.equal(ema_parameter, online[name].float())
    assert model._ema_updates.item() == 0


def test_ema_update_arithmetic_and_count() -> None:
    """One update applies configured decay and increments both counters."""
    model = _fixture(decay=0.75)
    model.copy_online_to_ema()
    old = model._ema_model.module.online.weight.detach().clone()
    with torch.no_grad():
        model.online.weight.add_(4.0)
    model.update_ema()
    assert torch.allclose(model._ema_model.module.online.weight, old + 1.0)
    assert model._ema_updates.item() == 1
    assert model._ema_model.n_averaged.item() == 1


def test_optimizer_contains_only_trainable_online_parameters() -> None:
    """Optimizer parameter groups exclude the frozen EMA copy."""
    model = _fixture()
    optimizer = model.configure_optimizers()["optimizer"]
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    expected = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("_ema_model.")
    ]
    assert len(actual) == len(expected)
    assert all(a is e for a, e in zip(actual, expected))


def test_validation_metric_tasks_select_ema_copy() -> None:
    """Validation metrics resolve the EMA task while training uses online state."""
    model = _fixture()
    assert model.metric_tasks_for_phase("train") is model._tasks
    assert model.metric_tasks_for_phase("val") is model._ema_model.module._tasks


def test_validation_step_uses_ema_weights(monkeypatch) -> None:
    """Validation delegates loss calculation to the averaged model."""
    model = _fixture()
    with torch.no_grad():
        model.online.weight.fill_(1.0)
        model._ema_model.module.online.weight.fill_(7.0)

    def ema_shared_step(ema_module, batch, batch_idx):
        """Return a weight-dependent scalar to identify the selected model."""
        return ema_module.online.weight.sum()

    model._ema_model.module.shared_step = MethodType(
        ema_shared_step, model._ema_model.module
    )
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_get_batch_size", lambda batch: 1)
    loss = model.validation_step([], 0)
    assert loss.item() == pytest.approx(14.0)


def test_checkpoint_metadata_restores_ema_decay() -> None:
    """Checkpoint metadata overrides a newly configured EMA decay."""
    model = _fixture(decay=0.999)
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    restored = _fixture(decay=0.5)
    restored.on_load_checkpoint(checkpoint)
    assert restored.ema_decay == pytest.approx(0.999)


def test_plain_source_checkpoint_initializes_online_and_ema() -> None:
    """A plain source checkpoint initializes both online and averaged weights."""
    source = _fixture()
    with torch.no_grad():
        source.online.weight.fill_(3.25)
    plain = _plain_state(source)
    restored = _fixture()
    restored.load_source_state_dict(plain)
    assert torch.equal(restored.online.weight, plain["online.weight"])
    assert torch.equal(restored._ema_model.module.online.weight, plain["online.weight"])
    assert restored._ema_updates.item() == 0


def test_full_resume_round_trip_restores_model_optimizer_scheduler_and_count() -> None:
    """Full resume state round-trips weights, optimizer, scheduler, and count."""
    model = _fixture()
    optimizer = model.configure_optimizers()["optimizer"]
    scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=0.01)
    loss = model.online(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    model.update_ema()
    checkpoint = {
        "state_dict": deepcopy(model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        "scheduler": deepcopy(scheduler.state_dict()),
    }

    restored = _fixture()
    restored_optimizer = restored.configure_optimizers()["optimizer"]
    restored_scheduler = CosineAnnealingLR(restored_optimizer, T_max=8, eta_min=0.01)
    restored.load_state_dict(checkpoint["state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_scheduler.load_state_dict(checkpoint["scheduler"])
    assert restored._ema_updates.item() == 1
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


@pytest.mark.parametrize("initial_lr", [2e-6, 6.25e-6, 2e-5])
def test_cosine_scheduler_reaches_ten_percent_floor(initial_lr: float) -> None:
    """Eight cosine epochs decrease each pilot learning rate to its 10% floor."""
    parameter = nn.Parameter(torch.ones(()))
    optimizer = AdamW([parameter], lr=initial_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=0.1 * initial_lr)
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert all(left > right for left, right in zip(values, values[1:]))
    assert values[-1] == pytest.approx(0.1 * initial_lr)


def test_inference_extraction_prefers_ema_and_preserves_plain_checkpoints() -> None:
    """Inference extraction preserves plain state and prefers EMA when present."""
    plain = {"weight": torch.tensor([1.0]), "bias": torch.tensor([2.0])}
    assert extract_inference_state_dict({"state_dict": plain}) == plain

    ema_checkpoint = {
        "state_dict": {
            **plain,
            "_ema_updates": torch.tensor(7),
            "_ema_model.n_averaged": torch.tensor(7),
            "_ema_model.module.weight": torch.tensor([3.0]),
            "_ema_model.module.bias": torch.tensor([4.0]),
        }
    }
    extracted = extract_inference_state_dict(ema_checkpoint)
    assert torch.equal(extracted["weight"], torch.tensor([3.0]))
    assert torch.equal(extracted["bias"], torch.tensor([4.0]))
    assert all(not key.startswith("_ema_model") for key in extracted)
