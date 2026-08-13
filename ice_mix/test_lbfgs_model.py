"""Tests for deterministic, distributed LBFGS closure support."""

import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS

from ice_mix.src.models.lbfgs_model import (
    DistributedLBFGSStandardModel,
    freeze_for_last_blocks_fine_tuning,
    seeded_model_rng,
    synchronize_loss_value,
)


def test_seeded_model_rng_repeats_dropout_and_restores_rng() -> None:
    """Repeated closures share dropout masks without consuming global RNG."""
    dropout = nn.Dropout(p=0.5)
    dropout.train()
    inputs = torch.ones(128)

    torch.manual_seed(99)
    expected_next = torch.rand(8)
    torch.manual_seed(99)

    with seeded_model_rng(1234, torch.device("cpu")):
        first = dropout(inputs)
    with seeded_model_rng(1234, torch.device("cpu")):
        second = dropout(inputs)

    actual_next = torch.rand(8)
    assert torch.equal(first, second)
    assert torch.equal(actual_next, expected_next)


def test_synchronize_loss_value_preserves_gradient(monkeypatch) -> None:
    """The synchronized value must not alter the local autograd path."""
    loss = torch.tensor(2.0, requires_grad=True)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op) -> None:
        """Emulate a two-rank summed loss value in place."""
        value.add_(6.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    synchronized = synchronize_loss_value(loss)
    synchronized.backward()

    assert synchronized.item() == 4.0
    assert loss.grad.item() == 1.0


class _FineTuneFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.input = nn.Linear(3, 3)
        self.backbone.blocks = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
        self._tasks = nn.ModuleList([nn.Linear(3, 1)])


def test_freeze_for_last_blocks_fine_tuning_selects_expected_parameters() -> None:
    """Only the requested final blocks and prediction tasks remain trainable."""
    model = _FineTuneFixture()

    trainable, total = freeze_for_last_blocks_fine_tuning(model, 1)

    expected = sum(p.numel() for p in model.backbone.blocks[-1].parameters())
    expected += sum(p.numel() for p in model._tasks.parameters())
    assert trainable == expected
    assert total == sum(p.numel() for p in model.parameters())
    assert not any(p.requires_grad for p in model.backbone.input.parameters())
    assert not any(p.requires_grad for p in model.backbone.blocks[0].parameters())
    assert all(p.requires_grad for p in model.backbone.blocks[-1].parameters())
    assert all(p.requires_grad for p in model._tasks.parameters())


def test_lbfgs_optimizer_excludes_frozen_parameters() -> None:
    """Frozen parameters must not consume flattened LBFGS history storage."""
    model = object.__new__(DistributedLBFGSStandardModel)
    nn.Module.__init__(model)
    model.frozen = nn.Linear(2, 2)
    model.trainable = nn.Linear(2, 1)
    model.frozen.requires_grad_(False)
    model._optimizer_class = LBFGS
    model._optimizer_kwargs = {"max_iter": 1}
    model._scheduler_class = None
    model._scheduler_kwargs = {}
    model._scheduler_config = {}

    optimizer = model.configure_optimizers()["optimizer"]
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected_parameters = list(model.trainable.parameters())

    assert len(optimizer_parameters) == len(expected_parameters)
    assert all(
        actual is expected
        for actual, expected in zip(optimizer_parameters, expected_parameters)
    )


def test_optimizer_step_resets_only_previous_batch_state() -> None:
    """Each mini-batch builds an independent within-batch LBFGS history."""
    model = object.__new__(DistributedLBFGSStandardModel)
    nn.Module.__init__(model)
    parameter = nn.Parameter(torch.tensor([5.0]))
    optimizer = LBFGS(
        [parameter],
        lr=0.1,
        max_iter=4,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )

    def run_step(target: float) -> None:
        """Perform one LBFGS batch toward a scalar target."""
        def closure() -> torch.Tensor:
            """Recompute the differentiable objective for LBFGS."""
            optimizer.zero_grad()
            loss = (parameter - target).square().sum()
            loss.backward()
            return loss

        model.optimizer_step(0, 1, optimizer, closure)

    run_step(0.0)
    first_state = optimizer.state[parameter]
    assert first_state["n_iter"] > 1
    assert len(first_state["old_dirs"]) > 0
    first_state["stale_history"] = object()

    run_step(2.0)
    second_state = optimizer.state[parameter]
    assert "stale_history" not in second_state
    assert 1 < second_state["n_iter"] <= optimizer.param_groups[0]["max_iter"]
    assert len(second_state["old_dirs"]) > 0


def _run_distributed_lbfgs(rank: int, world_size: int, init_file: str) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(7)
        model = DistributedDataParallel(nn.Linear(1, 1, bias=False))
        optimizer = LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=4,
            history_size=4,
            line_search_fn="strong_wolfe",
            tolerance_grad=0.0,
            tolerance_change=0.0,
        )
        inputs = torch.tensor([[1.0 + rank]])
        targets = torch.tensor([[3.0 - rank]])

        def closure() -> torch.Tensor:
            """Compute a rank-local loss with synchronized scalar value."""
            optimizer.zero_grad()
            with seeded_model_rng(100 + rank, torch.device("cpu")):
                prediction = model(inputs)
                local_loss = (prediction - targets).square().mean()
            loss = synchronize_loss_value(local_loss)
            loss.backward()
            return loss

        optimizer.step(closure)

        parameter = next(model.parameters()).detach()
        gathered = [torch.empty_like(parameter) for _ in range(world_size)]
        dist.all_gather(gathered, parameter)
        assert all(torch.equal(gathered[0], value) for value in gathered[1:])
    finally:
        dist.destroy_process_group()


def test_strong_wolfe_completes_with_rank_local_losses() -> None:
    """Real DDP closures must follow one synchronized line-search path."""
    fd, init_file = tempfile.mkstemp(prefix="icemix-lbfgs-ddp-")
    os.close(fd)
    os.unlink(init_file)
    try:
        mp.spawn(_run_distributed_lbfgs, args=(2, init_file), nprocs=2)
    finally:
        if os.path.exists(init_file):
            os.unlink(init_file)
