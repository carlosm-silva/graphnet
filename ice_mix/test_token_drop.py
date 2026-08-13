"""Regression tests for IceMix token dropping."""

from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from ice_mix.src.models.transformer import IceMix
from ice_mix.src.utils import TokenDropSeedCallback


def _make_model(token_drop: float = 0.5) -> IceMix:
    model = IceMix(
        seq_length=48,
        hidden_dim=32,
        n_features=6,
        token_drop=token_drop,
        drop_chance=1.0,
        include_dynedge=False,
        head_size=4,
        depth=2,
        depth_rel=2,
        n_rel=2,
    )
    model.train()
    return model


def _make_data(pulses_per_event=(16, 16, 16, 16)) -> Data:
    batch = torch.cat(
        [
            torch.full((count,), event_idx, dtype=torch.long)
            for event_idx, count in enumerate(pulses_per_event)
        ]
    )
    return Data(x=torch.randn(batch.numel(), 6), batch=batch)


def test_token_drop_preserves_every_event_and_input_batch() -> None:
    """Repeated forwards must retain all events without mutating input data."""
    model = _make_model(token_drop=1.0)

    pulses_per_event = (1, 2, 3, 4)
    batch = torch.cat(
        [
            torch.full((count,), event_idx, dtype=torch.long)
            for event_idx, count in enumerate(pulses_per_event)
        ]
    )
    data = Data(x=torch.randn(batch.numel(), 6), batch=batch)
    original_x = data.x.clone()
    original_batch = data.batch.clone()

    for _ in range(5):
        output = model(data)
        assert output.shape[0] == len(pulses_per_event)
        assert torch.equal(data.x, original_x)
        assert torch.equal(data.batch, original_batch)


def test_token_drop_is_fixed_within_batch_and_changes_between_epochs() -> None:
    """Closure re-evaluations reuse a mask, while later epochs do not."""
    model = _make_model()
    data = _make_data()
    callback = TokenDropSeedCallback(seed=42)
    module = SimpleNamespace(backbone=model)
    trainer = SimpleNamespace(current_epoch=0, global_rank=0)

    callback.on_train_batch_start(trainer, module, data, batch_idx=3)
    epoch_zero_seed = model.token_drop_seed
    first = model(data)
    second = model(data)

    trainer.current_epoch = 1
    callback.on_train_batch_start(trainer, module, data, batch_idx=3)
    epoch_one_seed = model.token_drop_seed
    third = model(data)

    assert torch.equal(first, second)
    assert epoch_zero_seed != epoch_one_seed
    assert not torch.equal(first, third)


def test_seeded_token_drop_does_not_consume_global_rng() -> None:
    """The dedicated mask generator must not perturb other randomness."""
    model = _make_model()
    data = _make_data()
    model.set_token_drop_seed(1234)

    torch.manual_seed(99)
    expected = torch.rand(8)
    torch.manual_seed(99)
    model(data)
    actual = torch.rand(8)

    assert torch.equal(actual, expected)
