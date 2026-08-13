"""Reproduce token-drop behavior on synthetic graph events."""

import torch
import sys
import os

# Add current directory to path so src imports work
sys.path.append(os.getcwd())

try:
    from src.models.transformer import IceMix
except ImportError:
    # Try adding subdirectory if running from top level
    sys.path.append("ice_mix")
    from src.models.transformer import IceMix

from torch_geometric.data import Data, Batch


def test_token_drop():
    """Print retained pulse/event counts for a forced synthetic drop pass."""
    print("Initializing IceMix model with token_drop=0.5...")
    # settings
    seq_length = 48
    batch_size = 4
    feature_dim = 6

    model = IceMix(
        seq_length=seq_length,
        hidden_dim=32,
        n_features=feature_dim,
        token_drop=0.5,  # 50% drop
        include_dynedge=False,
        head_size=4,
        depth=2,
        depth_rel=2,
    )

    # Create dummy data
    # Data object usually has x, batch, etc.
    # IceMix expects data.x to be (Num_pulses, Features)
    # The batch vector assigns each pulse to an event in the batch.

    # Create fully filled events to ensure we have tokens to drop
    num_pulses_per_event = seq_length
    total_pulses = num_pulses_per_event * batch_size

    x = torch.randn(total_pulses, feature_dim)
    batch_vec = torch.cat(
        [torch.full((num_pulses_per_event,), i) for i in range(batch_size)]
    )

    # We need to ensure batch_vec is long
    batch_vec = batch_vec.long()

    data = Data(x=x, batch=batch_vec)
    # Mocking Batch structure slightly if needed, but Data with batch attribute is standard input for GNNs

    print("Testing Eval Mode (Should be deterministic)...")
    model.eval()
    with torch.no_grad():
        out1 = model(data)
        out2 = model(data)

    if torch.allclose(out1, out2):
        print("PASS: Eval mode is deterministic.")
    else:
        print("FAIL: Eval mode is NOT deterministic.")
        diff = (out1 - out2).abs().sum()
        print(f"Diff: {diff}")

    print("Testing Train Mode (Should be stochastic due to token_drop)...")
    model.train()
    # We don't use no_grad here to simulate training forward pass, but we won't backward
    out3 = model(data)
    out4 = model(data)

    if not torch.allclose(out3, out4):
        print("PASS: Train mode is stochastic (tokens are being dropped).")
        # Optional: check if reasonable magnitude of difference?
    else:
        print("FAIL: Train mode is deterministic (tokens NOT dropped or lucky seed).")
        # If token_drop is 0.5, probability of exact match is very low unless input is tiny or effective drop is 0.

    print("Done verification.")


if __name__ == "__main__":
    test_token_drop()
