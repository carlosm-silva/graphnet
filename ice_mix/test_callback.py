"""Manually diagnose the random-rotation Lightning callback."""

import sys
import torch
import numpy as np
from src.utils import RandomRotationCallback

class DummyBatch:
    """Synthetic two-event batch for the rotation diagnostic."""

    def __init__(self):
        """Create pulse coordinates and joint/scalar truth tensors."""
        # 2 graphs in batch
        self.batch = torch.tensor([0, 0, 1, 1])
        # [x, y, z, time, charge]
        self.x = torch.tensor([
            [1.0, 0.0, 10.0, 1.0, 1.0],
            [0.0, 1.0, 20.0, 2.0, 1.0],
            [1.0, 1.0, 15.0, 1.5, 1.0],
            [-1.0, 0.0, 5.0, 0.5, 1.0]
        ])
        
        # [pos_x, pos_y, pos_z, dir_x, dir_y, dir_z]
        # graph 0 expected dir: (1, 0, 0)
        # graph 1 expected dir: (0, 1, 0)
        self.joint_labels = torch.tensor([
            [0.0, 0.0, -100.0, 1.0, 0.0, 0.0],
            [10.0, 0.0, -50.0, 0.0, 1.0, 0.0]
        ])
        self.azimuth = torch.tensor([0.0, np.pi/2])
        self.position_x = torch.tensor([0.0, 10.0])
        self.position_y = torch.tensor([0.0, 0.0])

batch = DummyBatch()

# Before rotation
print("Before Rotation:")
print("x_y: ", batch.x[:, :2])
print("pos_x_y: ", batch.joint_labels[:, :2])
print("dir_x_y: ", batch.joint_labels[:, 3:5])

callback = RandomRotationCallback(seed=42)
callback._apply_rotation(batch)

print("\nAfter Rotation:")
print("x_y: ", batch.x[:, :2])
print("pos_x_y: ", batch.joint_labels[:, :2])
print("dir_x_y: ", batch.joint_labels[:, 3:5])
print("azimuth:", batch.azimuth)
