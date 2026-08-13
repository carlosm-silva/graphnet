"""Compose Hydra configuration and print token-drop diagnostic values."""

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os

# Add local directory to path for imports if needed
sys.path.append(os.getcwd())


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def verify_config(cfg: DictConfig) -> None:
    """Print the configured and effective token-drop values.

    Parameters
    ----------
    cfg : omegaconf.DictConfig
        Configuration composed from ``conf/`` plus CLI overrides.

    Notes
    -----
    Use Hydra's ``--cfg job --resolve`` flags to print the complete resolved
    configuration without entering this function.
    """
    print(f"Testing configuration: {cfg.data.get('token_drop', 'NOT SET')}")

    # Simulate logic in train.py
    token_drop = cfg.data.get("token_drop", cfg.attention.get("token_drop", 0.0))
    print(f"Resolved token_drop: {token_drop}")


if __name__ == "__main__":
    verify_config()
