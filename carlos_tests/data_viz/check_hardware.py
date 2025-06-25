#!/usr/bin/env python
"""
check_hardware.py – quick system + version check for PyTorch Lightning projects
"""

from datetime import datetime
import os
import platform
import sys

import torch
import pytorch_lightning as pl

try:
    import psutil   # optional but nicer CPU info
except ImportError:
    psutil = None


def main() -> None:
    print("=" * 60)
    print(f"🔎  PyTorch-Lightning Hardware Check – {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)
    print(f"Python               : {platform.python_version()} ({sys.executable})")
    print(f"PyTorch              : {torch.__version__}")
    print(f"PyTorch Lightning    : {pl.__version__}")
    print("-" * 60)

    # ---- GPU section -------------------------------------------------------
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA GPUs detected   : {num_gpus}")
        for idx in range(num_gpus):
            props = torch.cuda.get_device_properties(idx)
            mem_gb = props.total_memory / 1024 ** 3
            print(f"  ▸ GPU {idx}: {props.name} — {mem_gb:.1f} GiB")
    else:
        print("CUDA GPUs detected   : none (running on CPU or other accelerator)")

    # ---- CPU section -------------------------------------------------------
    if psutil:
        print(f"Logical CPU cores    : {psutil.cpu_count(logical=True)}")
        print(f"Physical CPU cores   : {psutil.cpu_count(logical=False)}")
    else:
        # fall back to standard library
        print(f"Logical CPU cores    : {os.cpu_count()}  (install psutil for details)")

    # ---- Accelerator hint for Lightning -----------------------------------
    print("-" * 60)
    trainer_hint = "pl.Trainer(accelerator='auto', devices='auto')"
    print(f"ℹ️  To let Lightning pick the best accelerator:\n    {trainer_hint}")

    print("=" * 60)


if __name__ == "__main__":
    main()
