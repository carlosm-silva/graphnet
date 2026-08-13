#!/bin/sh
# Documentation-only wiring check; do not use this as a production launcher.
set -eu

if [ -z "${DATA_ROOT:-}" ]; then
    echo "DATA_ROOT must point to a directory containing the three configured SQLite files." >&2
    exit 43
fi

if [ ! -f "ice_mix/train.py" ]; then
    echo "Run this command from the GraphNeT repository root." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python ice_mix/train.py \
    project_name=IceMix-Local-Smoke \
    wandb=false \
    precision=32-true \
    max_epochs=1 \
    early_stopping_patience=0 \
    limit_train_batches=2 \
    limit_val_batches=2 \
    num_workers=0 \
    data.batch_size=2 \
    data.pin_memory=false \
    data.persistent_workers=false \
    data.prefetch_factor=null \
    data.max_pulses=64 \
    attention.hidden_dim=64 \
    attention.seq_length=32 \
    attention.depth=2 \
    attention.n_rel=2 \
    attention.head_size=16
