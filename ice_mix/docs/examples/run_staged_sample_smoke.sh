#!/bin/sh
# Local-only application smoke using the ignored handoff fixture directory.
# Run from the GraphNeT checkout root with the compatible environment active.
set -eu

sample_root="ice_mix/docs/_local_sample"
if [ ! -f "$sample_root/standard_sqlite/numu.db" ]; then
    echo "The ignored local sample is not staged at $sample_root." >&2
    exit 2
fi

python ice_mix/train.py \
    project_name=IceMix-Staged-Sample-Smoke \
    wandb=false \
    precision=32-true \
    max_epochs=1 \
    early_stopping_patience=0 \
    limit_train_batches=1 \
    limit_val_batches=1 \
    num_workers=0 \
    data/split=csv \
    "data.path=[$sample_root/standard_sqlite/numu.db,$sample_root/standard_sqlite/nue.db,$sample_root/standard_sqlite/nutau.db]" \
    "data.split.train_csvs=[$sample_root/csv_selections/numu_train.csv,$sample_root/csv_selections/nue_train.csv,$sample_root/csv_selections/nutau_train.csv]" \
    "data.split.val_csvs=[$sample_root/csv_selections/numu_validation.csv,$sample_root/csv_selections/nue_validation.csv,$sample_root/csv_selections/nutau_validation.csv]" \
    data.split.test_csvs=null \
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
