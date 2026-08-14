# Reduced single-GPU smoke test

[`single_gpu_smoke.yaml`](single_gpu_smoke.yaml) is a documentation-only record
of reduced Hydra overrides: one GPU, batch size 2, at most 64 pulses, a
64-channel two-block model, two train/validation batches, no W&B, no loader
workers, and FP32. It does not modify or shadow the production config.

This test checks configuration composition, three-database loading, graph
construction, forward/backward execution, validation, logging, and checkpoint
writing. It does not test physics quality or production-scale DDP. Run it from
the GraphNeT checkout root in the compatible environment, with one CUDA GPU
visible and `DATA_ROOT` containing the three exact basenames in
`conf/data/standard.yaml`.

```bash
export DATA_ROOT=/path/to/prepared/sqlite
export CUDA_VISIBLE_DEVICES=0

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
```

The command is a documentation-only invocation of the existing training entry
point; no wrapper or new executable helper is supplied. Its overrides are
**VERIFIED-STATIC**. The same composition built a 258K-parameter model and
entered Lightning locally on 2026-08-13 (**VERIFIED-LOCAL**), but that laptop
had no visible GPU and the run stopped before its first batch when
`EpochMonitorCallback.on_train_epoch_start` queried
`torch.cuda.current_device()`. Completion remains **UNVERIFIED-CLUSTER** or
otherwise unverified on a suitable CUDA workstation.

For a successful run, the statically expected artifacts are a timestamped
`ice_mix/outputs/IceMix-Local-Smoke_*_job-local/` directory containing CSV logs
below `logs/training_logs/` and `checkpoints/last.ckpt` plus any retained
`best-*` checkpoint. Hydra writes its own run metadata under its configured
Hydra output tree, while IceMix also prints the resolved configuration into the
run log. Retain both. Confirm that the metrics CSV has at least one finite
training and validation loss and that the checkpoint files are nonempty. These
artifacts have not been produced by this reduced command on the documentation
laptop; do not fabricate or assume their values.

To inspect the suggested overrides without training, compare the YAML with a
resolved Hydra configuration:

```bash
sed -n '1,200p' ice_mix/docs/examples/single_gpu_smoke.yaml
python ice_mix/verify_config.py --cfg job --resolve
```

A successful reduced run checks wiring, not physics quality, production data
provenance, pulse-selection reproducibility, resume correctness, or eight-GPU
DDP behavior.
