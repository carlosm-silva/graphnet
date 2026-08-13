# Tutorial 3 — Base training and resume

**Goal:** submit a recommended base IceMix run, monitor it, and resume safely from `last.ckpt`.

**Prerequisites:** Tutorials 1–2, a successor-owned launcher, valid allocation, all three databases, and W&B access or `wandb=false`.

**Expected duration:** 30 minutes to prepare; queue and training time are **UNVERIFIED-CLUSTER**. Launchers request up to 72 hours.

**Run on:** submit/monitor from a Phoenix login node; computation runs on one allocated 8×L40S node.

**Verification:** launcher resources and code path are **VERIFIED-STATIC**. Multi-GPU success, timing, and expected loss ranges are **UNVERIFIED-CLUSTER**.

## 1. Select the scientific variant

| Question | `ICE_MIX_DATA_CONFIG` / project label |
|---|---|
| Baseline pulses | `standard` / `IceMix-Standard` |
| Test robustness through training-time token removal | `drop` / `IceMix-Drop` |
| Enforce azimuthal symmetry by augmentation | `augmented_rotation` / `IceMix-Augmented-Rotation` |
| Combine both | `drop_aug_rot` / `IceMix-Drop-Augmented-Rotation` |

These are base variants, not successive required stages. Choose based on the physics study.

## 2. Configure the supported launcher

Use `run_training.sbatch`; it is the supported base path and stages all three
database basenames with fail-fast validation. Validate every account and node
value listed in [the job catalog](../job-scripts.md). Keep the requested
one-node/eight-L40S geometry unless intentionally running a reduced test.

Set secrets and paths outside Git:

```bash
export DATA_ROOT=/successor/project/path/to/prepared/sqlite
export WANDB_ENTITY=your-entity-if-needed
export ICE_MIX_WANDB_ROOT=/successor/scratch/path/to/ice_mix_wandb
```

Do not put W&B credentials or `.env` content in the launcher.

## 3. Resolve the intended config

For a rotation run:

```bash
python ice_mix/verify_config.py --cfg job --resolve \
    data=augmented_rotation project_name=IceMix-Augmented-Rotation
```

Inspect model dimensions, all three paths, split mode, batch size, precision, and output directories before allocating GPUs. **VERIFIED-LOCAL:** this exact Hydra composition succeeded with the staged data root on 2026-08-13.

## 4. Submit

```bash
job_id=$(ICE_MIX_DATA_CONFIG=augmented_rotation \
    ICE_MIX_PROJECT_NAME=IceMix-Augmented-Rotation \
    sbatch --parsable ice_mix/run_training.sbatch)
printf 'submitted %s\n' "$job_id"
```

The literal successor paths remain a statically reconstructed template and the
full eight-GPU submission is **UNVERIFIED-CLUSTER**. Older variant launchers are
preserved as historical workflow evidence, not the recommended entry point.

## 5. Monitor without interfering

```bash
squeue -j "$job_id"
scontrol show job "$job_id"
sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode,AllocTRES,MaxRSS
tail -f "IceMixTraining-${job_id}.out"
```

Check for database staging, healthy GPUs, DDP rank initialization, finite loss, validation epochs, and checkpoint writes. Queue waiting is normal and is not a code failure.

## 6. Locate durable artifacts

The default run directory is:

```text
ice_mix/outputs/<project>_<YYYY-MM-DD>_<HH-MM-SS>_job-<SLURM_JOB_ID>/
├── checkpoints/
│   ├── best-*.ckpt
│   └── last.ckpt
└── logs/training_logs/version_*/metrics.csv
```

W&B runtime files appear below `ICE_MIX_WANDB_ROOT`, or below
`ice_mix/outputs/wandb_runtime` when it is unset. A staged real run bundle
confirmed the checkpoint and CSV layout (**VERIFIED-LOCAL**, 2026-08-13);
future Phoenix output remains **UNVERIFIED-CLUSTER**.

## 7. Resume

Resume only from an explicitly inspected `last.ckpt` and its existing W&B run
ID. The supported launcher refuses an implicit “newest run” choice and verifies
that the source run still has `.hydra/config.yaml`.

```bash
ICE_MIX_DATA_CONFIG=augmented_rotation \
ICE_MIX_PROJECT_NAME=IceMix-Augmented-Rotation \
ICE_MIX_CKPT_PATH=/successor/project/path/to/run/checkpoints/last.ckpt \
ICE_MIX_WANDB_RUN_ID=the-existing-wandb-id \
    sbatch ice_mix/run_training.sbatch
```

Read the log line naming the checkpoint, run directory, and W&B ID. The W&B ID
can be found in the inherited run metadata; if it cannot be established, ask
the author on Slack rather than silently creating a duplicate continuation.

## Common failures

- **OOM:** production settings assume L40S capacity; confirm allocation and per-rank batch semantics.
- **Walltime:** `last.ckpt` must be in project storage, then submit resume manually.
- **Wrong W&B continuation:** inspect recovered metadata and `WANDB_RESUME`; do not accept a duplicate silently.
- **Bad GPU:** health filtering may reduce `NPROC`; verify DDP/world size and report repeat faults to PACE.
- **Sampler assertion:** rank partition coverage differed from the dataset; preserve the full report for diagnosis.

## What to try next

After at least one usable checkpoint, continue to [Tutorial 4](04_inference_evaluation_plots.md).
