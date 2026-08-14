# Tutorial 3 — Preparing base training and resume

**Goal:** understand the inherited production launch path, perform every
documentation-only preflight, and hand the remaining launcher/resume decisions
to the researcher before consuming an allocation.

**Prerequisites:** Tutorials 1–2, PACE access, all three databases, the exported
environment, and a researcher responsible for software changes.

**Expected duration:** 30–60 minutes for inspection. Queue and training time are
**UNVERIFIED-CLUSTER**; inherited launchers request up to 72 hours.

**Run on:** read-only inspection and Hydra composition on a Phoenix login node;
training only from a researcher-reviewed Slurm script on an allocated 8×L40S
node.

**Verification:** resources, config composition, and inherited script behavior
are **VERIFIED-STATIC**. No inherited base launcher is certified submit-ready.

## 1. Choose the scientific family

| Scientific intent | Hydra data group | Historical base launcher |
|---|---|---|
| Baseline pulses | `standard` | `run_standard.sbatch` or `run_training.sbatch` |
| On-the-fly azimuthal rotation | `augmented_rotation` | `run_augmented.sbatch` |
| Training-time token removal | `drop` | `run_drop.sbatch` |
| Rotation plus token removal | `drop_aug_rot` | `run_drop_aug_rot.sbatch` |

“Augmented” here means current on-the-fly rotation. Stored fixed rotations are
deprecated. These are alternative experiments, not successive stages.

## 2. Read the launcher before submitting

```bash
sed -n '1,240p' ice_mix/run_training.sbatch
sed -n '1,240p' ice_mix/run_standard.sbatch
```

Stop and involve the researcher if the intended launcher still has any of the
following inherited properties:

- author-owned checkout, output, cache, account, or email values;
- only nu_mu and nu_e copied even though the config uses nu_tau too;
- background copies without per-file status and destination checks;
- a fallback to shared `/tmp` instead of required Slurm-local storage;
- a fixed rank count that can disagree with the healthy visible GPUs;
- automatic newest-run/checkpoint discovery;
- no comparison between the saved and proposed resolved configuration.

All are present somewhere in the inherited base/resume family. This tutorial
flags them; it does not supply a software patch.

## 3. Resolve the intended Hydra configuration

For a rotation run:

```bash
export DATA_ROOT=/successor/project/path/to/prepared/sqlite
python ice_mix/verify_config.py --cfg job --resolve \
    data=augmented_rotation project_name=IceMix-Augmented-Rotation
```

Save the output for the researcher. Inspect the three database paths, split
mode, pulse cap, batch size, precision, optimizer, encoder constants, output
directories, rotation seed, and W&B identity. This composition succeeded with
the staged sample root locally (**VERIFIED-LOCAL**, 2026-08-13); it does not
validate a Slurm script.

## 4. Researcher approval gate

Before submission, the researcher should provide or approve a launcher and
record:

```text
launcher path and Git revision
resolved Hydra configuration
three source database identities
allocation/QoS and requested node/GPU/CPU/memory/walltime
durable output/checkpoint directory
W&B project/entity/run policy
expected rank count
```

Only then submit the approved file:

```bash
job_id=$(sbatch --parsable /path/to/researcher-reviewed-launcher.sbatch)
printf 'submitted %s\n' "$job_id"
```

The placeholder is intentional. Replacing it with an inherited launcher
without researcher review defeats the safety gate.

## 5. Monitor without interfering

```bash
squeue -j "$job_id"
scontrol show job "$job_id"
sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode,AllocTRES,MaxRSS
```

Inspect the actual Slurm output path declared by the approved launcher. A
healthy run should show all three staged inputs, the intended visible GPU/rank
count, the resolved config, dataset construction, DDP initialization, finite
losses, validation, and new checkpoints. A Slurm `COMPLETED` state is necessary
but not sufficient; confirm artifacts directly.

## 6. Locate durable artifacts

The inherited training configuration normally creates:

```text
ice_mix/outputs/<project>_<YYYY-MM-DD>_<HH-MM-SS>_job-<SLURM_JOB_ID>/
├── .hydra/config.yaml
├── checkpoints/
│   ├── best-*.ckpt
│   └── last.ckpt
└── logs/training_logs/version_*/metrics.csv
```

A staged real bundle confirms this shape (**VERIFIED-LOCAL**, 2026-08-13).
Confirm that the real directory is durable project storage rather than
`$TMPDIR` before relying on resume.

## 7. Treat resume as a scientific consistency check

The inherited resume paths restore Lightning state but do not prove that the
newly composed model/data configuration equals the interrupted run. Before the
researcher approves resume:

1. identify `last.ckpt` and the original W&B run without “newest” guessing;
2. preserve the source `.hydra/config.yaml`;
3. resolve the proposed resume config without running training;
4. compare data paths/versions, split, augmentation, architecture numerics,
   optimizer/scheduler, precision, batch/worker settings, and output identity;
5. have the researcher decide which runtime-only differences are allowed.

Do not describe the existing resume path as safe or config-validated merely
because the checkpoint loads.

## Common failures

- **Missing tau staging:** configured reads can fall back to project storage or
  fail after the launcher reports that copying completed.
- **OOM:** production settings assume L40S capacity and a per-rank batch of 256.
- **Walltime:** verify that `last.ckpt` is durable before any resubmission.
- **Wrong continuation:** newest-directory discovery can select an unintended
  run; saved/current configs can drift silently.
- **Bad GPU:** compare visible devices with world size; retain the complete log.
- **NaN loss:** treat it as failed scientific output even if a process continues.

## What to try next

After a researcher-approved run has produced an inspected checkpoint, continue
to [Tutorial 4](04_inference_evaluation_plots.md).
