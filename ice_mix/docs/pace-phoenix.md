# Running IceMix on PACE Phoenix

Phoenix is the canonical IceMix runtime. The author's TensorBook is only a documentation and reduced-verification workstation; none of its driver, CUDA, conda-prefix, or filesystem details are production requirements.

## Account and login

1. Obtain a Georgia Tech PACE account and authorization for the successor's allocation.
2. Follow the current official [PACE documentation](https://docs.pace.gatech.edu/) for VPN, login host, MFA, and data-transfer instructions.
3. Log into a Phoenix login node. Do not run training on the login node.
4. Clone or copy the repository into the successor's project space.
5. Send the original author a Slack message for the location of his PACE conda files and for unresolved project history.
6. Ask Jiyuan on Slack for current PACE procedures, database access, provenance, or units when the repository is insufficient.

The need for a PACE account and Slurm submission is **VERIFIED-STATIC**. Exact hostname/onboarding commands are **UNVERIFIED-CLUSTER** because they can change; use current PACE documentation rather than copying an old host from notes.

The L40S allocation is restricted to L40S nodes and should be prioritized for IceMix. **Author-confirmed 2026-08-13.**

## Storage model

Use this ownership rule:

| Tier | IceMix use | Persistence |
|---|---|---|
| Project storage | Repository, authoritative prepared SQLite inputs, checkpoints, logs, predictions | Durable/group-managed |
| Job-local `$TMPDIR` | Per-job copy of SQLite inputs for fast reads | Deleted after the job |
| Scratch | W&B caches and reproducible transient products | Purgeable; not authoritative |
| Home | Small personal config only | Quota-limited; not the production dataset location |

This project → `$TMPDIR` workflow is author-confirmed 2026-08-13. Exact quotas, purge policy, and successor paths are **UNVERIFIED-CLUSTER**; confirm them with `pace-quota` or current PACE guidance on first login.

The checked-in `/storage/project/r-itaboada3-0/cfilho3`,
`/storage/scratch1/8/cfilho3`, `gts-pli77-ideas_l40s`, and related identifiers
are an intentional map of the inherited setup; the author says the successor
will receive equivalent permissions. Validate access before relying on them.
The successor's email, user-owned output/cache directories, and any path that
must be writable **must be changed by the new user**. The L40S allocation may
remain only if `sacctmgr`/PACE confirms authorization.

## Rebuild the exported environment

The current author-confirmed runtime commands are:

```bash
module load anaconda3/2022.05.0.1
conda activate graphnet
```

The exact author export and a sanitized, prefix-free bootstrap now live in
[`docs/graphnet_env/`](graphnet_env/README.md). Follow that page from the
GraphNeT checkout root. It pins the custom GraphNeT source commit and avoids the
nonportable personal prefix and `file://` URLs in the raw forensic exports.
**VERIFIED-STATIC; environment supplied by the author 2026-08-13.**

`ice_mix/requirements.txt` now delegates to the same pinned PACE pip
specification and intentionally omits a public `graphnet` requirement. Install
this checkout separately with `--no-deps --editable .` as shown in the recipe;
silently substituting a public GraphNeT release does not satisfy the custom
boundary.

For reference only, a separate local CPU verification environment succeeded with Python 3.11,
PyTorch `2.2.0+cu118`, the matching PyG extension wheels, and
`torch-geometric==2.5.3`. Leaving PyG unconstrained installed 2.8, which could
not use the available PyTorch-2.2 `pyg-lib` wheel for `KNNGraph`. This is a
**VERIFIED-LOCAL** compatibility observation, not a replacement for the PACE
environment export.

Before requesting GPUs, verify imports on a login or interactive development allocation according to PACE policy:

```bash
python -c 'import torch, graphnet, hydra, pytorch_lightning'
python -c 'from graphnet.models.graphs.nodes import IceMixNodes; from graphnet.data.datamodule import GraphNeTDataModulecustom'
DATA_ROOT=/path/to/prepared/data \
    python ice_mix/verify_config.py --cfg job --resolve
```

The commands are **VERIFIED-STATIC** reconstructions and **UNVERIFIED-CLUSTER** until the successor runs them.

## Prepare the supported launcher

Use `run_training.sbatch` for a new base run or explicit base-run resume. Before
submission validate or replace:

- allocation/account (`-A`);
- the submit directory (submit from the checkout root);
- the Slurm output destination if the current relative file is unsuitable;
- `ICE_MIX_WANDB_ROOT` when repository-local W&B runtime files are unsuitable;
- any explicit resume checkpoint and W&B run ID;
- obsolete node exclusions after checking current PACE health guidance.

Set `DATA_ROOT` to the project-storage directory containing all three expected
SQLite basenames. The launcher reads `ice_mix/.env` when present, but the file
must stay untracked and must never be pasted into documentation.

```bash
export DATA_ROOT=/successor/project/path/to/prepared/sqlite
ICE_MIX_DATA_CONFIG=standard \
ICE_MIX_PROJECT_NAME=IceMix-Standard \
    sbatch ice_mix/run_training.sbatch
```

The literal command is a template and **UNVERIFIED-CLUSTER** until submitted by
the new user. The older variant-specific launchers are preserved for provenance
but do not have the supported launcher's complete validation.

## Submit and monitor

Common Slurm commands are:

```bash
job_id=$(ICE_MIX_DATA_CONFIG=standard \
    ICE_MIX_PROJECT_NAME=IceMix-Standard \
    sbatch --parsable ice_mix/run_training.sbatch)
squeue -j "$job_id"
scontrol show job "$job_id"
sacct -j "$job_id" --format=JobID,State,Elapsed,ExitCode,AllocTRES,MaxRSS
tail -f "IceMixTraining-${job_id}.out"
```

These are standard Slurm patterns but **UNVERIFIED-CLUSTER** against current Phoenix policy. Prefer PACE's current examples if flags differ.

## Read a healthy log

Statically expected landmarks are:

1. all three databases pass source and staged-copy checks;
2. at least one healthy GPU and the chosen visible-device list;
3. fully resolved Hydra configuration;
4. dataset split generation;
5. Lightning/DDP initialization;
6. finite training and validation loss;
7. physics metrics and CSV/W&B logger activity;
8. new `best-*` and `last.ckpt` files.

The author reports that regular eight-L40S training progresses at roughly one epoch per hour. Monitor training and validation curves in W&B, treat any NaN loss as a failed state, and confirm that checkpoints continue to appear. The author expects the earlier NaN failure source to have been fixed by the 3D-vMF change. A staged real run report confirms eight-process launch, deterministic splitting, best-weight restoration, and exit code 0; its metrics CSV contains 31 train/validation epoch summaries with finite logged values across all populated metric fields (**VERIFIED-LOCAL**, 2026-08-13). Runtime remains **UNVERIFIED-CLUSTER** against the successor's future setup.

Do not assess improvement from the combined-event curve alone. Tracks reconstruct better than cascades, so changing their mixture can worsen the aggregate curve even when both topology-specific curves improve—an instance of Simpson's paradox. Always inspect the comparison plots stratified by topology. **Author-confirmed 2026-08-13.**

## Failure guide

| Symptom | First checks |
|---|---|
| Pending in queue | `squeue` reason, requested L40S count, QoS/allocation availability, walltime. Do not assume the code failed. |
| Exit 42 | GPU health probe found no usable device; inspect node/GPU diagnostics and contact PACE if reproducible. |
| Exit 43 | `DATA_ROOT` was not available to the script. |
| Exit 44/45 | Fine-tune checkpoint or required W&B resume metadata was absent. |
| Exit 46–49 | Slurm `TMPDIR`, a required source database, a copy process, or a staged destination failed validation. |
| CUDA OOM | Confirm this is an L40S job; lower per-rank batch size, pulse cap, workers/prefetch, or model dimensions for nonproduction hardware. |
| DDP hang | Inspect every rank's last output, GPU health, NCCL messages, sampler callback, and whether all ranks entered the same collective. |
| Non-finite loss | With `fail_on_non_finite=true`, identify the batch/rank and inspect inputs, precision, learning rate, checkpoint compatibility, and kappa. |
| Crash just after initialization on the known bad L40S node | The author reports that Phoenix may ignore `--exclude`. Use the current group-maintained allow-list of acceptable nodes and ask Jiyuan to confirm it; do not invent or preserve a stale list in documentation. |
| Exit 50–55 | W&B runtime directory, data variant, Hydra preflight, explicit checkpoint, W&B ID, or source run config failed validation. |
| Walltime kill | Confirm `last.ckpt` is durable, then resubmit `run_training.sbatch` with explicit `ICE_MIX_CKPT_PATH` and `ICE_MIX_WANDB_RUN_ID`. Do not resume from `$TMPDIR` or guess the newest directory. |
| W&B failure | Training also has CSV logs. Check network/auth separately; never paste tokens into scripts or documentation. |

Code-derived failure causes are **VERIFIED-STATIC**. A staged failed report confirms that `sqlite3.OperationalError: unable to open database file` is the root cause behind one exit-code-1 job, while nearby NVML warnings were secondary (**VERIFIED-LOCAL**, 2026-08-13). The bad-node workaround and runtime guidance are author-confirmed; their current effectiveness remains **UNVERIFIED-CLUSTER**.
