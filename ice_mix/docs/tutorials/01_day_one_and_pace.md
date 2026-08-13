# Tutorial 1 — Day one and PACE setup

**Goal:** understand the repository, gain Phoenix access, and verify the inherited environment without spending a production allocation.

**Prerequisites:** Georgia Tech identity, group/allocation sponsorship, repository access, and the original author's Slack contact.

**Expected duration:** 1–3 hours after PACE access is approved; account approval itself may take longer.

**Run on:** browser and PACE login node. Do not run training on the login node.

**Verification:** repository paths/imports are **VERIFIED-STATIC**. Account creation, hostnames, module availability, and commands actually succeeding are **UNVERIFIED-CLUSTER**.

## 1. Establish the human handoff

The original author is available on Slack for questions about scientific intent,
historical runs, and the inherited PACE file locations. The code and this
handoff were produced with substantial Codex and ChatGPT assistance under his
scientific direction; do not assume every implementation detail was written or
memorized manually. Send a specific file, command, or physics question when you
need context.

Confirm with him:

- the authoritative project directory containing the three prepared databases;
- the preferred base launcher and a known-good run directory;
- the group allocation/account the successor should use, if still applicable.

## 2. Obtain and use PACE access

Follow the current [PACE documentation](https://docs.pace.gatech.edu/) for account onboarding, VPN/MFA, and the current Phoenix login hostname. Do not copy a hostname from this repository: none is encoded as a stable project interface.

Once logged in:

```bash
hostname
pwd
sinfo
squeue -u "$USER"
```

These read-only commands establish that you are on PACE and can see Slurm. Queue/partition output varies; no sample output is prescribed.

## 3. Locate project storage

Use the successor's group-owned project path, not any `/storage/.../cfilho3` path copied from a launcher.

```bash
cd /successor/project/path/to/graphnet
git status --short
ls ice_mix
```

The expected package landmarks are `train.py`, `conf/`, `src/`, and the `run_*.sbatch` files. If the working tree has local changes, preserve and understand them before editing.

## 4. Rebuild the exported environment

The complete exported evidence and portable successor recipe are checked in at
[`docs/graphnet_env/`](../graphnet_env/README.md). From the checkout root:

```bash
module load anaconda3/2022.05.0.1
git rev-parse HEAD
conda env create -f ice_mix/docs/graphnet_env/pace-environment.yml
conda activate graphnet
python -m pip install -r ice_mix/docs/graphnet_env/pace-pip-requirements.txt
python -m pip install --no-build-isolation --no-deps --editable .
```

The required source revision is
`4394131647b4a581e7d4923361b2814ab9e03ff5`; stop if the checkout differs and
locate the inherited revision. Then record, without exposing credentials:

```bash
which python
python --version
python -c 'import torch, graphnet, hydra, pytorch_lightning; print(torch.__version__)'
python -c 'from graphnet.models.graphs.nodes import IceMixNodes; from graphnet.data.datamodule import GraphNeTDataModulecustom'
```

If the module or an exact artifact is no longer available, compare against the
raw exports in the same directory and ask the author on Slack before changing
PyTorch, PyG, or GraphNeT. Do not install into a shared or base environment.

## 5. Resolve configuration only

Point `DATA_ROOT` at the prepared-data project directory for this shell and use Hydra's built-in flags to print the complete resolved configuration:

```bash
export DATA_ROOT=/successor/project/path/to/prepared/sqlite
python ice_mix/verify_config.py --cfg job --resolve
```

Inspect that all three database paths resolve. A departing-author project path may remain useful because the successor is expected to inherit equivalent permissions, but update user-specific output, email, and ownership locations. This command does not train a model. **VERIFIED-LOCAL:** the command composed and printed the staged rotation configuration on 2026-08-13.

## Common failures

- **Module not found:** the saved environment is incomplete or the wrong Python is active.
- **Missing `IceMixNodes` or `GraphNeTDataModulecustom`:** the public GraphNeT version does not match this checkout's custom boundary.
- **Hydra interpolation error for `DATA_ROOT`:** export it before invoking Python.
- **Permission denied in project storage:** request group access; do not copy data into home as a workaround.

## What to try next

Continue to [Tutorial 2](02_data_and_splits.md). Do not submit an eight-GPU job until the data layout and a reduced smoke run are understood.
