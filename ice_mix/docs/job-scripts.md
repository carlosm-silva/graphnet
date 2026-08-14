# Slurm and shell script catalog

This is the adjacent line-by-line explanation for every `.sbatch` and shell submission helper in `ice_mix/`. Repeated launchers share large literal blocks, so this page explains each repeated line/construct once and then records every file's resource and command differences. **VERIFIED-STATIC.**

## Identity and path validation before submission

The following checked-in values are intentionally retained as a map of the
departing author's working PACE setup. The successor is expected to receive
equivalent permissions, but must validate each value before submission:

- allocations `gts-pli77-ideas_l40s` and `gts-itaboada3` (retain only when the successor is authorized; prioritize the L40S allocation for this model);
- email `cfilho3@gatech.edu`;
- project checkout/output paths below `/storage/project/r-itaboada3-0/cfilho3`;
- scratch/W&B paths below `/storage/scratch1/8/cfilho3`;
- source paths below another user's `/storage/home/.../jliao74`;
- pinned/excluded node names after confirming current PACE status;
- hard-coded checkpoint paths in recent pilots.

User-specific email, home/output ownership, and W&B paths **must be changed by
the new user** even when the shared project/allocation remains accessible.
The existing identifiers help locate inherited data and runs; do not erase
them from the historical handoff. The author confirms that the successor will
have read/write access to the complete `cfilho3` project tree, including
`/storage/project/r-itaboada3-0/cfilho3/graphnet/ice_mix/.env`. That file was not
readable from the documentation workstation and is intentionally absent from
Git. Inspect it directly on PACE; do not paste access tokens into a launcher,
Markdown, Slack, or logs.

## Every Slurm directive used here

| Directive | Plain-language effect | Important correction |
|---|---|---|
| `#!/bin/bash` | Execute the file with Bash on the allocated node. | Helpers use Bash arrays/process substitution and are not POSIX shell scripts. |
| `#SBATCH -J...` | Human-readable job name shown by Slurm. | Does not determine the Hydra run name. |
| `#SBATCH -A...` | Charge the named PACE allocation/account. | Must be replaced with the successor's authorized allocation. |
| `#SBATCH -N1` | Allocate one compute node. | All current jobs are single-node, even when multi-GPU. |
| `#SBATCH --cpus-per-task=N` | Give the launch task `N` CPU cores. | Usually 32 for eight GPUs; inference scripts omit it and accept site default. |
| `#SBATCH --gres=gpu:l40s:N` | Request `N` L40S GPUs as generic resources. | Production uses 8; smoke uses 2; inference uses 1. `run_predict_temp` instead requests one A100 and is historical. |
| `#SBATCH -qinferno` | Request QoS `inferno`. | This is a QoS, not a `--partition`; no script explicitly selects a partition. Confirm current mapping on Phoenix. |
| `#SBATCH --mem=0` | Request all memory Slurm makes available on the node. | Existing comments saying “memory per GPU” are misleading. |
| `#SBATCH --mem=128G` / `200G` | Request that much node memory. | Not per-GPU memory. |
| `#SBATCH --time=HH:MM:SS` | Set hard walltime; Slurm terminates the job afterward. | Some 72-hour launchers incorrectly comment “8 hours.” |
| `#SBATCH -o PATH` | Write combined standard output/error to a job-ID-expanded path (`%j`). | Parent directory must exist before Slurm opens it; a `mkdir` inside the job cannot repair a missing parent for this directive. |
| `#SBATCH --mail-type=BEGIN,END,FAIL` | Email at start, normal end, and failure. | Requires a valid successor email. |
| `#SBATCH --mail-user=...` | Destination for Slurm mail. | Must be changed. |
| `#SBATCH --exclude=NODE` | Ask Slurm not to allocate a named node. | The author reports Phoenix may ignore this flag. The operational workaround is an allow-list of all other L40S nodes; obtain the current list from Jiyuan rather than copying a stale list. |
| `#SBATCH --tmp=3000G` | Request 3 TB job-local temporary storage and normally expose it through `$TMPDIR`. | Contents are ephemeral. Falling back to `/tmp` does not provide the requested capacity. |
| `#SBATCH --nodelist=NODE` | Force a node-probe job onto exactly one node. | Historical diagnostic only, not a production practice. |

## Repeated executable lines

| Line/construct | What it does and failure mode |
|---|---|
| `cd /storage/project/.../graphnet` | Enters the author's checkout. Replace it; add `|| exit` so a failed `cd` cannot run elsewhere. |
| `mkdir -p ice_mix/logs/sbatch_reports` | Creates a log directory after job start. It cannot make the `#SBATCH -o` parent early enough for Slurm. |
| `.env` load via `set -a; source ...` | Exports variables such as `DATA_ROOT` from the inherited `ice_mix/.env`. The successor is expected to inherit access to the author's PACE copy; keep it untracked. The `grep | xargs` variant in older launchers mishandles spaces. |
| `$TMPDIR` handling | Several inherited launchers and `resume_common.sh` fall back to shared `/tmp`. They do not establish that a Slurm-local directory is writable or large enough. |
| `cp ... "$TMPDIR" &; wait` | Base/evaluation launchers copy nu_mu and nu_e in the background and do not check individual exit statuses or destinations. The shared helper loops over all three names but warns and continues when a source is absent, then performs an undifferentiated `wait`. |
| `export LOCAL_DATA_DIR="$TMPDIR"` | Makes Python prefer staged files with matching basenames. Inherited scripts export it even when one or more copies are absent. |
| `module load anaconda3/2022.05.0.1` | Loads the author-confirmed current Phoenix Anaconda module. Availability remains cluster-dependent. |
| `conda activate graphnet` | Activates the project environment reproducibly described in [`docs/graphnet_env/`](graphnet_env/README.md). |
| embedded Python GPU probe | Checks ECC counters, executes a small matrix multiplication on each visible GPU, and prints comma-separated healthy indices. Exceptions cause a GPU to be omitted. |
| `CUDA_VISIBLE_DEVICES="$GOOD"` | Restricts child processes to healthy GPUs. |
| `NPROC=...` | Counts comma-separated healthy devices and determines `torchrun` worker count. Several older scripts still hard-code 8 and can mismatch after filtering. |
| `WANDB_CACHE_DIR`, `WANDB_DIR`, `WANDB_DATA_DIR` | Point W&B cache, run, and data artifacts at three literal directories below `/storage/scratch1/8/cfilho3`. No `ICE_MIX_WANDB_ROOT` override is implemented. Review these paths in any successor launcher. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Requests a CUDA allocator mode intended to reduce fragmentation. |
| `CUDA_LAUNCH_BLOCKING=0` | Leaves CUDA execution asynchronous; this is the normal performant mode, not a debugging synchronization. |
| `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` | Limits CPU math threads per rank; eight ranks imply up to about 32 threads. |
| `TORCH_CUDNN_V8_API_ENABLED=1` | Requests PyTorch's cuDNN v8 execution-plan API. Whether this setting is still useful on Phoenix is **UNVERIFIED-CLUSTER**. |
| `NCCL_P2P_DISABLE=0` | Leaves direct GPU-to-GPU peer transport enabled when NCCL and the node topology support it. |
| `NCCL_IB_DISABLE=0` | Leaves NCCL InfiniBand transport enabled when available. |
| `CUDA_DEVICE_MAX_CONNECTIONS=32` | Requests up to 32 work queues/connections per CUDA device. Its performance effect on current L40S nodes is **UNVERIFIED-CLUSTER**. |
| `SLURM_MPI_TYPE=pmix_v4` | Appears as an unexported shell assignment in 15 newer training/fine-tuning files and is apparently intended to select Slurm PMIx v4 support. Whether it reaches `srun`, is required, or has any effect on current Phoenix jobs is **UNVERIFIED-CLUSTER**. |
| `srun --cpu-bind=cores` | Starts the job step and binds CPU execution to allocated cores. |
| `torchrun --nproc_per_node=N` | Starts `N` local DDP processes. Current jobs are single-node and do not supply multi-node rendezvous arguments. |
| `exit $?` / captured exit code | Returns the training/evaluation status to Slurm so failures are visible. |

## Base production and resume launchers

No base launcher is currently documented as submit-ready. `run_training.sbatch`
requests one node, 8 L40S, 32 CPUs, `inferno`, all node memory, and 72 hours, but
its staging and resume coverage are incomplete. The variant-specific files are
the author's historical day-to-day workflow. Their resource scale is
cluster-only and their operational defects require researcher review.

| File | Python/Hydra action | Status and sharp edges |
|---|---|---|
| `run_training.sbatch` | Default `train.py` | Not submit-ready: hard-coded author checkout/account/email/cache paths; copies only nu_mu and nu_e; does not validate copy results; falls back to `/tmp`; dynamically sizes `torchrun` after a GPU probe. It does not consume `ICE_MIX_DATA_CONFIG`, `ICE_MIX_PROJECT_NAME`, `ICE_MIX_CKPT_PATH`, or `ICE_MIX_WANDB_RUN_ID`. |
| `run_standard.sbatch` | `train.py project_name=IceMix-Standard` | Historical variant launcher; older staging parses literal `.db` YAML lines and may fail after `${oc.env:DATA_ROOT}` interpolation because it does not resolve Hydra. Hard-codes 8 workers. |
| `run_augmented.sbatch` | `train.py data=augmented_rotation` | Current on-the-fly rotation variant despite the name “Augmented.” Same older staging issue and hard-coded 8 workers. |
| `run_drop.sbatch` | `train.py data=drop` | Token-drop base variant; same older staging issue and hard-coded 8 workers. |
| `run_drop_aug_rot.sbatch` | `train.py data=drop_aug_rot` | Combined on-the-fly rotation/token-drop variant; same older staging issue and hard-coded 8 workers. |
| `run_sanity_check.sbatch` | Default `train.py`, one epoch | Still requests all 8 L40S/32 CPUs and baseline batch limits are not reduced; a cluster sanity run, not a cheap laptop smoke test. |
| `run_training_resume.sbatch` | Default `train.py` plus automatically discovered `ckpt_path`/`run_name` | Historical resume path; guesses the newest matching run and does not validate the saved resolved config against the new composition. |
| `run_augmented_resume.sbatch` | Rotation config plus latest matching checkpoint | Uses shared checkpoint discovery but older duplicated runtime block; hard-codes 8 workers. |
| `run_drop_resume.sbatch` | Drop config plus latest matching checkpoint | Same pattern. |
| `run_drop_aug_rot_resume.sbatch` | Combined config plus latest matching checkpoint | Same pattern. |

“Augmented” in the current non-OLD launchers means on-the-fly rotation. Stored fixed-rotation data are deprecated. **Author-confirmed 2026-08-13.**

## Experimental fine-tuning launchers

These request one node, 8 L40S, 32 CPUs, all memory, and 72 hours unless marked smoke. They source `resume_common.sh`, stage three files when present, activate the environment, configure healthy GPUs, then run `train.py`.

| File | Data / mode | Resume behavior |
|---|---|---|
| `run_training_fine_tune.sbatch` | Standard, LBFGS, last block + head, batch 32, FP32 | Loads weights from latest `IceMix` base checkpoint into a new run. |
| `run_training_fine_tune_resume.sbatch` | Same | Resumes latest `IceMix-FineTune-LBFGS` Lightning/W&B state. |
| `run_augmented_fine_tune.sbatch` | On-the-fly rotation, same LBFGS settings | Loads latest rotation base weights. |
| `run_augmented_fine_tune_resume.sbatch` | Same | Resumes latest rotation LBFGS run. |
| `run_drop_fine_tune.sbatch` | Token drop, same LBFGS settings | Loads latest drop base weights. |
| `run_drop_aug_rot_fine_tune.sbatch` | Combined, same LBFGS settings | Loads latest combined base weights. No matching dedicated resume file exists. |
| `run_fine_tune_smoke.sbatch` | Standard LBFGS, 2 L40S/8 CPUs/30 min, 2 train + 2 val batches | Wiring smoke, still cluster-only and checkpoint-dependent. |
| `run_drop_fine_tune_smoke.sbatch` | Drop LBFGS, 2 L40S/8 CPUs/4 h, 5,000 train + 2 val batches | Production-like stress pilot, not a quick smoke despite its name. |

The fine-tuning family is experimental with some success, not the recommended starting path. **Author-confirmed 2026-08-13.**

## AdamW+EMA rotation pilot

| File | Action |
|---|---|
| `run_rotation_adamw_ema_lr2e5.sbatch` | Load a hard-coded rotation checkpoint; last-block AdamW+EMA at `2e-5`, cosine scheduler, 8 epochs. |
| `run_rotation_adamw_ema_lr2e6.sbatch` | Same at `2e-6`. |
| `run_rotation_adamw_ema_lr6p25e6.sbatch` | Same at `6.25e-6`. |
| `run_rotation_adamw_ema_smoke.sbatch` | Same family at `2e-5`, 2 L40S/8 CPUs/1 hour, 2 train + 2 validation batches. |
| `run_rotation_adamw_ema_lr2e5_resume.sbatch` | Resource header then invokes shared resume helper with project name and learning rate. |
| `run_rotation_adamw_ema_lr2e6_resume.sbatch` | Same for `2e-6`. |
| `run_rotation_adamw_ema_lr6p25e6_resume.sbatch` | Same for `6.25e-6`. |
| `run_rotation_adamw_ema_resume_common.sh` | Changes to author checkout, loads shared functions/environment/data, finds latest exact-pilot checkpoint/W&B ID, then resumes `train.py`. |

The three non-resume pilots embed `SOURCE_CKPT`; replace it and confirm architecture compatibility. The resume wrappers invoke the shared helper by an absolute author path, which **must be changed by the new user**.

## Prediction and robustness jobs

These request one node, one L40S, 128 GB node memory, and 3 TB `$TMPDIR`, with no explicit CPU count.

| File | Walltime / action | Status and sharp edges |
|---|---|---|
| `run_predict.sbatch` | 14 h; `predict.py --n-gpus 1` | Not submit-ready: hard-coded author/Jiyuan paths and output tree, stages only nu_mu/nu_e without copy validation, falls back to `/tmp`, and exposes no force/test-split environment controls. |
| `run_predict_force.sbatch` | 14 h; adds `--force` | Regenerates existing prediction CSVs; use deliberately because it overwrites derived outputs. Same stale staging. |
| `run_predict_temp.sbatch` | 4 h; one A100, 200 GB, different allocation | Author-confirmed abandoned/historical temporary variant. |
| `run_resilience_test.sbatch` | 2 h; forced token-drop study at 1% of validation | Not submit-ready: same hard-coded/two-database staging; no explicit seed exists in the Python CLI; output tree is fixed. |
| `run_checkerboard_test.sbatch` | 12 h; complementary-half evaluation for `MODEL_CONFIG` | Not submit-ready: same staging hazards; only `MODEL_CONFIG` is configurable; no run, seed, or fraction environment controls are implemented. Despite its name, the perturbation is not spatial. |
| `run_checkerboard_test_aug_rot.sbatch` | 12 h; historical checkerboard wrapper for rotation model | On-the-fly rotation model checkpoint, not stored rotation data. It inherits the same evaluation limitations. |
| `run_rotation_checkpoint_averaging.sbatch` | 8 h; one L40S/8 CPUs; interpolate two checkpoints at five lambdas | Experimental. Sources shared staging/environment, uses explicit checkpoint/output variables, and runs single-process evaluation. |

## Shared `resume_common.sh`, line by line by function

- `find_wandb_run_id RUN_DIR`: searches newest W&B metadata JSON, parses `id`/`run_id`/`runId`; if absent, extracts the suffix of a `run-*` directory.
- `configure_resume_from_latest_run PROJECT LABEL`: finds newest `outputs/${PROJECT}_*/checkpoints/last.ckpt`, adds `ckpt_path` and stable `run_name`; sets `WANDB_RUN_ID` and `WANDB_RESUME=allow`. If none exists, it starts from scratch.
- `configure_fine_tune_from_latest_run SOURCE LABEL`: requires newest source `last.ckpt`, adds `fine_tune_from_ckpt`, and returns 44 when absent.
- `configure_fine_tune_resume_from_latest_run PROJECT LABEL`: requires the fine-tune checkpoint and a W&B ID; returns 44/45 when missing and uses `WANDB_RESUME=must`.
- `load_ice_mix_env`: exports non-comment `.env` assignments. Keep the file untracked.
- `copy_standard_data_to_local_tmp`: falls back to `/tmp`; loops over the three standard basenames; warns and continues for missing sources; starts available copies concurrently; waits without mapping statuses to files; then exports `LOCAL_DATA_DIR` without validating destinations.
- `configure_healthy_gpus`: performs the ECC/matrix probe, exports visible devices and computed `NPROC`, or returns 42.
- `configure_training_environment`: exports the three literal author-owned W&B scratch paths; it neither creates them nor reads `ICE_MIX_WANDB_ROOT`. It then sets allocator, thread, cuDNN, NCCL, and CUDA variables.

The helper defines functions only; callers choose their order. Its automatic
“latest run” functions and permissive staging are historical workflow behavior,
not a validated successor contract.

## Deprecated and duplicate launchers

| File | Evidence/status |
|---|---|
| `run_augmented_OLD.sbatch` | Explicit OLD; points through deprecated fixed-rotation config history. |
| `run_drop_aug_rot_OLD.sbatch` | Explicit OLD; deprecated stored-data era. |
| `run_augmented_indexed.sbatch` | Byte-identical to `run_augmented_OLD.sbatch` in this checkout; author-confirmed abandoned/historical. |

They remain for provenance and must not be presented as current rotation setup. Their abandoned/historical status is author-confirmed 2026-08-13.

## Historical node probes

The seven files `test_atl1-1-01-010-{29,31,35}-0.sbatch`, `test_atl1-1-03-004-{29,31}-0.sbatch`, and `test_atl1-1-03-007-{29,31}-0.sbatch` each:

1. request job name `NodeTest` on the author's L40S allocation;
2. request one named node and four L40S GPUs;
3. request `inferno`, `--mem=100`, and one minute;
4. set `-o` to a directory rather than a filename, which may be invalid or ambiguous;
5. execute `python hello.py`, although `hello.py` lives under `ice_mix/` when launched from the repository root;
6. do not load the conda environment or inspect GPUs.

These are author-confirmed abandoned access probes, not meaningful model or node-health tests. Do not pin production jobs to them.

## Complete inventory check

This page covers all 42 `.sbatch` files and both shell helpers present on 2026-08-13. Use `find ice_mix -maxdepth 1 \( -name '*.sbatch' -o -name '*.sh' \) -print | sort` and compare after future additions; any new script requires an entry here.
