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
them from the historical handoff. `.env` values were not readable during this
documentation pass. Recover that file from the author's PACE project area or
ask him on Slack; do not paste access tokens into a launcher or Markdown.

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
| `.env` load via `set -a; source ...` | Exports variables such as `DATA_ROOT`. Recover the successor's file from the inherited PACE area or create it there; never put tokens in Markdown. The `grep | xargs` variant in older launchers mishandles spaces. |
| `$TMPDIR` validation | The supported launcher fails unless Slurm provides a writable job-local directory. Older launchers fall back to `/tmp`, which is unsafe for multi-terabyte staging. |
| `cp ... "$TMPDIR" &; wait` | The supported helper first verifies all three source databases, copies them concurrently, checks every exit status, then verifies every staged file is readable and nonempty. |
| `export LOCAL_DATA_DIR="$TMPDIR"` | Makes Python replace configured paths with staged files having the same basename. In the supported path this is exported only after all three copies pass validation. |
| `module load anaconda3/2022.05.0.1` | Loads the author-confirmed current Phoenix Anaconda module. Availability remains cluster-dependent. |
| `conda activate graphnet` | Activates the project environment reproducibly described in [`docs/graphnet_env/`](graphnet_env/README.md). |
| embedded Python GPU probe | Checks ECC counters, executes a small matrix multiplication on each visible GPU, and prints comma-separated healthy indices. Exceptions cause a GPU to be omitted. |
| `CUDA_VISIBLE_DEVICES="$GOOD"` | Restricts child processes to healthy GPUs. |
| `NPROC=...` | Counts comma-separated healthy devices and determines `torchrun` worker count. Several older scripts still hard-code 8 and can mismatch after filtering. |
| W&B cache exports | The supported helper writes below `ice_mix/outputs/wandb_runtime` by default or below explicit `ICE_MIX_WANDB_ROOT`; older scripts contain author scratch paths that must be changed. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Requests a CUDA allocator mode intended to reduce fragmentation. |
| `CUDA_LAUNCH_BLOCKING=0` | Leaves CUDA execution asynchronous; this is the normal performant mode, not a debugging synchronization. |
| `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` | Limits CPU math threads per rank; eight ranks imply up to about 32 threads. |
| cuDNN/NCCL/CUDA exports | Enable cuDNN v8, P2P, InfiniBand, and more CUDA device connections. Effectiveness on Phoenix is **UNVERIFIED-CLUSTER**. |
| `srun --cpu-bind=cores` | Starts the job step and binds CPU execution to allocated cores. |
| `torchrun --nproc_per_node=N` | Starts `N` local DDP processes. Current jobs are single-node and do not supply multi-node rendezvous arguments. |
| `exit $?` / captured exit code | Returns the training/evaluation status to Slurm so failures are visible. |

## Base production and resume launchers

`run_training.sbatch` is the one supported base launcher. It requests one node,
8 L40S, 32 CPUs, `inferno`, all node memory, and 72 hours. The resource scale is
cluster-scale: the production batch size and loader settings are not expected to
fit an 8 GB laptop GPU. The variant-specific launchers remain as historical
workflow evidence until they are migrated to the same fail-fast path.

| File | Python/Hydra action | Status and sharp edges |
|---|---|---|
| `run_training.sbatch` | `train.py` with `ICE_MIX_DATA_CONFIG` and `ICE_MIX_PROJECT_NAME` | **Supported.** Validates the checkout root, environment, GraphNeT boundary, all three database sources/copies, Hydra config, GPU health, and explicit resume identity; dynamically sizes `torchrun`. |
| `run_standard.sbatch` | `train.py project_name=IceMix-Standard` | Historical variant launcher; older staging parses literal `.db` YAML lines and may fail after `${oc.env:DATA_ROOT}` interpolation because it does not resolve Hydra. Hard-codes 8 workers. |
| `run_augmented.sbatch` | `train.py data=augmented_rotation` | Current on-the-fly rotation variant despite the name “Augmented.” Same older staging issue and hard-coded 8 workers. |
| `run_drop.sbatch` | `train.py data=drop` | Token-drop base variant; same older staging issue and hard-coded 8 workers. |
| `run_drop_aug_rot.sbatch` | `train.py data=drop_aug_rot` | Combined on-the-fly rotation/token-drop variant; same older staging issue and hard-coded 8 workers. |
| `run_sanity_check.sbatch` | Default `train.py`, one epoch | Still requests all 8 L40S/32 CPUs and baseline batch limits are not reduced; a cluster sanity run, not a cheap laptop smoke test. |
| `run_training_resume.sbatch` | Default `train.py` plus `ckpt_path`, stable `run_name` | Historical automatic-resume path; duplicates staging/GPU setup and guesses the newest matching run. Prefer explicit `ICE_MIX_CKPT_PATH` and `ICE_MIX_WANDB_RUN_ID` with `run_training.sbatch`. |
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
| `run_predict.sbatch` | 14 h; `predict.py --n-gpus 1` | **Supported.** Uses checkout-root validation, the shared fail-fast three-flavor staging/GPU/environment preflight, successor-selectable output tree, and optional force/test-split flags. |
| `run_predict_force.sbatch` | 14 h; adds `--force` | Regenerates existing prediction CSVs; use deliberately because it overwrites derived outputs. Same stale staging. |
| `run_predict_temp.sbatch` | 4 h; one A100, 200 GB, different allocation | Author-confirmed abandoned/historical temporary variant. |
| `run_resilience_test.sbatch` | 2 h; forced token-drop study at 1% of validation | **Supported research utility.** Shared fail-fast staging/preflight with explicit seed and configurable fraction/output tree. |
| `run_checkerboard_test.sbatch` | 12 h; complementary-half evaluation for configured project | **Supported research utility.** Despite its filename it is not spatial; project, seed, fraction, and output tree are submit-time variables. |
| `run_checkerboard_test_aug_rot.sbatch` | 12 h; historical checkerboard wrapper for rotation model | On-the-fly rotation model checkpoint, not stored rotation data. Prefer the supported generic script with `ICE_MIX_MODEL_CONFIG`. |
| `run_rotation_checkpoint_averaging.sbatch` | 8 h; one L40S/8 CPUs; interpolate two checkpoints at five lambdas | Experimental. Sources shared staging/environment, uses explicit checkpoint/output variables, and runs single-process evaluation. |

## Shared `resume_common.sh`, line by line by function

- `find_wandb_run_id RUN_DIR`: searches newest W&B metadata JSON, parses `id`/`run_id`/`runId`; if absent, extracts the suffix of a `run-*` directory.
- `configure_resume_from_latest_run PROJECT LABEL`: finds newest `outputs/${PROJECT}_*/checkpoints/last.ckpt`, adds `ckpt_path` and stable `run_name`; sets `WANDB_RUN_ID` and `WANDB_RESUME=allow`. If none exists, it starts from scratch.
- `configure_fine_tune_from_latest_run SOURCE LABEL`: requires newest source `last.ckpt`, adds `fine_tune_from_ckpt`, and returns 44 when absent.
- `configure_fine_tune_resume_from_latest_run PROJECT LABEL`: requires the fine-tune checkpoint and a W&B ID; returns 44/45 when missing and uses `WANDB_RESUME=must`.
- `load_ice_mix_env`: exports non-comment `.env` assignments. Keep the file untracked.
- `copy_standard_data_to_local_tmp`: requires writable Slurm `TMPDIR` and `DATA_ROOT`; preflights all three exact sources, copies concurrently, validates every exit status and destination, then sets `LOCAL_DATA_DIR`.
- `configure_healthy_gpus`: performs the ECC/matrix probe, exports visible devices and computed `NPROC`, or returns 42.
- `configure_training_environment`: creates W&B cache/run/data directories below `ICE_MIX_WANDB_ROOT` or the repository output tree, then sets allocator/thread and L40S/NCCL settings.

The helper defines functions only; callers choose their order. The supported
launcher treats staging failure as fatal. The older automatic “latest run”
functions remain for historical launchers; they are not used by the supported
explicit-resume path.

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
