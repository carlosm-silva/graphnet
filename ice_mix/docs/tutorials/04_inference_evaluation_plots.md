# Tutorial 4 — Inference, evaluation, and plots

**Goal:** turn a trained checkpoint into a prediction table and scientifically interpretable comparison plots.

**Prerequisites:** a run directory with resolved Hydra config and checkpoint, three input databases, and the matching GraphNeT environment.

**Expected duration:** dry-run minutes; full inference hours are **UNVERIFIED-CLUSTER**.

**Run on:** dry-run and lightweight plots may run on a suitable workstation/login allocation; full inference uses the one-GPU Phoenix prediction job.

**Verification:** CLI behavior and output columns are **VERIFIED-STATIC**. Full runtime and numerical output are **UNVERIFIED-CLUSTER**.

## 1. Discover runs without inference

```bash
python ice_mix/predict.py \
    --base-dir /successor/project/path/to/graphnet/ice_mix/outputs \
    --n-gpus 1 \
    --dry-run
```

The script searches old date/time Hydra directories and newer project-named directories. It skips runs without recoverable config/checkpoints and, unless `--force` is used, runs with an existing `predictions/results.csv`.

## 2. Inspect checkpoint selection

`predict.py` prefers filenames matching `best-epoch=*-val_loss=*.ckpt`, choosing the smallest loss parsed from the filename; `last.ckpt` is only a fallback. For EMA checkpoints it extracts the EMA copy. **VERIFIED-STATIC.**

Confirm the printed run and checkpoint manually before removing `--dry-run`.

## 3. Submit inference

Set `DATA_ROOT` in the shell or the untracked `ice_mix/.env`. The supported
launcher stages and validates all three databases, then:

```bash
sbatch ice_mix/run_predict.sbatch
```

Use `--use-test-split` only for the held-out test partition. Default inference
uses validation selections. Avoid repeated test-set inspection while making
model choices. Every completed evaluation writes
`predictions/evaluation_manifest.json`, recording the partition, checkpoint,
split definition, and ordered database/event selections. Preserve it with
`results.csv`. **VERIFIED-STATIC.**

Set `ICE_MIX_OUTPUTS_DIR` when the run tree is not the checkout default, and set
`ICE_MIX_USE_TEST_SPLIT=1` only for deliberate held-out evaluation. The
launcher is shell-checked but remains **UNVERIFIED-CLUSTER**.

Both `DATA_ROOT` and `LOCAL_DATA_DIR` matter during staged inference:
OmegaConf resolves the original `DATA_ROOT` paths before `predict.py` replaces
matching basenames with `LOCAL_DATA_DIR`. Setting only `LOCAL_DATA_DIR` fails
configuration interpolation. **VERIFIED-LOCAL** from the staged CPU inference
run on 2026-08-13.

## 4. Inspect the table

Expected prediction columns are:

```text
pos_x_pred pos_y_pred pos_z_pred
dir_x_pred dir_y_pred dir_z_pred dir_kappa_pred
```

Additional copied attributes include truth angles/position, `event_no`, energy, PID, interaction type, oneweight, and pulse count. **VERIFIED-STATIC.**

A staged 50-row prediction table contains all of these inputs, finite energy,
pulse-count and kappa values, and predicted direction norms within about
`1e-7` of unity. The existing plotting utilities produced finite angular and
vertex errors for all rows in an existing local pandas/matplotlib environment
(**VERIFIED-LOCAL**, 2026-08-13). Its small cascade subset was insufficient for
the plotting code's per-bin minimum count, which correctly yielded empty
cascade summary bins.

Full `predict.py` reconstruction was also exercised locally on the staged
checkpoint with `--n-gpus 0`: it selected nine validation events across the
three databases and wrote finite `results.csv`, `state_dict.pth`, and
`model_config.yml` artifacts (**VERIFIED-LOCAL**, 2026-08-13). This required
`torch-geometric==2.5.3`; unconstrained PyG 2.8 was incompatible with the
PyTorch-2.2 extension set at `KNNGraph` construction.

Check shape/headers without printing sensitive events:

```bash
python - <<'PY'
import pandas as pd
path = "/successor/project/run/predictions/results.csv"
frame = pd.read_csv(path, nrows=5)
print(frame.columns.tolist())
print(frame.shape)
PY
```

## 5. Plot one run and comparisons

```bash
python ice_mix/plot_run.py \
    --results-csv /successor/project/run/predictions/results.csv \
    --run-dir /successor/project/run \
    --output-dir /successor/project/run/plots \
    --baseline-csv /successor/project/explicit_baseline/predictions/results.csv

python ice_mix/generate_plots.py \
    --base-dir /successor/project/path/to/graphnet/ice_mix/outputs \
    --reference-csv /path/found/by/searching/inherited/PACE/files/JointLargeTC0.04results_LRNEW.csv
```

The historical reference CSV is not assigned a canonical personal path. Search
the author's inherited PACE files for `JointLargeTC0.04results_LRNEW.csv`; omit
`--reference-csv` if it is unavailable. Plots derive angular separation, vertex
distance, binned resolutions, and comparisons from the CSV. Current statistics
are unweighted: `oneweight` is carried through predictions but ignored by the
plotting functions, so every row contributes equally. **VERIFIED-STATIC.**

Pairwise fine-tune and nu-tau comparisons require explicit result pairs by
default and reject mismatched evaluation manifests/event populations:

```bash
python ice_mix/generate_fine_tune_comparison_plots.py \
    --pair /path/to/base/predictions/results.csv \
           /path/to/fine_tuned/predictions/results.csv

python ice_mix/generate_nutau_comparison_plots.py \
    --pair IceMix-Standard \
           /path/to/without_nutau/predictions/results.csv \
           /path/to/with_nutau/predictions/results.csv
```

The opt-in legacy discovery flags exist only for historical output trees where
manifests were not generated; job IDs or modification times are not scientific
treatment labels.

`plot_run.py` likewise never guesses a baseline unless the explicitly named
legacy-discovery flag is supplied. Its explicit baseline must have the same
evaluation manifest and row identities.

Each aggregate/comparison output directory receives `plot_manifest.json` with
the exact input paths, optional reference path, and
`weighting.mode=unweighted_equal_rows`. Cross-project master plots also require
matching evaluation manifests when more than one model is shown. A weighted
mode is intentionally not offered until the data owner defines the target
population and authoritative `oneweight` convention. **VERIFIED-STATIC.**

Always compare tracks and cascades separately as well as together. Tracks are normally reconstructed better, so a changed track/cascade mixture can make the combined curve worse even when both topology-specific curves improve (Simpson's paradox). This interpretation warning is author-confirmed 2026-08-13.

Sparse selections are a plotting sharp edge. With the staged 50-row table,
the seven cascade events did not satisfy the per-bin minimum and
`plot_run.py` crashed when log-scaling the resulting empty statistics; all and
track PNGs were written first. This is **VERIFIED-LOCAL** (2026-08-13). Use a
larger evaluation sample and check bin counts before diagnosing Matplotlib.

## 6. Optional checkpoint interpolation

`evaluate_rotation_checkpoint_averaging.py` evaluates linear weight interpolation between two architecture-compatible checkpoints. This is an experimental study, not ordinary inference:

```bash
python ice_mix/evaluate_rotation_checkpoint_averaging.py \
    --checkpoint-a /path/to/a.ckpt \
    --checkpoint-b /path/to/b.ckpt \
    --lambdas 0 0.25 0.5 0.75 1 \
    --batch-size 128 \
    --num-workers 3 \
    --output-dir /successor/project/path/to/interpolation
```

Start with `--limit-val-batches 2`; full validation requires a CUDA GPU.

## Common failures

- **Config not found:** preserve each run's `.hydra/config.yaml`; a saved GraphNeT model config alone does not replace Hydra data settings.
- **State-dict mismatch:** GraphNeT version, graph definition, architecture, ordinary/EMA format, or fine-tune mode differs.
- **No prediction rows:** inspect split generation and loader setup; do not accept an empty CSV.
- **Plot KeyError:** compare the script's required columns with the historical output generation version.
- **Graph metadata mismatch:** prediction still builds `KNNGraph`, but IceMix does not consume its edges; this remains only for backward compatibility. **Author-confirmed 2026-08-13.**

## What to try next

Proceed to [Tutorial 5](05_robustness_studies.md) for controlled pulse-removal studies.
