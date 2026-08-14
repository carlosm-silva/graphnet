# Tutorial 4 — Inspecting inference, evaluation, and plots

**Goal:** inspect checkpoint selection and prediction outputs without mistaking
the inherited automation for a matched scientific evaluation.

**Prerequisites:** an inspected run directory with Hydra config/checkpoint, the
three input databases, the compatible environment, and Tutorial 3's researcher
approval boundary.

**Expected duration:** dry-run minutes; full inference time is
**UNVERIFIED-CLUSTER**.

**Run on:** dry-run and lightweight CSV inspection where dependencies are
available; full inference only in a researcher-reviewed Phoenix GPU job.

**Verification:** CLI definitions, output columns, and the hazards below are
**VERIFIED-STATIC**. Full-scale scientific equivalence is not verified.

## 1. Discover runs without inference

```bash
python ice_mix/predict.py \
    --base-dir /successor/project/path/to/graphnet/ice_mix/outputs \
    --n-gpus 1 \
    --dry-run
```

The script searches old date/time Hydra directories and flat project-named run
directories. It skips missing configs/checkpoints and existing results unless
`--force` is supplied.

## 2. Inspect checkpoint selection manually

`predict.py` parses unsigned decimal losses from filenames matching
`best-epoch=*-val_loss=*.ckpt`. Negative values and scientific notation are not
recognized even though the vMF contribution can make a loss negative. With
mixed candidates the wrong positive checkpoint can win; with only unparseable
best files the run can be skipped. Confirm the selected path against the
checkpoint directory and training logs before inference.

Ordinary prediction uses the EMA extraction helper when an EMA copy is present.
The resilience and complementary-half evaluators instead load raw checkpoint
state and may reject EMA checkpoints. **VERIFIED-STATIC.**

## 3. Do not treat the inherited prediction launcher as ready

`run_predict.sbatch` embeds personal paths, stages only nu_mu and nu_e, does not
validate background copies, and fixes its output tree. Read it without running:

```bash
sed -n '1,220p' ice_mix/run_predict.sbatch
```

A researcher must approve any corrected launch path. The Python CLI itself also
reconstructs 42/80-10-10 dynamic selections from obsolete flat config keys; it
does not honor nested CSV/non-default training selections and writes no
evaluation manifest. Therefore `--use-test-split` means the regenerated dynamic
test slice, not necessarily the run's held-out test population.

Upstream `IceMixNodes` also resamples capped events with `torch.randperm` on
each loader pass. Recording only `event_no` would not fully identify the model
input even if an external event manifest were created.

## 4. Inspect produced artifacts

Successful ordinary prediction writes:

```text
predictions/results.csv
predictions/state_dict.pth
predictions/model_config.yml
```

Prediction columns are:

```text
pos_x_pred pos_y_pred pos_z_pred
dir_x_pred dir_y_pred dir_z_pred dir_kappa_pred
```

Copied attributes include truth angles/position, `event_no`, energy, PID,
interaction type, `oneweight`, and pulse count. Check headers and finiteness
without printing event content:

```bash
python - <<'PY'
import pandas as pd

path = "/successor/project/run/predictions/results.csv"
frame = pd.read_csv(path)
print(frame.columns.tolist())
print("rows", len(frame))
print("finite numeric", frame.select_dtypes("number").notna().all().all())
PY
```

The staged checkpoint produced nine finite CPU predictions in the disposable
local environment (**VERIFIED-LOCAL**, 2026-08-13). That run demonstrates model
wiring only; it does not cure the split or random pulse-cap findings.

The evaluation CLIs catch per-run failures and can exit zero after tracebacks or
after producing nothing. Confirm file modification times, nonzero row counts,
the intended checkpoint, and the complete log even when Slurm says `COMPLETED`.

## 5. Understand plotting before running it

The actual per-run CLI is:

```bash
python ice_mix/plot_run.py \
    --results-csv /successor/project/run/predictions/results.csv \
    --run-dir /successor/project/run \
    --output-dir /successor/project/run/plots
```

It automatically discovers the latest flat-format `IceMix` baseline beside the
run; it has no explicit baseline argument and does not verify event equality.

```bash
python ice_mix/generate_plots.py \
    --base-dir /successor/project/path/to/graphnet/ice_mix/outputs
```

The aggregate command chooses latest runs by inferred project/job identity.
`plot_master.py` also silently attempts the departing author's hard-coded
`JointLargeTC0.04results_LRNEW.csv`. As the author requested, the successor
should search inherited PACE files for that basename and establish provenance;
the current CLI has no argument for supplying the recovered file.

Fine-tune and nu-tau comparison CLIs expose only output-tree discovery:

```bash
python ice_mix/generate_fine_tune_comparison_plots.py \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --dry-run

python ice_mix/generate_nutau_comparison_plots.py \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --dry-run
```

Their selected treatments and event populations must be checked independently.
They do not accept explicit pairs or require matching manifests. Smooth ratio
curves are not evidence that samples were matched.

All current plotting statistics are unweighted: `oneweight` is carried but not
used. Every retained row contributes equally. The scripts do not record a plot
manifest. A weighted mode is researcher work pending a data-owner-approved
normalization and target population.

Always inspect tracks and cascades separately. Their different reconstruction
difficulty permits Simpson's-paradox behavior in the combined population.

## 6. Experimental checkpoint interpolation

`evaluate_rotation_checkpoint_averaging.py` linearly interpolates two state
dictionaries and evaluates several weights. It requires architecture-compatible
checkpoints and a CUDA GPU. Although it seeds PyTorch once, each pass through
over-cap events can select different pulses upstream, so interpolation weights
are not guaranteed matched at pulse level.

```bash
python ice_mix/evaluate_rotation_checkpoint_averaging.py \
    --checkpoint-a /path/to/a.ckpt \
    --checkpoint-b /path/to/b.ckpt \
    --lambdas 0 0.25 0.5 0.75 1 \
    --batch-size 128 \
    --num-workers 3 \
    --limit-val-batches 2 \
    --output-dir /successor/project/path/to/interpolation
```

Treat this as exploratory until the researcher resolves pulse-selection
provenance.

## Common failures

- **Config not found:** preserve `.hydra/config.yaml` with each run.
- **Wrong split:** compare actual result identities against the training split;
  do not trust the evaluation fallback.
- **Wrong checkpoint:** inspect negative/unparseable validation-loss filenames.
- **No rows but successful job:** examine tracebacks and artifact timestamps.
- **Sparse plots:** all-NaN bins can crash log scaling after partial PNG output.
- **Misleading comparison:** job order, project name, or smooth curves do not
  prove treatment or population identity.

## What to try next

Proceed to [Tutorial 5](05_robustness_studies.md) only for researcher-approved,
explicitly caveated robustness work.
