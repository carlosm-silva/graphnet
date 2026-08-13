# Tutorial 5 — Rotation and pulse-loss robustness

**Goal:** measure how reconstruction changes under random token removal and a
historically named complementary-half token study.

**Prerequisites:** Tutorial 4, a validated checkpoint, and a documented evaluation split.

**Expected duration:** dry-run minutes; one-GPU evaluations up to the launcher walltimes (2–12 hours requested), actual time **UNVERIFIED-CLUSTER**.

**Run on:** Phoenix GPU job for inference; plotting can run where dependencies and CSVs are available.

**Verification:** perturbation algorithms and CLI defaults are **VERIFIED-STATIC**. Scientific baselines and metric changes are **UNVERIFIED-CLUSTER**.

## 1. State the hypothesis first

- Random token removal tests tolerance to incomplete pulse collection.
- The historical “checkerboard” test compares deterministic complementary
  random halves of padded token positions. It does **not** encode detector cells
  or a spatial inefficiency pattern.
- Rotation augmentation tests whether azimuthally equivalent events are reconstructed consistently.

Record the checkpoint, split, seed, removal rates, and code revision before running.

## 2. Random token-removal resilience

Discover eligible runs:

```bash
python ice_mix/resilience_test.py \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --n-gpus 1 \
    --drop-percentages 0.05 0.10 0.25 0.50 \
    --test-fraction 0.01 \
    --seed 42 \
    --dry-run
```

The script enables forced token dropping even in evaluation and always adds a
0% table from the same seeded loader as the comparison baseline. Start at 1%
only to validate wiring; use a documented full split for final claims.

With `DATA_ROOT` available, the supported launcher uses seed 42 and a 1%
fraction by default; override them with `ICE_MIX_SEED` and
`ICE_MIX_TEST_FRACTION`:

```bash
sbatch ice_mix/run_resilience_test.sbatch
python ice_mix/plot_resilience_test.py \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --output-dir /successor/project/path/to/resilience_plots
```

## 3. Historical complementary-half (“checkerboard”) removal

Choose the model project explicitly:

```bash
python ice_mix/checkerboard_test.py \
    --model-config IceMix-Standard \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --n-gpus 1 \
    --test-fraction 0.01 \
    --seed 42 \
    --dry-run
```

For a production evaluation, set the project explicitly and submit the generic
launcher, then plot:

```bash
ICE_MIX_MODEL_CONFIG=IceMix-Standard \
ICE_MIX_RUN_DIR=/successor/project/path/to/exact/run \
ICE_MIX_SEED=42 \
    sbatch ice_mix/run_checkerboard_test.sbatch

python ice_mix/plot_checkerboard_test.py \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --output-dir /successor/project/path/to/checkerboard_plots
```

The test seeds a random permutation of each padded sequence, sends one half to
each complementary branch, and averages the two predictions. It changes the
tokens presented to the model and is not detector simulation. The historical
CLI and output names remain for compatibility.

## 4. Compare fairly

Use the same checkpoint family, event split, event fraction, database version,
seed, and weighting/filtering rules across perturbation levels. Preserve each
study's `evaluation_manifest.json`; it records the split and the deterministic
mask semantics. Plotters require matching row identities and use `drop_0.00.csv`
rather than a possibly full-size run prediction as the baseline. Plot both absolute reconstruction quality and change relative
to zero-removal inference. Do not use the test split to tune token-drop rates
and then report it as untouched evaluation.

## Common failures

- **Different event counts:** each removal method must preserve at least one pulse per event; verify result-row identities.
- **Non-reproducible drop masks:** training uses batch/epoch/rank-derived seeds; robustness scripts may use their own forcing path. Record seeds and code revision.
- **Apparent improvement after removal:** check selection, weighting, event identities, and whether difficult events became empty or were filtered.
- **Spatial interpretation:** there is none in the current implementation; do
  not describe the complementary random halves as detector geometry.

## What to try next

Use [Tutorial 6](06_model_changes_and_fine_tuning.md) only after the base and robustness pipelines are reproducible.
