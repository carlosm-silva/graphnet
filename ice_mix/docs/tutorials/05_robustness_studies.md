# Tutorial 5 — Inspecting rotation and pulse-loss studies

**Goal:** understand what the inherited robustness utilities actually perturb,
run only their read-only discovery modes where available, and identify the
researcher decisions required before a physics interpretation.

**Prerequisites:** Tutorial 4, an inspected checkpoint, and researcher ownership
of any software correction.

**Expected duration:** discovery takes minutes. GPU evaluation time is
**UNVERIFIED-CLUSTER**; inherited jobs request 2–12 hours.

**Run on:** discovery where the environment and run tree are available; GPU
evaluation only after researcher review.

**Verification:** algorithms, CLI defaults, and reproducibility gaps are
**VERIFIED-STATIC**. Scientific robustness conclusions are not verified.

## 1. State the implemented perturbations accurately

- Training-time token drop removes random input tokens from selected events.
- `resilience_test.py` forces token drop during evaluation at requested rates.
- The historically named checkerboard path partitions **padded sequence
  positions** into fixed-seed random complementary halves. It never constructs
  detector cells and is not a spatial inefficiency simulation.
- On-the-fly rotation changes event azimuth in the training callback. Stored
  fixed-rotation databases are deprecated.

## 2. Inspect random token-removal discovery

```bash
python ice_mix/resilience_test.py \
    --base-dir /successor/project/path/to/ice_mix/outputs \
    --n-gpus 1 \
    --drop-percentages 0.05 0.10 0.25 0.50 \
    --test-fraction 0.01 \
    --dry-run
```

There is no `--seed` option. Fractional event selection uses unseeded
`numpy.random.choice`; token masks are not persisted; repeated loader passes can
resample capped pulses. The script does not automatically create a 0% table.
`plot_resilience_test.py` instead labels the ordinary run prediction as “0%
Drop,” even when it contains a different population from a fractional
robustness table.

The inherited `run_resilience_test.sbatch` hard-codes paths, stages only two
databases, and has no copy validation or seed control. Read it, but do not treat
it as a successor launcher:

```bash
sed -n '1,220p' ice_mix/run_resilience_test.sbatch
```

## 3. Inspect the historical complementary-half path

The checkerboard CLI has no dry-run, seed, or exact-run option. It discovers
all Hydra configs for a required project name, parses validation loss from
checkpoint filenames, and chooses what it believes is the best run.

```bash
python ice_mix/checkerboard_test.py --help
sed -n '1,220p' ice_mix/run_checkerboard_test.sbatch
```

Do not execute the evaluator until the researcher has checked:

- the automatically selected run and checkpoint;
- negative-loss filename parsing;
- regenerated evaluation split identities;
- hard-coded `max_pulses=256` and omitted numerical model settings;
- raw checkpoint loading for an EMA run;
- unseeded fractional event selection;
- random upstream pulse capping on each half's loader pass.

Inside the monkey-patched forward method, a generator is always seeded with 42
for each call. It ranks valid padded positions and assigns the first and second
halves to separate predictions. This fixed mask does not make the entire study
reproducible because batching, event subsampling, pulse order, and upstream
capping can change its inputs.

The plotter merges half tables only on `event_no`. Event numbers are not proven
globally unique across the three databases, and duplicate keys can create a
many-to-many merge. Independently validate database-qualified identities and
row multiplicities before interpreting half-to-half separation.

## 4. Rotation reproducibility

Modern rotation is on the fly, but `conf/data/standard.yaml` currently sets
`rotation_seed: null`. `RandomRotationCallback` therefore seeds its private
generator from system entropy rather than the global run seed. A saved global
`seed: 42` does not reproduce the training rotations.

The deprecated staged fixed-rotation sample remains useful only for checking
the historical $10^{10}$ event-ID offset and coordinate transformation. It is
not the recommended augmentation workflow.

## 5. Minimum evidence for a researcher-approved study

For every perturbation level preserve:

```text
code revision and researcher-approved implementation
checkpoint and resolved training config
database identities and evaluation event IDs per database
retained pulse identities or deterministic capping policy
event-subsampling and perturbation seeds/policies
row counts and database-qualified identities
unweighted/weighted statistical convention
process exit status and artifact hashes
```

Without these, label results exploratory and do not claim matched robustness.

## Common failures

- **Different event populations:** ordinary prediction and fractional studies
  can select different rows.
- **Different pulse populations:** each loader pass can resample events at the
  pulse cap.
- **EMA load failure:** robustness evaluators load raw checkpoint state.
- **Slurm says completed but files are absent:** evaluator exceptions are caught
  without a nonzero final exit status.
- **Spatial interpretation:** there is none in the checkerboard implementation.
- **Apparent improvement after removal:** first rule out selection, topology,
  weighting, and pulse-sampling changes.

## What to try next

Use [Tutorial 6](06_model_changes_and_fine_tuning.md) only after the responsible
researcher has established a reproducible evaluation contract.
