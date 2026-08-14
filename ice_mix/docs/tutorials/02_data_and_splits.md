# Tutorial 2 — Prepared data and event splits

**Goal:** verify the three GraphNeT SQLite inputs, choose a split strategy, and understand what reaches the transformer.

**Prerequisites:** Tutorial 1, read access to prepared project-storage databases, and the compatible `graphnet` environment.

**Expected duration:** 30–90 minutes, excluding data transfer.

**Run on:** PACE login node for lightweight inspection; use a compute allocation if local policy forbids database-intensive queries.

**Verification:** filenames, requested tables, selection algorithms, and tensor expectations are **VERIFIED-STATIC**. A staged three-flavor fixture was inspected locally as described below (**VERIFIED-LOCAL**, 2026-08-13). Production counts and provenance remain **UNVERIFIED-CLUSTER**.

## 1. Identify the inputs

`conf/data/standard.yaml` expects, below `$DATA_ROOT`:

```text
my_numu_database_part_1 (1).db
my_nue_database_part_1 (1).db
my_nutau_database_part_1 (1).db
```

All three flavors are part of current production. Confirm without copying the files:

```bash
export DATA_ROOT=/successor/project/path/to/prepared/sqlite
for flavor in numu nue nutau; do
    find "$DATA_ROOT" -maxdepth 1 -name "my_${flavor}_database_part_1 (1).db" -print
done
```

If naming differs, prefer a successor-specific config override or symlink policy approved by the group; do not rename the only authoritative data copy casually.

## 2. Inspect schema safely

Read only metadata and small samples:

```bash
sqlite3 "$DATA_ROOT/my_numu_database_part_1 (1).db" '.tables'
sqlite3 "$DATA_ROOT/my_numu_database_part_1 (1).db" 'PRAGMA table_info(truth);'
sqlite3 "$DATA_ROOT/my_numu_database_part_1 (1).db" 'PRAGMA table_info(SRTInIcePulses);'
sqlite3 "$DATA_ROOT/my_numu_database_part_1 (1).db" 'SELECT COUNT(*) FROM truth;'
```

Do not publish event content or absolute storage paths. The successor should keep his own PACE copies and ask Jiyuan on Slack for access, provenance, and units; column names alone do not prove units. **Author-confirmed 2026-08-13.**

The staged fixture contains 30 events per flavor and exactly the `truth` and
`SRTInIcePulses` tables. It preserves the production column declarations,
integer `truth.event_no`, and the pulse-table event index. Each flavor contains
an event above the 256-pulse cap. These statements are **VERIFIED-LOCAL** by
read-only SQLite queries and `verify_sample.py` on 2026-08-13; fixture counts
are not production counts.

The author-supplied sample manifest states: detector/truth positions in metres,
pulse time in nanoseconds, energy in GeV, angles in radians, charge in
photoelectrons, `n_pulses` as a count, and dimensionless vMF `kappa`. A SQLite
schema cannot prove those units, so confirm provenance/units with Jiyuan before
scientific use.

## 3. Understand graph construction

For each selected event, GraphNeT reads pulse features and truth, normalizes
detector features through `IceCube86`, and invokes `IceMixNodes`. At most 256
pulses are retained. With `hlc_name=None`, excess-pulse selection is random
rather than HLC-prioritized. The exact input order, transforms, Fourier
embedding, pairwise bias, and output units are in the
[numeric contract](../numeric-contract.md). **VERIFIED-STATIC.**

The transformer receives flattened pulse features plus `batch`, a vector mapping each pulse to an event. It pads this representation only inside the backbone. The seventh named feature and graph edges are unused backward-compatibility artifacts; only indices 0–5 are Fourier-encoded. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

## 4. Choose a split

The recommended currently working default is random mode:

```bash
python ice_mix/verify_config.py --cfg job --resolve data/split=random
```

Training shuffles each database independently with seed 42, then slices
80/10/10. Reproduction also depends on receiving the same SQL row order because
the query has no explicit `ORDER BY`. More importantly, prediction and the two
robustness entry points do **not** read this nested split group: they look for
obsolete flat keys, normally regenerate 42/80-10-10 selections, ignore CSV
mode, and write no evaluation manifest. Verify their event identities outside
the scripts before calling a result validation or test data. **VERIFIED-STATIC.**

CSV mode is intended for frozen selections:

```bash
python ice_mix/verify_config.py --cfg job --resolve data/split=csv
```

Do not train with the checked-in CSV config yet: it has two CSVs for three
databases, and `load_csv_splits` deliberately raises a length error. A staged
fixture confirmed that the compatible layout is one `event_no` CSV per flavor,
positionally ordered nu_mu, nu_e, nu_tau; all six sample files contain unique
integer IDs present in their matching database (**VERIFIED-LOCAL**, 2026-08-13).
For the production CSVs, search the inherited PACE files for the configured
basenames and ask the author or Jiyuan on Slack if ownership/provenance is
unclear; this handoff intentionally does not prescribe a hidden personal path.

## 5. Understand rotation and token dropping

- `data=augmented_rotation` is the current rotation workflow: it rotates pulses and targets at batch time and does not require a physically duplicated database.
- `data=drop` removes 5% of tokens from 5% of events under current overrides.
- `data=drop_aug_rot` composes both.
- A deprecated stored-rotation convention offsets copied event IDs by `rotation_index * 10**10`; compatibility code expands training selections when a database path contains `augmented`.

The last convention is historical only. A staged legacy fixture confirmed IDs
`original + 10**10` and consistent pi/2 rotation of pulse x/y, truth x/y, and
truth azimuth (**VERIFIED-LOCAL**, 2026-08-13). It remains author-confirmed
deprecated and must not replace on-the-fly augmentation.

## Common failures

- **Missing tau file:** inherited base/evaluation launchers commonly stage only
  muon/electron inputs. A researcher must provide or approve three-flavor
  staging and validation.
- **SQLite locked/corrupt:** inspect the authoritative copy read-only and work on a controlled copy for indexing.
- **Database read failure:** `get_dynamic_splits` logs the exception, appends
  empty selections, and continues. Treat any empty split as a possible path or
  permission failure rather than valid zero-event data.
- **CSV length mismatch:** provide exactly one CSV per database, in the same order.

## What to try next

Run the [single-GPU wiring smoke test](../examples/README.md) if appropriate, then proceed to [Tutorial 3](03_base_training_and_resume.md).
