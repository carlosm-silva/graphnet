# Questions for the IceMix author

Questions are batched during implementation. Durable answers are copied into `MEMORY.md`.

## Data-staging checklist

Use `ice_mix/docs/_local_sample/`; it is gitignored and will never appear in documentation as a real data path.

- [x] D001 — 20–50 sanitized standard events for each of nu_mu, nu_e, and nu_tau. Delivered and dependency-light checks passed 2026-08-13; GraphNeT construction/truncation still needs a compatible environment.
- [x] D002 — 2–5 original events plus one rotated copy each from one deprecated fixed-rotation database (legacy verification only). Delivered and verified 2026-08-13.
- [x] D003 — train and validation CSVs for all three databases, 2–5 `event_no` rows each. Delivered and verified 2026-08-13.
- [x] D004 — one checkpoint plus its `.hydra/config.yaml` and CSV metrics log. Delivered 2026-08-13; container/metadata/metrics, strict PyTorch loading, one-event CPU forward, and CPU inference verified. True Lightning resume remains unexecuted locally.
- [x] D005 — one prediction `results.csv` plus `model_config.yml`. Delivered and schema/metric-input checks passed 2026-08-13.
- [x] D006 — one successful and one failed/killed Slurm report, sanitized. Delivered and expected success/root-cause landmarks verified 2026-08-13.

The author later authorized account identifiers and personal PACE paths in the
handoff because the successor is expected to receive equivalent permissions.
No `.env` or credential values were staged: those paths were technically
inaccessible to the documentation agent. The successor should recover the
actual file directly from the author's PACE project area or ask him on Slack.

## Questions answered during implementation

### Q001 — Literal day-to-day workflow

- **Question:** Which exact base launcher and command do you normally use for a new production run, then for resume, inference, and plots?
- **Why it matters:** Several overlapping launchers exist, and the tutorials need one recommended path.
- **Inferred:** `train.py` is canonical; `run_standard.sbatch`, `run_augmented.sbatch`, `run_drop.sbatch`, and `run_drop_aug_rot.sbatch` select base variants.
- **Best guess:** Submit the variant-specific launcher, use its matching `_resume` script after preemption/walltime, then `run_predict.sbatch` and `generate_plots.py`.
- **Answer (2026-08-13):** The inference is correct: the variant-specific base files start new runs, and their matching resume versions continue them.
- **Status:** answered

### Q002 — PACE operational facts

- **Question:** What login host/onboarding steps, typical queue waits and walltimes, allocation limits, and checkpoint/requeue practice should the successor expect?
- **Why it matters:** These cannot be verified from code or from this laptop.
- **Inferred:** Phoenix uses Slurm; jobs use the `inferno` QoS, the L40S allocation, one node, and up to 72 hours.
- **Best guess:** The successor obtains a PACE account and group allocation, logs into the Phoenix login node, and manually resubmits a matching resume job after walltime.
- **Answer (2026-08-13):** The inference is correct. The L40S account can only be used on L40S nodes and should be prioritized. Other group members know PACE; ask Jiyuan on Slack when in doubt.
- **Status:** answered

### Q003 — Data provenance and storage

- **Question:** Where do the three SQLite databases originate, what preprocessing have they received, what units do their pulse/truth columns use, and what authoritative project-storage directory should replace `DATA_ROOT`?
- **Why it matters:** The data tutorial must distinguish raw IceCube data from prepared GraphNeT inputs.
- **Inferred:** They contain `SRTInIcePulses` and `truth`; all three flavors are production inputs and should live in project storage before `$TMPDIR` staging.
- **Best guess:** They are prepared GraphNeT SQLite databases with detector coordinates/vertex in metres, time in nanoseconds, and event IDs in `truth`.
- **Answer (2026-08-13):** The successor should keep his own copies in his PACE workspace. Ask Jiyuan for access and for authoritative clarification of units and provenance.
- **Status:** answered

### Q004 — Correct output and known failures

- **Question:** What log/metric behavior tells you a run is healthy, and what bugs, bad nodes, numerical failures, or misleading outputs should the successor know first?
- **Why it matters:** Expected values and failure signatures must not be fabricated.
- **Inferred:** `val_loss`, position/angular errors, kappa, vMF calibration, finite-loss checks, sampler diagnostics, and checkpoints are monitored; node `atl1-1-01-010-35-0` is excluded throughout.
- **Best guess:** Stable finite training/validation loss and newly written best/last checkpoints are the minimum success criteria; the excluded node had GPU faults.
- **Answer (2026-08-13):** To the author's knowledge there is one bad L40S node. A Slurm bug can ignore the exclusion flag, so the operational workaround is to list every other node as acceptable; selection of the bad node crashes loudly after initialization. Regular L40S training runs at roughly one epoch per hour. Follow training/validation curves in W&B; any NaN loss is a bad state, although the author expects the 3D-vMF commit to have fixed that failure. Always judge improvement with comparison plots. Combined plots can exhibit Simpson's paradox because tracks reconstruct better than cascades: changing their mixture can worsen the combined curve even when each topology improves separately.
- **Status:** answered

### Q005 — Graph and feature mismatches

- **Question:** Is it intentional that training uses edge-free `GraphDefinition` while prediction uses `KNNGraph`, and that GraphNeT supplies seven named features while IceMix is configured with `n_features=6`?
- **Why it matters:** These may be harmless metadata differences or real checkpoint/inference sharp edges.
- **Inferred:** `include_dynedge=false`, so the transformer itself consumes only dense pulse features and Fourier-encodes indices 0–5; `pmt_area` at index 6 is ignored by the encoder.
- **Best guess:** Both are inherited compatibility artifacts, but this needs author confirmation.
- **Answer (2026-08-13):** IceMix is not a GNN model. The graph parameters and extra feature are backward-compatibility artifacts and are unused by the transformer.
- **Status:** answered

### Q006 — Historical code status

- **Question:** May the handoff classify `src/trainer.py`, `hello.py`/node probes, `*_OLD`, `run_augmented_indexed.sbatch`, `run_predict_temp.sbatch`, and exploratory tests as abandoned or historical?
- **Why it matters:** They should not be recommended accidentally.
- **Inferred:** `src/trainer.py` has no caller; the node probes only print hello; OLD is explicit; indexed is byte-identical to OLD; the temporary predictor uses another account and A100.
- **Best guess:** All are historical/suspected-dead and should remain only for provenance.
- **Answer (2026-08-13):** The inferred classification is correct; these files are abandoned/historical.
- **Status:** answered

## Planning questions answered earlier

### Q007 — Documentation file scope

- **Answer (2026-08-13):** Root ledgers and source/config-folder READMEs are allowed. Documentation may add/change/remove comments and docstrings anywhere as long as functionality does not change.
- **Status:** answered

### Q008 — Smoke configuration

- **Answer (2026-08-13):** Include the optional docs-only reduced single-GPU smoke configuration because the successor's hardware is unknown and it may also help local testing.
- **Status:** answered

### Q009 — Canonical training family

- **Answer (2026-08-13):** Recommend base launchers. Fine-tuning is the latest work and has shown some success, but is not known to be fully ready.
- **Status:** answered

### Q010 — PACE runtime setup

- **Answer (2026-08-13):** `anaconda3/2022.05.0.1` and conda environment
  `graphnet` are current. The author subsequently supplied exact Conda history,
  explicit, YAML, and pip exports under `docs/graphnet_env/`; the pip export pins
  the custom GraphNeT checkout to commit
  `4394131647b4a581e7d4923361b2814ab9e03ff5`.
- **Status:** answered

### Q011 — Production flavor inputs

- **Answer (2026-08-13):** Production uses all three standard databases: nu_mu, nu_e, and nu_tau.
- **Status:** answered

### Q012 — Storage tiers

- **Answer (2026-08-13):** Keep authoritative prepared databases and durable outputs in project storage, stage databases into `$TMPDIR`, and use scratch for caches/transient work.
- **Status:** answered

### Q013 — AI-assisted development and contact

- **Answer (2026-08-13):** Make clear that Codex and ChatGPT substantially assisted development. The author mostly no longer writes code directly but is willing to help; the successor should send a Slack message with questions.
- **Status:** answered

### Q014 — Rotation augmentation status

- **Answer (2026-08-13):** Current rotation augmentation is generated on the fly. Stored fixed-rotation databases are deprecated; any staged copy is for completeness and legacy documentation only.
- **Status:** answered
