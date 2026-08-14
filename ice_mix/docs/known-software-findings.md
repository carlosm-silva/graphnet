# Known software findings — researcher action only

This page records likely defects and reproducibility hazards found during the
handoff audits. They are **not fixes** and they are not instructions for a
documentation maintainer to change the implementation. Software changes are
owned by the researcher responsible for IceMix. A successor should reproduce,
prioritize, and test a finding before changing scientific behavior.

Unless stated otherwise, each item is **VERIFIED-STATIC** from the inherited
source. PACE, GPU, DDP, and full-data consequences remain
**UNVERIFIED-CLUSTER**.

## Scientific-comparison blockers

### Random pulse capping is not represented in provenance

`IceMixNodes` calls `torch.randperm` when an event reaches the 256-pulse cap.
Evaluation can therefore reconstruct a different capped pulse set on each
loader pass. Prediction artifacts identify events but do not record retained
pulse rows or a deterministic per-event capping policy. Two models, token-drop
levels, complementary halves, or interpolation weights can consequently see
different pulses while appearing to use the same events.

**Researcher decision required:** define the scientific pulse-selection policy
and its persisted provenance before treating repeated evaluations as matched.

### Evaluation does not reconstruct the saved split policy

Training reads the nested `data.split` Hydra group and supports random and CSV
selections. `predict.py`, `resilience_test.py`, and `checkerboard_test.py` look
for older flat `data.split_seed` and `data.split_ratio` keys, fall back to
42/80-10-10, and do not reproduce CSV selections. They do not write an event
selection manifest.

**Researcher decision required:** do not call those outputs validation/test
results from the training split until event identities have been independently
checked.

### Comparison scripts infer treatments and do not prove matched events

Fine-tune and nu-tau comparisons discover runs from project names and job-number
ordering, bin tables independently, and do not require equal event populations.
`plot_run.py` also discovers a baseline automatically. `plot_master.py` may
silently load the departing author's hard-coded
`JointLargeTC0.04results_LRNEW.csv` path when it exists.

**Researcher decision required:** select every treatment/reference explicitly,
recover the reference CSV by searching the inherited PACE files, document its
provenance, and check event populations outside these scripts before making a
physics comparison.

### Current statistics are unweighted

Prediction tables carry `oneweight`, but the supplied plotting functions use
ordinary row-wise medians and 16th/84th percentiles. They do not write a plot
input/weighting manifest. The authoritative target population and `oneweight`
normalization are not defined in this package.

**Researcher decision required:** describe existing plots as unweighted. A
weighted analysis requires a data-owner-approved convention and researcher
implementation.

## Training and checkpoint hazards

### Resume can compose a different configuration

The inherited launchers locate or accept checkpoints, but do not reconstruct
and compare every scientific setting from the saved `.hydra/config.yaml` before
Lightning restores state. Shape-compatible changes to data, augmentation,
optimization, or model numerics can therefore enter a nominal continuation.

**Researcher decision required:** compare the saved and proposed resolved
configuration manually before resuming. Do not call the current path a safe or
configuration-validated resume.

### Complementary-half evaluation reconstructs incomplete model semantics

`checkerboard_test.py` hard-codes `max_pulses=256`, omits several numerical
`IceMix` constructor settings that training reads from Hydra, and loads the raw
checkpoint state into a plain `StandardModel`. Shape-compatible overrides can
silently change evaluation mathematics; EMA checkpoint state can be
incompatible with this load path.

### Relative-attention DropPath is applied twice

With baseline `init_values=1.0`, the relative-attention residual calls the same
`DropPath` module inside and outside the layer-scale multiplication. At nominal
rate $p$, the attention branch survives both independent calls with probability
$(1-p)^2$; the last relative block therefore has effective drop probability
$1-0.8^2=0.36$, not 0.20. DropPath is inactive during evaluation, but training
dynamics and attempts to reproduce training are affected.

**Researcher decision required:** establish whether this is intentional before
changing or documenting a new architecture contract.

### Checkpoint filename parsing excludes negative losses

The discovery regex accepts only unsigned decimal validation losses. The 3D-vMF
continuous-density negative log likelihood is not structurally restricted to
positive values. Negative-loss checkpoints may be skipped or a worse positive
checkpoint may be selected. Scientific notation is also unsupported.

**Researcher decision required:** inspect the selected checkpoint path manually;
do not assume automatic discovery chose the true minimum.

## Reproducibility and operational hazards

### Stochastic robustness inputs are incompletely controlled

The current rotation config sets `rotation_seed: null`, so on-the-fly angles use
system entropy rather than the global run seed. Resilience event subsampling uses
unseeded NumPy selection, and its forced token masks are not persisted. The
complementary-half mask itself uses a fixed seed of 42, but upstream random pulse
capping and unseeded fractional event selection remain uncontrolled.

### Base and evaluation launchers are not submit-ready successor templates

`run_training.sbatch` stages only nu_mu and nu_e even though production uses all
three flavors, does not validate copy exit statuses/destinations, falls back to
shared `/tmp`, and embeds the author's paths/account/email. The prediction and
robustness launchers also hard-code author/Jiyuan paths and stage only two
databases. Other variant launchers have additional literal-YAML staging and
fixed-rank hazards. These files are historical workflow evidence, not safe
successor launchers.

### Evaluation programs can report process success after failures

Prediction and robustness CLIs catch per-run exceptions, log tracebacks, and
normally return exit status zero. Missing configs/checkpoints are also skipped.
A Slurm `COMPLETED` state therefore does not prove that requested artifacts were
created or refreshed.

### Database read failure can degrade to empty selections

`get_dynamic_splits` logs a database exception, appends empty train/validation/
test selections, and continues. Downstream empty loaders can obscure the true
path or permission failure.

### Dependency declarations conflict with the custom boundary

The inherited `ice_mix/requirements.txt` is loose and requests public
`graphnet>=1.0.0`, while IceMix requires checkout-specific `IceMixNodes` and
`GraphNeTDataModulecustom`. Use the author-supplied environment evidence in
`docs/graphnet_env/` and verify the custom imports; do not treat the loose file
as a reproducible environment lock.

## Data and plotting prerequisites

- Production campaign identity, preprocessing, source checksums, physical
  units, event population, and `oneweight` meaning require confirmation from
  Jiyuan or another data owner. A dataset card has not been supplied.
- The checked-in CSV split configuration lists two selections for three
  databases. Search inherited PACE files for the configured basenames before
  deciding whether CSV mode is still used.
- Sparse topology subsets can leave every plotted bin empty and trigger a
  Matplotlib log-scale failure after some PNGs have already been written.
- The historically named checkerboard operation is a random complementary
  partition of padded sequence positions, not detector cells or spatial
  inefficiency. Never give it a spatial interpretation.

## How to close a finding

Only mark an item resolved after the researcher has supplied or approved the
behavioral change, added an appropriate characterization/regression test, and
verified the intended scientific comparison. PACE-specific changes also need a
small cluster run whose command, resolved configuration, selected events,
checkpoint, exit status, and produced artifacts are retained.
