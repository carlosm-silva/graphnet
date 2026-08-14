# IceMix durable knowledge

This file records stable project knowledge for the handoff. It is not a work log or task list.

## Package purpose and high-level architecture

- IceMix reconstructs an IceCube interaction vertex and neutrino direction from a variable-length set of detector pulses.
- Each event is read from GraphNeT-compatible SQLite databases. GraphNeT constructs a PyTorch Geometric `Data` graph, while IceMix supplies the transformer backbone and training-specific extensions.
- The backbone Fourier-encodes pulse coordinates, time, charge, and RDE; applies pairwise spacetime-relative attention; prepends a learned class token; applies ordinary transformer blocks; and returns one event embedding.
- GraphNeT's `JointPositionandDirectionReco` maps the embedding to seven values: three position coordinates, a three-component unit direction, and von Mises-Fisher concentration `kappa`.
- Production variants are standard data, token dropping, on-the-fly azimuthal rotation, and token dropping plus rotation. Stored fixed-rotation databases are deprecated.
- Base training is the recommended production workflow. Distributed LBFGS and AdamW+EMA last-block fine-tuning are recent experiments with some success, not the default.

## Execution flows

### PACE base training: submission to checkpoint

1. The author recommends the base-training family scientifically, but none of the checked-in launchers is a successor-ready canonical launcher. Each must be reviewed and edited by the researcher before submission.
2. `run_training.sbatch` requests one Phoenix GPU node, eight L40S GPUs, 32 CPUs, `inferno` QoS, all node memory, and 72 hours. It contains author-specific paths/account values, stages only the muon- and electron-neutrino databases, assumes `/tmp`, and does not validate each copy.
3. It loads `anaconda3/2022.05.0.1`, activates the `graphnet` conda environment, probes GPUs, and launches one worker per healthy visible GPU. Other variant launchers implement related but inconsistent historical workflows.
4. `srun torchrun --nproc_per_node=<healthy GPU count> ice_mix/train.py ...` starts distributed training after those launcher-specific setup steps.
5. Hydra composes `conf/config.yaml`, one attention config, one data config, and one split config. Command-line overrides name the experiment variant.
6. `train.py` seeds the run, creates a GraphNeT graph definition/data module, constructs IceMix and the joint reconstruction task, then calls `StandardModel.fit` with Lightning DDP.
7. CSV and optional W&B loggers receive losses and physics metrics. `ModelCheckpoint` writes the best three checkpoints and `last.ckpt` below the run's `checkpoints/` directory.

### Resume and fine-tuning

- Historical resume launchers use `resume_common.sh` to select a recent run/checkpoint. They do not prove that the resumed Hydra configuration matches the original run; the researcher must compare the saved and proposed configurations before submitting.
- Resume (`ckpt_path`) restores the Lightning training state. Fine-tuning (`fine_tune_from_ckpt`) loads weights into a new run and may freeze all but the task head and last transformer blocks.
- `DistributedLBFGSStandardModel` makes repeated LBFGS closures deterministic per batch and synchronizes loss values across DDP ranks.
- `EMAStandardModel` trains online AdamW weights and validates/checkpoints with an FP32 exponential-moving-average copy.

### Inference and analysis

- `predict.py` scans run directories, reconstructs the configured model, chooses the lowest-loss checkpoint by filename, prefers EMA weights when present, and writes prediction CSV/model artifacts. Its evaluator-side split defaults do not reconstruct every training split policy, and filenames with negative or exponent-form losses may be misparsed.
- `resilience_test.py` forces token removal at selected rates. The historically named `checkerboard_test.py` does **not** implement spatial cells: it partitions padded pulse-sequence positions into seeded random complementary halves. Its filename/artifact names remain only for compatibility and must not support claims about spatial detector inefficiency.
- `evaluate_rotation_checkpoint_averaging.py` linearly interpolates two checkpoint state dictionaries and evaluates each interpolation weight.
- `generate_plots.py` orchestrates per-run, reference, and master plots. Dedicated fine-tuning and tau-neutrino comparison scripts infer input pairs from directory layouts, do not verify identical event populations, and include hard-coded reference behavior. Statistics are unweighted and no plot manifest records input provenance.
- `run_predict.sbatch`, `run_resilience_test.sbatch`, and `run_checkerboard_test.sbatch` are historical evaluation launchers, not successor-ready supported paths. Their account/path assumptions, staging, run selection, and model settings require researcher review.

## Per-file summaries

### Core and configuration

| Path | Purpose and key symbols | Depends on / used by |
|---|---|---|
| `conf/config.yaml` | Hydra defaults, training, optimizer, scheduler, EMA, paths | All `train.py` launches |
| `conf/attention/baseline.yaml` | IceMix dimensions, regularization, encoder scales | `train.py`, evaluation scripts |
| `conf/data/*.yaml` | Standard, rotation, token-drop data variants | Hydra composition |
| `conf/data/split/*.yaml` | Random or per-database CSV event selection | `train.py` |
| `src/models/layers.py` | Relative attention, transformer blocks, MLP, stochastic depth; also older graph layers | `src/models/transformer.py` |
| `src/models/transformer.py` | `IceMix`, the event-level transformer backbone | Training and inference entry points |
| `src/models/ema_model.py` | `EMAStandardModel`, EMA checkpoint extraction | EMA training and `predict.py` |
| `src/models/lbfgs_model.py` | DDP-safe LBFGS and last-block freezing | Fine-tuning launches |
| `src/metrics_logging.py` | Joint loss decomposition and distributed physics metrics | `train.py` callbacks |
| `src/utils.py` | GraphNet feature/truth lists, splits, augmentation and diagnostic callbacks | Training and inference entry points |
| `src/trainer.py` | Unreferenced custom OneCycle/SWA loops | Suspected-dead; no caller found |
| `train.py` | Canonical Hydra/Lightning training entry point | Production training scripts |

### Evaluation and utilities

| Path/group | Purpose | Status/dependencies |
|---|---|---|
| `predict.py` | Batch inference over discovered runs | Active, called by prediction jobs |
| `resilience_test.py`, `checkerboard_test.py` | Robustness inference | Active research utilities |
| `evaluate_rotation_checkpoint_averaging.py` | Checkpoint interpolation evaluation | Recent experimental utility |
| `plot_*.py`, `generate_*.py` | CSV-derived plots and comparisons | Postprocessing; several are manually invoked |
| `scripts/add_indices.py` | Adds SQLite indexes for lookup performance | Standalone preprocessing utility |
| `verify_config.py` | Composes Hydra config and prints token-drop values; `--cfg job --resolve` prints the full config | Standalone diagnostic |
| `reproduce_token_drops.py` | Synthetic token-drop reproduction | Developer diagnostic |
| `test_*.py` | Unit, distributed, and exploratory checks | Mixed pytest and manual scripts |
| `hello.py`, `test_atl*.sbatch` | Historical node-access probes | Suspected-dead |
| `*_OLD`, `run_augmented_indexed.sbatch`, `run_predict_temp.sbatch` | Older/duplicate launch variants | Author-confirmed abandoned/historical |
| `outputs/` | Checked-in historical Hydra and model-config metadata; large run products are ignored | Evidence of prior runs, not source |

## GraphNeT interface boundary

- `IceMix` subclasses GraphNeT `GNN`, which supplies `nb_inputs`/`nb_outputs` and requires `forward(Data) -> Tensor`.
- `EMAStandardModel` and `DistributedLBFGSStandardModel` subclass GraphNeT `StandardModel`; IceMix relies on its task chaining, loss computation, Lightning hooks, `fit`, prediction, and serialization.
- `GraphDefinition` or `KNNGraph` combines `IceCube86` detector normalization with `IceMixNodes`. `IceMixNodes` caps pulses at `max_pulses`; with `hlc_name=None`, sampling is random.
- `SQLiteDataset` reads pulse and truth tables. `GraphNeTDataModulecustom` accepts one selection list per database and creates loaders.
- `JointLabel` creates `[position_x, position_y, position_z, dir_x, dir_y, dir_z]`; `JointPositionandDirectionReco` returns those six physical values plus `kappa`.
- `EuclideanDistanceLoss`, `VonMisesFisher3DLoss`, and `JointLoss` define the objective. IceMix extends `JointLoss` only to expose metric components.
- `FourierEncoder`, `SpacetimeEncoder`/`MahalanobisEncoder`, `array_to_sequence`, and optional `DynEdge` are called directly by the backbone.
- This checkout contains project-specific GraphNeT additions (`IceMixNodes`, `GraphNeTDataModulecustom`, embedding parameters). Compatibility with an arbitrary public `graphnet>=1.0.0` install is not established.

## Cluster and environment facts

- Target: Georgia Tech PACE Phoenix, not the documentation laptop.
- Author-confirmed 2026-08-13: `module load anaconda3/2022.05.0.1` and `conda activate graphnet` remain current.
- Author supplied 2026-08-13: `docs/graphnet_env/` contains raw PACE exports plus a prefix-free two-stage successor recipe pinned to GraphNeT commit `4394131647b4a581e7d4923361b2814ab9e03ff5`, Python 3.8.20, PyTorch 2.2.0+cu118, PyG 2.6.1, and Lightning 2.4.0. Recreation on Phoenix is still unverified.
- The handoff documentation is committed after that source commit, but the GraphNeT source tree remains the exported tree `23b5e9fdf028460f1ea9808e409e08e5e9a793cc`; use `git rev-parse HEAD:src/graphnet` as the compatibility guard.
- Author-confirmed 2026-08-13: authoritative prepared databases and durable outputs belong in project storage; database reads are accelerated by copying to job-local `$TMPDIR`; scratch is transient/cache storage.
- Author-confirmed 2026-08-13: production uses all three standard databases: muon-, electron-, and tau-neutrino samples.
- Typical production scale is 8 L40S GPUs with 48 GB VRAM each. Baseline batch size 256 and 16-mixed precision are cluster-scale, not safe defaults for an 8 GB consumer GPU.
- Checked-in accounts, allocations, emails, and paths map the departing author's setup. The successor is expected to receive equivalent permissions and may retain an authorized shared allocation/project location, but must validate access and change user-specific email and writable home/output/cache paths.
- Local-only 2026-08-13: initially, none of the visible conda environments contained PyTorch.
- Local-only 2026-08-13, superseding the prior environment limitation: the author authorized dependency installation. The named Miniconda environment directories were unwritable, so verification used a disposable external prefix with Python 3.11, PyTorch 2.2.0+cu118, matching PyG wheels, and this checkout installed editable. This is not the PACE environment or a durable handoff path.
- Local-only 2026-08-13: the CUDA-enabled runtime imported, but the workstation driver/NVML was unavailable and no GPU was visible. CPU GraphNeT checks remained possible.

## Conventions, invariants, magic numbers, and units

- Production pulse map is `SRTInIcePulses`; truth table is `truth`; event identity is `event_no`.
- `max_pulses=256`; events with more pulses are subsampled by GraphNeT `IceMixNodes`.
- Pulse capping is random when `hlc_name=None`; the selected pulse indices are not written to prediction provenance, so repeat evaluations need not use identical pulses even for the same event.
- Model input is a flattened pulse tensor `[total_pulses, features]` plus a graph-to-event batch vector. The transformer uses padded `[batch, pulses, features]` sequences.
- Code expects coordinate/time/charge/RDE ordering at least through indices 0–5. The GraphNeT feature list currently also includes `pmt_area`; the encoder is configured with `n_features=6`, so the seventh feature is not Fourier-encoded.
- Output layout is `[x, y, z, dir_x, dir_y, dir_z, kappa]`; target layout omits `kappa`.
- Position loss and logged `pos_err_m` are treated as metres; angular error is logged in degrees; azimuthal augmentation angles are radians.
- Joint objective is `alpha * position_loss + direction_loss`, with default `alpha=0.026`.
- Deprecated stored-rotation databases used event IDs inferred as `base_event_no + rotation_index * 10**10`; this convention is retained only for historical compatibility.
- Random split defaults to `[0.8, 0.1, 0.1]` with seed 42.
- Training supports nested random/CSV split configuration, whereas several evaluation utilities retain legacy flat random defaults. This can silently evaluate a population different from the one implied by the training configuration.
- CSV split mode presently lists two CSVs for three production databases. The code validates the list length; a three-database CSV run therefore needs researcher-supplied configuration.
- The global training seed does not control every stochastic path. On-the-fly rotation can use entropy when its seed is `null`; pulse capping, robustness event subsampling, and complementary-half subset selection also have unrecorded or incomplete seed control.
- Resilience plotting may substitute the ordinary prediction CSV for a 0%-drop baseline. Complementary-half plotting merges on `event_no` only, which is not sufficient evidence that all rows refer to the same source database and event population.
- Baseline transformer: hidden size 384, 12 ordinary blocks, 4 relative blocks, head size 32, and learned class-token pooling.
- Current plotting quantiles are unweighted and give every retained prediction row equal weight. `oneweight` is propagated but unused; its normalization/target population is not defined in this package. Plotting utilities do not emit an input-provenance manifest.

## Author-supplied knowledge

- 2026-08-13: Base launchers are the recommended production workflow. Fine-tuning is the newest work and has had some success, but is not yet known to be fully reliable.
- 2026-08-13: The package was developed with substantial assistance from OpenAI Codex and ChatGPT. The author mostly does not type implementation code directly now, but remains willing to answer successor questions by Slack.
- 2026-08-13: Software changes of every kind are strictly outside this handoff's scope. Documentation may update, add, or remove comments and docstrings only when executable behavior remains unchanged. Suspected bugs must be flagged for the researcher; the researcher alone decides and implements any fix.
- 2026-08-13: Rotation augmentation is now generated on the fly by the training callback. Stored fixed rotations are deprecated and are being staged only for completeness.
- 2026-08-13: The author described variant-specific base/resume launchers as the historical day-to-day workflow. The base-training family remains the scientific recommendation, but the checked-in launchers are not successor-ready and no documentation change may alter their behavior.
- 2026-08-13: The L40S allocation can only run on L40S nodes and should be prioritized. Ask Jiyuan on Slack for PACE operational help.
- 2026-08-13: The successor should maintain his own PACE copies of all three databases and ask Jiyuan for access, provenance, and authoritative units.
- 2026-08-13: IceMix is a transformer rather than a graph neural network; persistent graph parameters and the extra seventh feature are unused backward-compatibility artifacts.
- 2026-08-13: `src/trainer.py`, node probes, `*_OLD`, `run_augmented_indexed.sbatch`, `run_predict_temp.sbatch`, and exploratory tests are confirmed abandoned/historical.
- 2026-08-13: To the author's knowledge one L40S node is bad. Phoenix Slurm may ignore its exclusion flag; the author's workaround is an allow-list of all other nodes. Selection of the bad node crashes loudly after initialization.
- 2026-08-13: Regular eight-L40S training progresses at roughly one epoch per hour. NaN in any loss is a failed state, although the author expects the 3D-vMF change to have fixed the known source.
- 2026-08-13: Use W&B train/validation curves and comparison plots to assess a model. Inspect tracks and cascades separately: their different reconstruction difficulty means a changed topology mixture can move the combined curve in the opposite direction from both strata (Simpson's paradox).
- 2026-08-13: The successor is expected to receive permissions equivalent to the author's. Checked-in PACE allocation/account/storage identifiers may remain in the handoff as an access map. Secret `.env` values were technically inaccessible to the documentation agent and were not copied.

## Open uncertainties

- Exact Phoenix login host, allocation onboarding procedure, typical queue wait, and allocation limits; ask Jiyuan where code/current PACE docs are insufficient.
- Provenance, preprocessing, detailed schemas, and physical units of the production SQLite databases; Jiyuan is the authoritative contact.
- Whether CSV split mode is current; the checked-in production config presently lists two CSVs for three databases and will fail its length check. Search the inherited PACE files for the configured CSV basenames and confirm provenance with the author or Jiyuan.
- The local verification bundle is deliberately ignored. `docs/local-sample-manifest.md` records byte sizes/SHA-256 hashes and the run identity; its upstream production-dataset version remains unverified.

## Documentation conventions

- NumPy-style sections are used for new complete docstrings because the inherited code has no dominant complete parameter/return convention.
- Narrative pages explain physics, data flow, PACE operations, and limitations; symbol-level mechanics live in Python docstrings.
- The adjacent `docs/job-scripts.md` catalog is the line-by-line explanation for all 42 Slurm and two shell files.
- Historical output subdirectories are documented centrally in `outputs/README.md`, not modified individually.
- Local static audit 2026-08-13: all 33 package Python files parse and have module/public-symbol docstrings; stripping docstrings produced executable ASTs identical to the then-current code; all 42 root Slurm files and both root shell helpers passed `bash -n`; local Markdown links resolved; `git diff --check` passed.
- Local runtime audit 2026-08-13: project-specific GraphNeT imports passed; `verify_sample.py` passed all six artifact groups; `verify_graphnet.py` built/truncated all three flavors to 256 nodes, strictly loaded the real checkpoint, and produced a finite CPU output of shape `(1, 7)`; 21 token-drop/EMA/LBFGS tests passed; a synthetic on-the-fly rotation check passed; Hydra `--cfg job --resolve` printed the rotation config; prediction and resilience dry-runs selected the staged checkpoint; full CPU `predict.py` inference wrote nine finite rows plus state/config artifacts.
- Local runtime sharp edges 2026-08-13: full reduced training reached a 258K-parameter Lightning model but failed before the first batch because `CheckSamplerCallback` calls `torch.cuda.current_device()` unconditionally. The GPU driver/NVML was unavailable, so GPU execution was not tested. `plot_run.py` wrote all/track plots but crashed on the seven-row cascade subset when log-scaling all-NaN/nonpositive binned statistics.
- Environment warning: `ice_mix/requirements.txt` contains a loose public `graphnet>=1.0.0` requirement, while the working project depends on additions in this checkout. The successor environment recipe therefore treats the supplied GraphNeT tree/export as authoritative; arbitrary public GraphNeT compatibility is unverified.
- The 2026-08-13 independent audits invalidated the earlier final-consistency claim and identified both documentation gaps and possible software defects. The accepted software findings are catalogued in `docs/known-software-findings.md` without implementation.
- Scope correction 2026-08-13: executable Python, configs, requirements, shell helpers, and Slurm launchers changed by the prior remediation commit were restored to their exact pre-remediation behavior (`43941316`), while documentation/docstrings were retained and corrected. Executable helpers created under `docs/` were removed; the approved reduced-scale YAML remains a documentation reference only.
- Post-correction verification 2026-08-13: all 33 package Python files parse and have module/public-symbol docstrings; after stripping docstrings, their executable ASTs match `43941316`; the restored runtime YAML, requirements, shell helper, and four affected launchers match that commit byte-for-byte. All 44 inherited root Slurm/shell files pass `bash -n`, all local Markdown links resolve, the four raw environment-export hashes match, and the 21 focused token-drop/EMA/LBFGS tests pass. `git diff --check` reports only two blank lines with trailing spaces inherited from the pre-remediation Python source; they were deliberately not reformatted under the strict documentation-only rule.
