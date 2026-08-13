# Configuration and hyperparameters

Hydra starts from `conf/config.yaml`, composes `attention/baseline.yaml`, `data/standard.yaml`, and `data/split/random.yaml`, then applies launcher overrides. **VERIFIED-STATIC.**

Inspect the exact resolved configuration for a run in its `.hydra/config.yaml`; never infer it only from the base YAML after overrides have been applied.

## Configuration groups

### Experiment and hardware

| Key | Default | Meaning and scale |
|---|---:|---|
| `project_name` | `IceMix` | W&B project fallback and run-name prefix. |
| `run_name` | project/time/Slurm ID | Output directory identity. |
| `seed` | 42 | Lightning/global seed; token-drop batches derive deterministic seeds from it. |
| `precision` | `16-mixed` | Cluster-oriented automatic mixed precision. LBFGS forces `32-true`. |
| `num_workers` | 3 | Loader workers per process, not per node. Eight ranks can therefore create 24 workers. |
| `wandb` | true | Adds W&B alongside the always-enabled CSV logger. |

### Model dimensions

| Key | Default | Meaning |
|---|---:|---|
| `hidden_dim` | 384 | Token/event latent width and backbone output. Must be divisible by `head_size`. |
| `seq_length` | 128 | Fourier basis width, despite its historical name; not the pulse count. |
| `depth` | 12 | Ordinary transformer blocks after class-token insertion. |
| `n_rel` | 4 | Both the number constructed and the number using relative bias in current wiring. |
| `head_size` | 32 | Channels per attention head; baseline has 12 heads. |
| `include_dynedge` | false | Adds GraphNeT DynEdge pulse features when enabled. |
| `maha_encoder` | false | Uses Euclidean-sign spacetime alternative instead of signed spacetime interval when true. |

### Regularization and augmentation

- `dropout`, `attn_drop`, and `proj_drop` affect MLP, attention weights, and projection paths.
- `drop_path_rate` increases linearly with block depth.
- `data.token_drop` is the per-token probability within selected events; `data.drop_chance` is the event-selection probability.
- `data.augment_rotation` rotates each event independently in the $xy$ plane immediately before a training batch.

The token-drop code always restores one token if an event would otherwise become empty. Rotation changes pulse $x/y$, target position $x/y$, direction $x/y$, azimuth, and scalar position fields in place. **VERIFIED-STATIC.**

On-the-fly rotation is the current workflow. The stored augmented-database paths and event-ID expansion code are deprecated compatibility artifacts. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

Although `features` currently contains seven names and the inference graph may carry edges, baseline IceMix uses only pulse feature indices 0–5 and ignores graph edges. These are backward-compatibility artifacts. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

### Optimization

- Default AdamW: learning rate `6.25e-5`, epsilon `1e-5`, weight decay `0.01`.
- Default scheduler: reduce on validation-loss plateau, patience 6, factor 0.5.
- Cosine scheduling is used by recent AdamW+EMA fine-tuning launches.
- LBFGS uses strong-Wolfe search and, by default, clears curvature history before each new mini-batch. It disables mixed precision and the scheduler.
- EMA defaults to decay 0.999 and is supported only for AdamW.

These behaviors are **VERIFIED-STATIC**. Their convergence quality at production scale is **UNVERIFIED-CLUSTER**.

## Data and split keys

The standard configuration resolves three database paths from `$DATA_ROOT`, reads `SRTInIcePulses` and `truth`, caps at 256 pulses, and uses batch size 256. Random mode partitions each database independently at 80/10/10 after a deterministic shuffle. **VERIFIED-STATIC.**

CSV mode requires exactly one train and validation CSV per database. The checked-in CSV config has only two of each for three database paths and therefore fails fast. **VERIFIED-STATIC.** A complete staged three-flavor fixture was parsed and cross-checked against its databases (**VERIFIED-LOCAL**, 2026-08-13).

## Numeric encoder constants

`pos_time_multiplier`, `charge_rde_multiplier`, `spacetime_distance_scale`, clipping bounds, distance multiplier, and `n_freq` change the numeric arguments to sinusoidal features. They are learned-feature conditioning choices, not physical calibration constants. Their exact formulas, ordered features, inherited GraphNeT normalization, output transformation, and joint loss are documented in [Numeric feature, encoder, and loss contract](numeric-contract.md). Their derivation and tuning history are **UNVERIFIED-CLUSTER** pending author input; do not reinterpret `18.0` as a measured property of ice.

## Production versus a small GPU

| Setting | Production baseline | Suggested wiring smoke test |
|---|---:|---:|
| GPUs | 8 × L40S (48 GB each) | 1 GPU |
| `batch_size` | 256 per loader/rank as configured | 2 |
| `max_pulses` | 256 | 64 |
| `hidden_dim` | 384 | 64 |
| `seq_length` | 128 | 32 |
| `depth` / `n_rel` | 12 / 4 | 2 / 2 |
| precision | `16-mixed` | `32-true` for maximum compatibility |
| workers | 3 per rank | 0 |
| batches | full epoch | 2 train + 2 validation |

The production column is **VERIFIED-STATIC**. The reduced values are a
conservative documentation example. In the disposable local environment, Hydra
composition, data-loader/model construction, and Lightning startup were
**VERIFIED-LOCAL** on 2026-08-13; the run could not enter its first batch because
no CUDA device was visible and a callback requires CUDA. It is not a
physics-quality training configuration or an eight-GPU check.

## Stale configuration guide

`config_parameters_guide.md` predates parts of the current implementation. It mentions a `gpus` key removed from the current config, explicit selections no longer present in `standard.yaml`, and multistage settings that `train.py` does not consume. Treat the current YAML and source as authoritative. **VERIFIED-STATIC.**
