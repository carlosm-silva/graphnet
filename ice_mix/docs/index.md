# IceMix handoff overview

This documentation is for a physics graduate student taking ownership of IceMix without the original author beside them. It explains the scientific and computational flow, the boundary with GraphNeT, and how the package is operated on Georgia Tech PACE Phoenix.

> **Who wrote the implementation?**
>
> The original author made the scientific and workflow decisions with substantial coding assistance from OpenAI Codex and ChatGPT. He now usually specifies, reviews, and tests AI-assisted implementation rather than typing most code himself. For history or intent that the repository cannot answer, send him a Slack message; he is willing to help.

## What the model reconstructs

One IceCube event is represented by a variable-size set of pulses. Each pulse contributes detector position, time, charge, relative DOM efficiency, and related detector features. IceMix maps the pulse set to one learned event representation and predicts:

$$
(x, y, z, n_x, n_y, n_z, \kappa),
$$

where $(x,y,z)$ is the interaction vertex, $\mathbf{n}$ is the reconstructed unit direction, and $\kappa$ is the von Mises-Fisher directional concentration. **VERIFIED-STATIC.**

The target contains the same position and direction but no $\kappa$. Position loss is Euclidean distance and direction loss is the GraphNeT 3D von Mises-Fisher loss. The joint objective is

$$
\mathcal{L} = \alpha\mathcal{L}_{\mathrm{position}} + \mathcal{L}_{\mathrm{direction}},
$$

with default $\alpha=0.026$. **VERIFIED-STATIC.**

## Architecture and data flow

```text
PACE sbatch launcher
  └─ project-storage SQLite → node-local $TMPDIR
      └─ Hydra config composition
          └─ GraphNeT SQLiteDataset + GraphNeTDataModulecustom
              └─ IceCube86 normalization + IceMixNodes pulse cap
                  └─ PyG Data: x[all pulses, features], batch[all pulses]
                      └─ optional token removal
                          └─ padded sequence + Fourier features
                              └─ spacetime-relative transformer blocks
                                  └─ learned class token + transformer blocks
                                      └─ event embedding [events, 384]
                                          └─ GraphNeT joint task
                                              └─ prediction [events, 7]
                                                  ├─ Lightning loss/metrics
                                                  └─ best/last checkpoints
```

The baseline uses a 384-channel event representation, four relative-attention blocks, twelve ordinary blocks, 32 channels per head, and at most 256 pulses per event. **VERIFIED-STATIC.**

## End-to-end execution

1. Choose a base data variant: standard, token drop, rotation, or both.
2. Submit the supported `run_training.sbatch` launcher with explicit data-variant and project-name environment variables. Do not execute a production launcher on a laptop.
3. The launcher stages databases, activates the `graphnet` conda environment, rejects unhealthy GPUs, and invokes `torchrun`.
4. `train.py` constructs GraphNeT data and task objects around the IceMix backbone and calls Lightning through `StandardModel.fit`.
5. The run writes Hydra configuration, CSV/W&B logs, and `best-*` plus `last.ckpt` checkpoints.
6. The same launcher restores an explicitly named `last.ckpt` and W&B run after walltime or interruption.
7. `predict.py` produces `predictions/results.csv` plus an evaluation manifest; plotting scripts validate comparison populations and record their unweighted policy.

Steps 1–7 are **VERIFIED-STATIC** from the launchers and source. Actual queue behavior, full-scale runtime, and successful multi-GPU execution are **UNVERIFIED-CLUSTER**.

## Read next

- [Glossary](glossary.md)
- [GraphNeT interface boundary](graphnet-boundary.md)
- [Configuration and hyperparameters](configuration.md)
- [Numeric feature, encoder, and loss contract](numeric-contract.md)
- [Reproducible exported environment](graphnet_env/README.md)
- [PACE Phoenix operations](pace-phoenix.md)
- [Slurm and shell script catalog](job-scripts.md)
- [Verification and handoff status](handoff-status.md)
- [Independent-audit remediation](audit-remediation.md)
- [Local verification artifact manifest](local-sample-manifest.md)
- [Tutorials](tutorials/README.md)
- [Single-GPU smoke example](examples/README.md)

## Verification policy

The markers used throughout the handoff have intentionally narrow meanings:

- **VERIFIED-STATIC:** code, signatures, configs, CLI definitions, or checked-in metadata establish the claim.
- **VERIFIED-LOCAL:** the documented check actually ran on the documentation workstation; the command or input is named nearby.
- **UNVERIFIED-CLUSTER:** PACE, real data, full-scale resources, or the author's operational knowledge is required. The text states how to confirm it.

No runtime, queue wait, metric value, loss curve, or sample log is invented.
