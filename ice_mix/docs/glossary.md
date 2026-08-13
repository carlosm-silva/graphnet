# Glossary

| Term | Meaning in IceMix |
|---|---|
| **Base training** | Recommended full-model training with AdamW using a standard/drop/rotation data variant. |
| **Class token** | Learned vector prepended after relative-attention blocks; its final state is the event representation. |
| **DDP** | PyTorch DistributedDataParallel, one process per visible GPU in production. |
| **DOM** | Digital Optical Module. Pulse features include DOM coordinates and detector-response quantities. |
| **Drop chance** | Probability that an event is selected for token dropping. |
| **EMA** | Exponential moving average of online weights, stored and validated in FP32 by `EMAStandardModel`. Experimental here. |
| **Event graph** | PyTorch Geometric `Data` object for one event; pulses are nodes. Training currently defines no persistent edges unless optional DynEdge is enabled. |
| **GraphNeT** | External IceCube graph-learning library providing datasets, graph construction, labels, tasks, losses, model orchestration, and Lightning integration. |
| **HLC** | Hard Local Coincidence. IceMix passes `hlc_name=None`, so `IceMixNodes` does not prioritize HLC pulses while subsampling. |
| **Hydra override** | `key=value` token after `train.py` that replaces a composed config value without editing YAML. |
| **IceMix** | Transformer backbone, formerly called DeepIce in its inherited Kaggle implementation. |
| **Joint label** | Six targets: vertex $(x,y,z)$ and direction $(n_x,n_y,n_z)$. |
| **Kappa ($\kappa$)** | Predicted non-negative directional concentration used by the 3D von Mises-Fisher likelihood. |
| **LBFGS** | Limited-memory BFGS optimizer used in experimental last-block fine-tuning. Repeated closures require deterministic stochastic layers. |
| **L40S** | 48 GB NVIDIA GPU requested by current Phoenix production jobs, normally eight per node. |
| **Node-local `$TMPDIR`** | Fast temporary storage allocated to a Slurm job. Staged data disappears when the job ends and is not authoritative storage. |
| **Oneweight** | IceCube simulation-weight field appended to GraphNeT truth attributes and propagated to prediction CSVs. Current IceMix plots ignore it and therefore compute unweighted, equal-row medians and percentiles. Its normalization and the scientifically intended weighted population remain undefined here. |
| **PACE Phoenix** | Georgia Tech high-performance computing environment where production IceMix jobs run. |
| **Pulse map** | SQLite table containing per-pulse detector measurements; current name is `SRTInIcePulses`. |
| **RDE** | Relative DOM efficiency; Fourier-encoded as the sixth pulse feature. |
| **Relative attention** | Attention whose logits and values receive pairwise pulse spacetime encodings. |
| **Resilience test** | Inference with forced stochastic pulse removal at several fractions. |
| **Rotation augmentation** | Independent random rotation of each event in the detector $xy$ plane, applied consistently to pulses and truth. |
| **Scratch** | Purgeable PACE storage used here for W&B caches and transient work, not the authoritative copy of results. |
| **Token** | One retained detector pulse after GraphNeT pulse capping and optional IceMix token dropping. |
| **Token drop** | Stochastic removal of pulse tokens while preserving at least one pulse in every event. |
| **VMF** | Three-dimensional von Mises-Fisher directional distribution/loss. |
| **W&B / wandb** | Weights & Biases experiment logger; optional in config but enabled by default. |

Definitions tied directly to symbols/configs are **VERIFIED-STATIC**. Storage and workflow definitions are author-confirmed 2026-08-13; operational behavior remains **UNVERIFIED-CLUSTER** until a real job is inspected.
