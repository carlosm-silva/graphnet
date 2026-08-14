# IceMix

IceMix is an IceCube event-reconstruction package built around a transformer backbone. It consumes GraphNeT event graphs whose nodes are detector pulses and predicts the interaction vertex, neutrino direction, and a directional concentration parameter. Production training is designed for the Georgia Tech PACE Phoenix cluster; inherited launchers request up to one node with eight NVIDIA L40S GPUs but are not submit-ready successor templates.

> **Development provenance and human contact**
>
> This project was developed with substantial assistance from OpenAI Codex and ChatGPT under the original author's scientific direction. The author increasingly works by specifying, reviewing, and testing AI-assisted changes rather than typing most implementation code personally. This does not make the code ownerless: if intent, history, or physics choices are unclear, send the original author a Slack message. He is willing to help the next student.

## Start here

1. Read [the documentation overview](docs/index.md).
2. Read the [known software findings](docs/known-software-findings.md) before trusting a launcher or comparison.
3. Follow the [tutorials in order](docs/tutorials/README.md), including data and split orientation before training.
4. Read [configuration and scale](docs/configuration.md) before changing a Hydra override.
5. Rebuild the [exported GraphNeT environment](docs/graphnet_env/README.md).

> **Scope and operational warning**
>
> The handoff documents inherited software; it does not repair it. Several
> launchers and evaluation paths have audit-identified correctness or
> reproducibility hazards. A researcher must review and implement any fix.

Every non-obvious statement in the hand-written documentation is marked as:

- **VERIFIED-STATIC** — established from source, configs, or CLI definitions;
- **VERIFIED-LOCAL** — exercised locally, with the exact check named; or
- **UNVERIFIED-CLUSTER** — requires PACE, production data, or multi-GPU confirmation.

## Package map

| Location | Role |
|---|---|
| `conf/` | Hydra training, data, split, and architecture configuration |
| `src/models/` | IceMix transformer layers and GraphNeT training wrappers |
| `src/` | Metrics, callbacks, split utilities, and a historical custom trainer |
| `train.py` | Canonical training entry point |
| `predict.py` | Run discovery and checkpoint inference |
| `run_*.sbatch` | PACE production, resume, evaluation, and experimental jobs |
| `plot_*.py`, `generate_*.py` | Prediction postprocessing and comparisons |
| `docs/` | Architecture, operations, tutorials, and the handoff inventory |
| `outputs/` | Small checked-in historical metadata; large outputs are ignored |

The folder layout is historical and intentionally unchanged during this handoff.

## Recommended versus experimental workflows

Base training with the standard, token-drop, rotation, or combined data variant is the recommended scientific family. No inherited base launcher is currently documented as submit-ready without researcher review. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

Distributed LBFGS and AdamW+EMA last-block fine-tuning are newer research workflows. They have shown some success, but the departing author does not consider them the default or fully established path. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

## Scale warning

The baseline batch size of 256, 12-block/384-channel transformer, 256-pulse cap, DDP setup, and data-loader tuning were selected for an eight-L40S production job. Do not copy a production launch to a laptop or small workstation. Use the [single-GPU smoke configuration](docs/examples/README.md) only as a wiring reference before spending a cluster allocation.

## Durable handoff state

- [MEMORY.md](MEMORY.md) is stable technical knowledge.
- [TODO.md](TODO.md) is the exact resume point.
- [QUESTIONS.md](QUESTIONS.md) contains author questions and the data-staging checklist.
