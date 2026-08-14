# Documentation source

Start with [`index.md`](index.md). This directory is plain Markdown rather than a separately configured MkDocs site; no new documentation build machinery was introduced during the handoff.

- `glossary.md` defines IceMix-specific language.
- `graphnet-boundary.md` documents only the interfaces IceMix consumes.
- `configuration.md` explains Hydra, model/data choices, and production scale.
- `numeric-contract.md` defines ordered features, inherited normalization, encoder equations, outputs, and loss scales.
- `pace-phoenix.md` is the operational runbook.
- `job-scripts.md` covers every Slurm and shell launcher.
- `handoff-status.md` gathers remaining cluster and sample verification work.
- `known-software-findings.md` flags likely defects for researcher action; no
  software repair is performed by the documentation handoff.
- `audit-remediation.md` records the audit disposition and scope correction.
- `tutorials/` contains six start-to-finish workflows.
- `examples/` contains a nonproduction single-GPU configuration reference.

`_local_sample/` is gitignored staging used only for verification; documentation
must never present it as a production data location. Its durable identity and
retrieval limitations are recorded in the
[local verification artifact manifest](local-sample-manifest.md).
