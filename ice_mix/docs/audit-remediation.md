# Independent-audit remediation

`AUDIT.md` is a snapshot of the pre-remediation working tree. This page records
the disposition of its 13 findings; it does not rewrite the auditor's evidence.

| Finding | Disposition after remediation |
|---:|---|
| 1 | Resolved: all spatial claims were withdrawn. The historical checkerboard path is documented in code and Tutorial 5 as seeded complementary random halves of padded sequence positions. |
| 2 | Resolved statically and with staged fixtures: training and evaluation share `get_configured_splits`; CSV test requests fail when undefined; prediction and robustness outputs persist exact per-database event selections. |
| 3 | Resolved as a reproducible specification: `docs/graphnet_env/` preserves the raw export and supplies a prefix-free recipe pinned to the exact checkout, PyTorch, PyG, and Lightning stack. Phoenix recreation remains **UNVERIFIED-CLUSTER**. |
| 4 | Resolved: no personal reference path is consulted. The reference CSV is explicit, optional, and recorded in plot metadata. The successor is told to search inherited PACE files for the historical basename. |
| 5 | Prepared but not complete until version control: the documentation/source changes and tracked hash manifest are ready; the large ignored bundle is identified by checksums and inherited run identity. It must remain outside ordinary Git. A commit still requires the author's approval. |
| 6 | Resolved: `numeric-contract.md` is the authoritative feature-order, normalization, encoder, output, and loss contract. Unknown scientific provenance is labeled rather than invented. |
| 7 | Resolved for scientific comparisons: fine-tune and nu-tau scripts require explicit pairs by default, matching evaluation manifests, and identical event multisets. Per-run baselines are explicit by default; master/reference comparisons validate populations; resilience uses a matched 0% table; complementary halves validate and one-to-one merge full identities. Legacy discovery is explicit opt-in. |
| 8 | Resolved statically: `run_training.sbatch` is the single supported base launcher, with checkout/environment/Hydra/GPU preflight, fail-fast three-flavor staging, dynamic rank count, and explicit resume identity. Phoenix execution remains **UNVERIFIED-CLUSTER**. |
| 9 | Resolved: on-the-fly rotation defaults to the persisted run seed; robustness CLIs expose/persist a seed; fractional event samples are seeded; fixed-batch token masks are deterministic and nested across thresholds. |
| 10 | The false weighted claim is resolved without fabricating a convention: current plots are explicitly unweighted and write that policy to `plot_manifest.json`. A weighted mode is intentionally absent until the data owner defines the target population and `oneweight` normalization. |
| 11 | Resolved according to the author's handoff preference: the ledgers no longer claim a completed final audit and distinguish completed implementation from first-run cluster verification. Author-answered questions remain answered; durable pages route successor-owned PACE checks to current PACE guidance, the author, or Jiyuan rather than pretending those external facts are repository facts. |
| 12 | Resolved: all public symbols have docstrings, core stateful/private change surfaces are documented, and abandoned probes cannot perform PACE I/O during pytest collection. The maintained 21-test command and expected scope are explicit. |
| 13 | Resolved: local evidence commands start from the repository root, use `python3` for dependency-light checks, and provide an exact application smoke wrapper. |

The remediation was checked locally on 2026-08-13 with Python compilation,
21 maintained tests, collection of the full `ice_mix` test tree, shell syntax,
Hydra composition, staged split/manifest checks, public-docstring coverage, and
Markdown-link validation. These checks do not promote any PACE/DDP claim beyond
**UNVERIFIED-CLUSTER**.
