# IceMix documentation task ledger

## Done

- Surveyed the complete `ice_mix` tree and classified entry points, configs, Python modules, job scripts, tests, and historical outputs.
- Traced the base PACE submission-to-checkpoint path and the GraphNeT boundary.
- Recorded the complete data-staging request and author decisions.
- Added `docs/_local_sample/` to `.gitignore`.
- Created the durable handoff ledgers.
- Added README files for the package, configuration/source folders, model folder, scripts, and historical outputs.
- Wrote the overview, glossary, GraphNeT boundary, configuration, PACE operations, and complete Slurm/shell catalog.
- Wrote all six approved tutorials and the docs-only single-GPU smoke example.
- Added module docstrings to all 33 Python files and docstrings to every public class, function, and method.
- Expanded core model/training docstrings with tensor shapes, returns, side effects, and boundary assumptions.
- Completed static consistency audits: Python compilation/docstring coverage, behavior-preserving AST comparison, Bash syntax, local Markdown links, path/verification-marker scan, and whitespace checks.
- Incorporated all author answers, including healthy-run criteria, the bad-node workaround, and topology-stratified comparison guidance.
- Validated all staged D001–D006 artifacts and promoted supported schema, rotation, split, checkpoint, prediction, metrics, and log claims to VERIFIED-LOCAL.
- Built a disposable local GraphNeT/PyTorch-CUDA environment after author approval; verified checkpoint loading, a finite CPU forward, 21 focused tests, Hydra composition, prediction/resilience discovery, and complete nine-event CPU inference with artifacts.
- Tested the docs smoke and plotting workflows and documented the CUDA-only sampler callback, PyG version drift, DATA_ROOT ordering, and sparse-cascade plotting failure.
- Reviewed the independent audit and classified accepted findings versus author-directed handoff expectations.
- Centralized current/legacy random and CSV split reconstruction across training, prediction, resilience, complementary-half evaluation, and checkpoint interpolation.
- Made evaluation preserve explicit GraphNeT selections, fail on a requested but undefined CSV test split, and write per-database event-selection provenance manifests.
- Withdrew the false spatial-checkerboard interpretation in code while retaining historical filenames for compatibility; added explicit reproducibility seeds for rotation and robustness studies.
- Required explicit fine-tune/nu-tau comparison pairs, matching evaluation manifests, and identical event populations; removed implicit personal reference-CSV lookup.
- Documented the exact feature normalization/encoder/loss contract and the unweighted statistics policy; added plot input/weighting manifests.
- Converted the author-supplied PACE export into a prefix-free pinned bootstrap tied to the exact GraphNeT commit.
- Repaired `run_training.sbatch` as the supported fail-fast, three-flavor, dynamically sized base launcher with explicit resume identity.
- Corrected tutorial commands, added an exact staged-sample application smoke wrapper, and made abandoned test probes safe to collect without PACE I/O.
- Added a tracked local-artifact hash/size manifest without versioning the large ignored checkpoint or event fixtures.
- Repaired prediction, token-removal, and complementary-half Slurm launchers to share the supported fail-fast staging/preflight path.
- Removed remaining silent plotting baselines: per-run comparison is explicit, cross-project and perturbation plots validate provenance/event identities, and resilience creates a matched 0% baseline.
- Completed the post-remediation consistency pass: 21 maintained tests pass; the real checkpoint loads strictly and produces a finite CPU forward; fixture/hash checks, full pytest collection, compilation, shell syntax, public-docstring coverage, links, environment-export hashes, and whitespace checks pass.
- Prepared the complete audited handoff as a reviewed version-control change; the ignored local fixture remains outside Git and is represented by its tracked verifier/manifest.

## In progress

- None.

## Queued

- None. Future work is limited to the first-run PACE confirmations listed in
  `docs/handoff-status.md`; those are successor verification, not missing
  documentation implementation.

## Blocked

- None.

## Deferred-data

- None.
