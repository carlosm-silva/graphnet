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
- Documented the exact feature normalization/encoder/loss contract and the unweighted statistics policy.
- Converted the author-supplied PACE export into a prefix-free pinned bootstrap tied to the exact GraphNeT commit.
- Added a tracked local-artifact hash/size manifest without versioning the large ignored checkpoint or event fixtures.
- Accepted the author's strict documentation-only scope correction: suspected bugs are researcher-owned findings and no software fix is part of this handoff.
- Restored executable Python, configs, requirements, shell helpers, and Slurm launchers to the exact behavior of pre-remediation commit `43941316`, retaining permitted docstrings.
- Removed executable smoke/verification helpers created under `docs/`; retained the approved reduced-scale YAML as a documentation reference only.
- Added `docs/known-software-findings.md` and revised tutorials/operations pages so unresolved split, stochasticity, comparison, plotting, resume, and launcher behavior is flagged rather than presented as repaired.
- Preserved the independent auditor's current `AUDIT.md` and `AUDIT_PROGRESS.md` changes without editing or staging them.
- Verified that all 33 Python files parse, have module/public-symbol docstrings, and retain the executable AST of `43941316`; all explicitly restored runtime files match that commit byte-for-byte.
- Re-ran 21 focused token-drop/EMA/LBFGS tests, shell syntax for all 44 inherited root scripts, Markdown link checks, and raw environment-export integrity checks successfully.
- Committed the corrective documentation-only scope restoration without staging the auditor-authored working-tree files.
- Expanded the `Mlp` and `Attention_rel` class docstrings with their transformer role and tensor contracts, without changing behavior.
- Recorded that audit artifacts are outside documentation ownership and that documented PACE/data/group contact routes are completed handoff guidance rather than blockers.
- Added prominent LLM-authorship and scientific-skepticism warnings to the package README and documentation overview.
- Documented the successor's inherited read/write access to the author's PACE project tree and private `ice_mix/.env`, without copying secret values.
- Reconciled the loose public GraphNeT declaration with the required checkout-specific tree and inventoried IceMix's private `StandardModel` contracts.
- Published the exact GPU-required reduced smoke command and corrected the finiteness check, callback identity, pytest prerequisite, W&B behavior, Slurm output lookup, and repeated runtime-variable explanations.
- Corrected the `DropPath` default documentation and recorded two researcher-owned software observations without changing behavior.
- Verified Python parsing, docstring-stripped `layers.py` behavior, unchanged effective requirements, local Markdown links, and diff whitespace after the final audit documentation pass.

## In progress

- None.

## Queued

- None. Future work is researcher-owned software review and the first-run PACE confirmations listed in `docs/handoff-status.md`.

## Blocked

- None.

## Deferred-data

- None.
