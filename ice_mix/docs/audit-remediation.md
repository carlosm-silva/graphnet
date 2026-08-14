# Audit disposition and scope correction

Two independent audits identified documentation gaps and likely software
defects. An earlier handoff pass implemented several proposed software changes,
but that exceeded the documentation-only scope. Those executable changes were
subsequently removed: pre-existing Python behavior, runtime YAML,
`requirements.txt`, shell helpers, and Slurm launchers now match commit
`4394131647b4a581e7d4923361b2814ab9e03ff5`. Docstrings and documentation remain.

The operative rule is now explicit:

> A documentation maintainer may explain or flag software behavior, but may not
> repair, refactor, or otherwise change it. Likely bugs belong to the researcher.

The current auditor reports are [`AUDIT.md`](../AUDIT.md) and
[`AUDIT_PROGRESS.md`](../AUDIT_PROGRESS.md). They are review evidence, not an
implementation plan. Their software findings have been consolidated in
[Known software findings — researcher action only](known-software-findings.md).

## Documentation work retained

- Module, class, method, and function docstrings that describe inherited
  behavior.
- Architecture, GraphNeT-boundary, numerical-contract, configuration, PACE,
  environment, glossary, script-catalog, and tutorial pages.
- Folder READMEs, durable ledgers, author answers, staged-artifact hashes, and
  local verification records.
- Clear withdrawal of the false spatial interpretation of the historically
  named checkerboard study.
- Accurate statements that current plots are unweighted and that production
  data provenance/units remain externally unverified.

## Proposals deliberately not implemented

The documentation does not centralize split reconstruction, add manifests,
seed stochastic evaluation, alter checkpoint selection, change resume
semantics, validate comparison event populations, repair launchers, alter
DropPath, or change failure exit codes. These are software decisions listed for
researcher review in the known-findings page.

## Verification meaning after rollback

Earlier local executions demonstrate that a particular checkpoint and sample
could load and produce finite output in the temporary laptop environment. They
do not verify that automatic split selection, pulse capping, comparisons,
resume, or Slurm launchers are scientifically reproducible. Those limitations
must travel with any cited `VERIFIED-LOCAL` result.
