# Verification and handoff status

This page gathers evidence the documentation workstation cannot supply. It is
updated as staged samples are checked; missing inputs never justify guessed
schemas, units, metrics, runtimes, or log excerpts.

## Remaining UNVERIFIED-CLUSTER claims

- Confirm the current Phoenix login/onboarding procedure, allocation limits,
  queue behavior, quotas, purge policy, and exact successor storage paths from
  current PACE guidance or Jiyuan on Slack.
- Recreate the checked-in [`graphnet` environment recipe](graphnet_env/README.md)
  on Phoenix and save the import/preflight output. The exact author export is
  now available, but the portable reconstruction remains unverified on-cluster.
- Submit a low-cost cluster sanity job before production and confirm GPU
  discovery, `$TMPDIR` staging, DDP startup, logging, checkpointing, and resume
  on current L40S nodes.
- Confirm the current acceptable-node allow-list with Jiyuan. The author
  reports one bad L40S node and a Slurm bug that can ignore `--exclude`.
- Measure current runtime and memory use. The author reports roughly one epoch
  per hour for regular eight-L40S training, but this remains cluster-dependent.
- Confirm full-scale convergence and scientific value of encoder multipliers,
  fine-tuning variants, and checkpoint interpolation.
- Confirm authoritative database provenance and physical units with Jiyuan;
  a sample can establish structure but not provenance by itself.

## Author-confirmed health criteria

Monitor W&B training/validation curves and checkpoint creation. Any NaN loss
is a failed state; the author expects the earlier source to have been fixed by
the 3D-vMF change. Use comparison plots, and inspect tracks and cascades
separately because their mixture can produce Simpson's-paradox behavior in an
aggregate curve. These statements were author-confirmed 2026-08-13.

## Staged-artifact status

All D001–D006 artifact families described in `QUESTIONS.md` arrived. The bundle
is intentionally ignored; its core byte sizes and hashes are recorded in the
[tracked artifact manifest](local-sample-manifest.md). On
2026-08-13, `verify_sample.py` passed all six integrity groups. Independent
read-only checks confirmed schemas/counts/indexes, split membership, legacy
rotation, checkpoint archive/metadata keys, finite metrics, prediction inputs,
unit direction outputs, and successful/failed log landmarks.

The checkpoint is a valid PyTorch zip container with state, optimizer,
scheduler, callback, epoch, and global-step metadata keys. In the temporary
local environment, `verify_graphnet.py` constructed events from all three
flavors, confirmed 256-node truncation, loaded the checkpoint strictly, and
produced a finite CPU output with shape `(1, 7)` (**VERIFIED-LOCAL**,
2026-08-13). True Lightning resume and GPU inference remain unexecuted.

## Local environment limitation

After explicit author approval, a disposable local Python 3.11 environment was
created with PyTorch 2.2.0+cu118, matching PyG wheels, and this checkout as an
editable install. It is local-only and not a production environment. The CUDA
runtime imported, but the driver/NVML was unavailable and no GPU was visible.

Twenty-one token-drop/EMA/LBFGS tests passed. A reduced training command built
the data loaders and 258K-parameter model, initialized one-process CPU DDP, and
then failed before its first batch because `CheckSamplerCallback` calls
`torch.cuda.current_device()` unconditionally. Thus the documented training
smoke is GPU-required in the current implementation and has not completed
locally.

The former loose `ice_mix/requirements.txt` conflict has been removed: it now
includes the pinned PACE application requirements and leaves GraphNeT to the
documented exact editable checkout. During earlier local reconstruction,
unconstrained `torch-geometric` selected 2.8, whose
`KNNGraph` path was incompatible with the PyTorch-2.2 extension index;
`torch-geometric==2.5.3` completed staged CPU inference. Treat the recovered
PACE environment/checkout as authoritative.
