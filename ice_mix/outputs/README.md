# Historical output metadata

This directory contains small checked-in Hydra and GraphNeT model-configuration artifacts from earlier runs. It does not contain the corresponding checkpoints, full CSV outputs, W&B state, or logs; those are ignored or remain on PACE.

Treat these files as provenance, not runnable canonical configuration. Some record older two-flavor datasets, absolute author paths, a previous Hydra directory scheme, `KNNGraph`, and config keys that no longer match the current source. **VERIFIED-STATIC.**

New runs use `ice_mix/outputs/<project>_<timestamp>_job-<id>/` by default. Production output locations and retention remain **UNVERIFIED-CLUSTER** until confirmed from an actual PACE run.
