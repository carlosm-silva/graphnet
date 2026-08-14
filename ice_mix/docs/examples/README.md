# Reduced single-GPU smoke configuration

[`single_gpu_smoke.yaml`](single_gpu_smoke.yaml) is a documentation-only record
of reduced Hydra overrides: one GPU, batch size 2, at most 64 pulses, a
64-channel two-block model, two train/validation batches, no W&B, no loader
workers, and FP32. It does not modify or shadow the production config.

The documentation previously included executable wrapper scripts and a Python
artifact verifier. They were removed when the handoff scope was corrected:
software of any kind is researcher-owned, even when intended only as a helper.

To inspect the suggested overrides without training, compare the YAML with a
resolved Hydra configuration:

```bash
sed -n '1,200p' ice_mix/docs/examples/single_gpu_smoke.yaml
python ice_mix/verify_config.py --cfg job --resolve
```

If the researcher elects to run a reduced smoke test, he should translate the
documented values into reviewed Hydra overrides and retain the resolved output.
This laptop could construct the reduced 258K-parameter model and enter
Lightning, but training stopped before the first batch because no GPU was
visible and `CheckSamplerCallback` unconditionally queries a CUDA device
(**VERIFIED-LOCAL**, 2026-08-13). End-to-end training and checkpoint writing
remain unverified.

A successful reduced run checks wiring, not physics quality, production data
provenance, pulse-selection reproducibility, resume correctness, or eight-GPU
DDP behavior.
