# Reduced single-GPU smoke example

This example checks that configuration, data loading, graph construction, forward/backward execution, validation, and checkpoint writing are connected. It is not a useful trained model and does not modify the production YAML.

Prerequisites are one CUDA GPU, an existing compatible `graphnet` environment, and the three small/real SQLite files expected by `standard.yaml` under `$DATA_ROOT`.

```bash
export DATA_ROOT=/path/to/prepared/sqlite
bash ice_mix/docs/examples/run_single_gpu_smoke.sh
```

The script uses one process, batch size 2, at most 64 pulses, a 64-channel
two-block model, two training and validation batches, no W&B, no loader workers,
and FP32. It also disables persistent workers, prefetching, and pinned memory to
match `num_workers=0`. **VERIFIED-STATIC.** A locally installed PyTorch
2.2.0+cu118/GraphNeT environment and staged samples constructed the reduced
258K-parameter model and entered Lightning, but the run stopped before its first
batch because no GPU was visible and `CheckSamplerCallback` unconditionally
queries the current CUDA device. This partial result is **VERIFIED-LOCAL**
(2026-08-13); end-to-end training and checkpoint writing are not.

When the author-supplied ignored fixture is present at the local staging
location, the exact one-batch application command is wrapped by:

```bash
cd "$(git rev-parse --show-toplevel)"
bash ice_mix/docs/examples/run_staged_sample_smoke.sh
```

That wrapper supplies all three fixture databases and CSV selections explicitly;
it does not alter or shadow a production config. Its Hydra composition is
**VERIFIED-LOCAL** (2026-08-13). GPU execution and checkpoint writing remain
unverified on this laptop.

There is no CPU fallback in the current training entry point because of that callback. If this script fails with `No CUDA GPUs are available`, first verify `nvidia-smi` and `torch.cuda.is_available()`; do not interpret it as a data or model-architecture failure.

Even reduced settings can fail on unusually long/invalid data or incompatible GraphNeT versions. A successful smoke test validates wiring, not physics performance or eight-GPU behavior.
