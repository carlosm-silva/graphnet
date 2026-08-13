# Reproducing the PACE `graphnet` environment

This directory records the exact environment exported from the author's PACE
Phoenix account and turns it into a portable successor bootstrap. The raw
exports are evidence; the `pace-*` files are the reusable specification.

## What was exported

| File | Purpose |
|---|---|
| `environment.yml` | Full Conda export, including pip packages and the author's nonportable environment prefix. |
| `environment-from-history.yml` | Short list of packages the author explicitly requested from Conda. |
| `environment-explicit.txt` | Exact Linux-64 Conda artifact URLs. Useful for forensic reproduction, but tied to package availability and platform. |
| `requirements-pip.txt` | Exact pip snapshot. It proves that GraphNeT was installed from commit `4394131647b4a581e7d4923361b2814ab9e03ff5`. Several Conda-owned packages appear as nonportable `file://` build paths, so do not install this file directly. |
| `EXPORT_SHA256SUMS` | Integrity hashes for the four author-supplied exports. |

These files were supplied by the author on 2026-08-13. Their content is
**author-confirmed** and has been inspected **VERIFIED-STATIC**. Recreating and
running the environment on Phoenix remains **UNVERIFIED-CLUSTER** until the
successor performs the checks below.

## Portable bootstrap on PACE Phoenix

Run from the GraphNeT checkout root. The checkout itself is the authoritative
custom GraphNeT package; do not additionally install `graphnet>=1.0.0` from
PyPI.

```bash
module load anaconda3/2022.05.0.1

git rev-parse HEAD
# Required: 4394131647b4a581e7d4923361b2814ab9e03ff5

conda env create -f ice_mix/docs/graphnet_env/pace-environment.yml
conda activate graphnet
python -m pip install -r ice_mix/docs/graphnet_env/pace-pip-requirements.txt
python -m pip install --no-build-isolation --no-deps --editable .
```

`pace-environment.yml` deliberately has no `prefix:` key. It preserves the
author's Python 3.8.20, Conda CUDA Toolkit 11.5.2, compiler/runtime, and GPU
diagnostic packages. `pace-pip-requirements.txt` preserves the core Python,
PyTorch 2.2.0+cu118, PyG 2.6.1, Lightning 2.4.0, Hydra, analysis, and logging
versions needed by IceMix. PyTorch's pip CUDA 11.8 runtime packages coexist in
the original export with Conda's CUDA Toolkit 11.5; this recipe preserves that
observed arrangement rather than guessing at a cleanup.

The final editable install uses `--no-deps` deliberately: all direct GraphNeT
dependencies are pinned in the preceding requirements file, while allowing pip
to resolve them again could silently replace the exported versions. If pip
reports a missing dependency, compare it with `requirements-pip.txt`, add the
exported version to the portable file, and record the change rather than
installing an unpinned latest release.

The `git rev-parse` guard matters: this IceMix checkout relies on custom
`IceMixNodes` and `GraphNeTDataModulecustom` interfaces. If the commit differs,
stop and locate the author-supplied checkout before installing.

## Preflight before requesting an L40S node

```bash
python -m pip check
python - <<'PY'
import graphnet
import hydra
import pytorch_lightning
import torch
from graphnet.data.datamodule import GraphNeTDataModulecustom
from graphnet.models.graphs.nodes import IceMixNodes

print("torch", torch.__version__)
print("CUDA runtime", torch.version.cuda)
print("GraphNeT boundary imports passed")
PY

DATA_ROOT=/path/to/the/three/prepared/databases \
    python ice_mix/verify_config.py --cfg job --resolve
```

Expected version landmarks are `torch 2.2.0+cu118`, CUDA runtime `11.8`,
Lightning `2.4.0`, and PyG `2.6.1`. These commands are **VERIFIED-STATIC**; save
their output from the first successor run to promote the Phoenix environment to
**VERIFIED-CLUSTER**.

## If exact package recreation fails

Do not silently upgrade PyTorch, PyG, or GraphNeT. First compare the failure to
the raw explicit export and ask the author on Slack. Conda artifact retirement,
PACE module changes, or compiler compatibility may require a controlled update;
record every deviation and rerun the import, configuration, smoke-training, and
checkpoint-loading checks.
