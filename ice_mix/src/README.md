# IceMix implementation modules

`src/` is imported by root-level entry points when they are launched from the repository root.

- `models/` contains the transformer, its layers, and specialized GraphNeT `StandardModel` wrappers.
- `metrics_logging.py` exposes per-event reconstruction metrics to Lightning loggers.
- `utils.py` defines GraphNeT feature/truth lists, dataset splitting, pulse rotation, token-drop seeding, and DDP sampler diagnostics.
- `trainer.py` contains older custom OneCycle/SWA loops. No caller exists in the current tree, and the author confirmed it is abandoned/historical; it is not recommended.

This directory is not a separately installable package and has no root `__init__.py`; entry points rely on `ice_mix` being on Python's import path. **VERIFIED-STATIC.**
