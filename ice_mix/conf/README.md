# Hydra configuration

Hydra composes one file from each configuration group before `train.py` runs:

```text
config.yaml
├── attention/baseline.yaml
├── data/standard.yaml (or a derived data variant)
└── data/split/random.yaml (or csv.yaml)
```

Command-line overrides in the Slurm launchers are applied last. For example, `data=drop_aug_rot project_name=IceMix-Drop-Augmented-Rotation` selects token dropping plus on-the-fly azimuthal rotation. **VERIFIED-STATIC.**

See [configuration and hyperparameters](../docs/configuration.md). Personal paths and secrets must be supplied outside tracked configuration; never commit `.env` files.
