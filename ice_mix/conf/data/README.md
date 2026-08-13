# Data configurations

- `standard.yaml` names the three production SQLite databases and common loader settings.
- `augmented_rotation.yaml` enables the current on-the-fly rotation of pulse coordinates and truth.
- `drop.yaml` enables stochastic pulse-token removal.
- `drop_aug_rot.yaml` combines the two augmentations.
- `augmented_rotation_OLD.yaml` points to stored fixed rotations, contains author-specific absolute paths, and is deprecated historical code.
- `split/` selects random or CSV-based event partitions.

All three neutrino flavors are production inputs. Scripts that stage only the muon- and electron-neutrino files are incomplete relative to the current configuration. **VERIFIED-STATIC; author-confirmed 2026-08-13.**

Stored fixed-rotation databases are no longer used. Rotation is applied per training batch by `RandomRotationCallback`. **VERIFIED-STATIC; author-confirmed 2026-08-13.**
