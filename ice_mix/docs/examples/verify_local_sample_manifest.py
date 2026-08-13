#!/usr/bin/env python3
"""Verify core ignored IceMix fixture artifacts against tracked SHA-256 values."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
    "standard_sqlite/numu.db": (151552, "588ebe9949d4b2af74e3abf6030926c5df4d33950bce54ad407e52cf2fbccbc6"),
    "standard_sqlite/nue.db": (180224, "cc5f256ad844ae98c416064849f9b84b27ed05975879202ae1391138bf33fc7c"),
    "standard_sqlite/nutau.db": (147456, "af50f556f40da96353335ac89293eb823348d9947f55b3a7f4b90e265be284bf"),
    "augmented_sqlite/numu_augmented.db": (24576, "0dc5e79685c6eb0a7abaa38aa666d8eb1653e4e51d8839d131bca865ad158b3e"),
    "base_run/.hydra/config.yaml": (1662, "fa77fb9c90755b6fca971d6d427b43d761c7c873232ef1401a9c9c5b3e5167fc"),
    "base_run/checkpoints/best-epoch=18-val_loss=1.4589.ckpt": (353310946, "177a93cc53e73a8b9b1193ea3c06a1293891c6941aadcdc1bb2b8f9c6b5b1746"),
    "base_run/predictions/state_dict.pth": (117758703, "4616be18c664bbb3be66dc96f558851ac941b1c1cf206487b7ed246c46cb2fcb"),
    "base_run/predictions/results.csv": (2764, "a96c52c7dc7d220188b7f2a713b8784c15aa100f147c7b92b27dc178336253f2"),
    "prediction/results.csv": (14852, "a05aafa26f78cbd2cafba2be379425dfebbd6b34d3190185ed19fad3b2262403"),
    "slurm_reports/successful-base-training.out.gz": (1291160, "b1bd4b0eebe792da42a7cccb5a1c73d519fd2da5e00bb3cb9871cc72338df52c"),
    "slurm_reports/failed-database-path.out.gz": (2636, "86e22e4aa95dfbcb0cc78fd4d691adf186cf1c4200c5c1a7a29585ccea74bbda"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Parse the bundle path, verify every core file, and exit nonzero on drift."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bundle",
        nargs="?",
        type=Path,
        default=Path("ice_mix/docs/_local_sample"),
        help="Restaged local fixture root (default: local verification location).",
    )
    args = parser.parse_args()

    failures = []
    for relative_path, (expected_size, expected_hash) in EXPECTED.items():
        path = args.bundle / relative_path
        if not path.is_file():
            failures.append(f"MISSING {relative_path}")
            continue
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if (actual_size, actual_hash) != (expected_size, expected_hash):
            failures.append(
                f"MISMATCH {relative_path}: bytes={actual_size}, sha256={actual_hash}"
            )
            continue
        print(f"PASS {relative_path}")

    if failures:
        raise SystemExit("\n".join(failures))
    print("All core local verification artifacts match the tracked manifest.")


if __name__ == "__main__":
    main()
