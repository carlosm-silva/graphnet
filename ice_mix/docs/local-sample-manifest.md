# Local verification artifact manifest

The author staged a local-only fixture bundle on 2026-08-13. It is intentionally
excluded from Git because it contains a 337 MiB checkpoint and selected event
data. This page lets a successor identify the same bundle without treating an
ignored laptop directory as part of the released package.

The bundle is derived from IceCube simulation inputs and run
`IceMix_2026-05-21_08-14-23_job-8770973`. Its exact upstream dataset production
campaign and source-database checksums are **UNVERIFIED-CLUSTER**: search the
author's inherited PACE files for that run name and the three standard database
basenames, then ask the author or Jiyuan on Slack to confirm provenance before
using it for a scientific claim. The hashes below verify only that two local
copies of this handoff bundle are byte-identical.

## Core files

| Relative bundle path | Bytes | SHA-256 |
|---|---:|---|
| `standard_sqlite/numu.db` | 151552 | `588ebe9949d4b2af74e3abf6030926c5df4d33950bce54ad407e52cf2fbccbc6` |
| `standard_sqlite/nue.db` | 180224 | `cc5f256ad844ae98c416064849f9b84b27ed05975879202ae1391138bf33fc7c` |
| `standard_sqlite/nutau.db` | 147456 | `af50f556f40da96353335ac89293eb823348d9947f55b3a7f4b90e265be284bf` |
| `augmented_sqlite/numu_augmented.db` | 24576 | `0dc5e79685c6eb0a7abaa38aa666d8eb1653e4e51d8839d131bca865ad158b3e` |
| `base_run/.hydra/config.yaml` | 1662 | `fa77fb9c90755b6fca971d6d427b43d761c7c873232ef1401a9c9c5b3e5167fc` |
| `base_run/checkpoints/best-epoch=18-val_loss=1.4589.ckpt` | 353310946 | `177a93cc53e73a8b9b1193ea3c06a1293891c6941aadcdc1bb2b8f9c6b5b1746` |
| `base_run/predictions/state_dict.pth` | 117758703 | `4616be18c664bbb3be66dc96f558851ac941b1c1cf206487b7ed246c46cb2fcb` |
| `base_run/predictions/results.csv` | 2764 | `a96c52c7dc7d220188b7f2a713b8784c15aa100f147c7b92b27dc178336253f2` |
| `prediction/results.csv` | 14852 | `a05aafa26f78cbd2cafba2be379425dfebbd6b34d3190185ed19fad3b2262403` |
| `slurm_reports/successful-base-training.out.gz` | 1291160 | `b1bd4b0eebe792da42a7cccb5a1c73d519fd2da5e00bb3cb9871cc72338df52c` |
| `slurm_reports/failed-database-path.out.gz` | 2636 | `86e22e4aa95dfbcb0cc78fd4d691adf186cf1c4200c5c1a7a29585ccea74bbda` |

The manifest and byte sizes are **VERIFIED-LOCAL** on the documentation
workstation on 2026-08-13. The earlier tracked Python verifier was removed when
the scope was corrected to documentation only. A successor can calculate a
candidate file's digest with `sha256sum /path/to/file` and compare it with this
table. Never commit the bundle or use its ignored local staging location in a
PACE launcher.
