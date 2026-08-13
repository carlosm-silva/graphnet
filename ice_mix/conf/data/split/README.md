# Event split configurations

`random.yaml` deterministically shuffles each database's `truth.event_no` values with seed 42 and makes 80/10/10 train/validation/test partitions. `csv.yaml` loads explicit `event_no` lists matched positionally to database paths. **VERIFIED-STATIC.**

The current CSV config lists two train and validation files while the standard data config lists three databases. `load_csv_splits` rejects this mismatch; add tau-neutrino selection files before using CSV mode. **VERIFIED-STATIC.**
