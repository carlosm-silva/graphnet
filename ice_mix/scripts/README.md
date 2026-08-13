# Standalone scripts

`add_indices.py` inspects a GraphNeT SQLite database and creates indexes on `event_no` columns when missing. Index creation mutates the database and should be performed on a controlled project-storage copy, never on the only copy of an input dataset.

This preprocessing utility is not invoked by training launchers. Its place in the current data-production workflow is **UNVERIFIED-CLUSTER** pending the author's answer about data provenance.
