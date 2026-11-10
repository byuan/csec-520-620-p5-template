# Data — UNSW-NB15

**Do not commit the dataset.** `.gitignore` already excludes everything in `data/` except
this file.

1. Download the course dataset ZIP from Google Drive (the same download as Project 2):
   https://drive.google.com/file/d/1jDsXYALnEsLzYYGCEtxgkAYygS-vKIt7/view?usp=sharing
2. Extract these two files into this folder:
   - `data/UNSW_NB15_training-set.csv` (175,341 rows) — training
   - `data/UNSW_NB15_testing-set.csv` (82,332 rows) — testing, for final evaluation only
3. Check the paths in `config.yaml` (`data.train_csv`, `data.test_csv`).

- **Target:** `label` (0 = normal, 1 = attack).
- **Used for analysis only:** `attack_cat` (Normal plus nine attack categories).
- **Dropped:** `id`.
- **Categorical:** `proto`, `service`, `state`.
- **Reference:** N. Moustafa and J. Slay, "UNSW-NB15: A Comprehensive Data Set for
  Network Intrusion Detection Systems," MilCIS 2015.

No data yet? `make smoke` runs the whole pipeline on small synthetic data with the same
columns. Its numbers are meaningless; use it only to check that your code runs.
