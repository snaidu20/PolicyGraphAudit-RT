# Data directory

This repository ships with **only small, canonical artifacts**. Larger raw datasets
and intermediate processed graphs are reproducible from the M1 acquisition scripts.

## What is shipped

```
data/
├── raw/
│   ├── opp115/categories.json              # OPP-115 10-category schema
│   ├── exodus/trackers.json                # 432 Exodus tracker registry
│   ├── yale_pl/trackers_parsed.json        # 77 Yale Privacy Lab profiles
│   ├── trackercontrol/xray-blacklist.json  # 771 tracker-domain mappings
│   ├── MANIFEST.md                         # full data inventory
│   └── M1_REPORT.md                        # acquisition report
└── processed/
    ├── third_parties.json                  # canonical 817-entity third-party list
    ├── sdk_registry.json                   # unified SDK→DataType registry (M3)
    ├── discrepancy_labels_full.parquet     # 3,202 weak-supervision labels
    └── m5_best_model_masked.pt             # trained GNN checkpoint (macro F1=0.9561)
```

## What is NOT shipped (reproducible)

- **Princeton Privacy Policy Corpus** (~5K sampled markdown files, ~30 MB)
- **Google Play Data Safety** parquet sample (~5K rows, ~5 MB)
- **iOS privacy labels** sample (~22K entries, ~12 MB)
- **268 fetched privacy policy texts** (~6 MB)
- **268 fused HeteroData graphs** (~42 MB)
- **252 generated audit PDFs** (~3 MB; 5 are shipped as samples in `reports/audits/`)

## How to reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run M1 acquisition (downloads ~190 MB to data/raw/)
python -m src.m1_acquire.load_opp115
python -m src.m1_acquire.load_princeton_ppc
python -m src.m1_acquire.load_play_data_safety
python -m src.m1_acquire.load_ios_labels
python -m src.m1_acquire.load_exodus
python -m src.m1_acquire.load_trackercontrol
python -m src.m1_acquire.load_yale_pl

# 3. Fetch policies + fuse graphs (M4)
python -m src.m4_fusion.smoke_test            # 20-app sanity check
python -m src.m4_fusion.run_full_fusion        # all 268 apps (~13 min)

# 4. Train the GNN with edge masking (M5)
python -m src.m5_model.run_masked_experiment   # ~6 min on CPU

# 5. Generate per-app audit PDFs (M6)
python -c "from src.m6_report.batch import generate_all; generate_all()"

# 6. Launch the dashboard (M7)
python -m src.m7_dashboard.app                  # http://localhost:8050
```

All scripts are deterministic with `seed=42`. End-to-end reproduction takes
about 25 minutes on a modern CPU (no GPU required).
