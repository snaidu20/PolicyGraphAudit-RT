"""
M7 Dashboard Smoke Test
========================
Verifies:
1. All required data files exist and are loadable
2. All tab modules import without errors
3. All tab layouts render without errors
4. App layout builds successfully

Usage:
    cd /home/user/workspace/PolicyGraphAudit-RT
    python -m src.m7_dashboard.smoke_test
"""

import sys
import os
import traceback
import time

# Make project root importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

errors = []

# ---- 1. Data file checks --------------------------------------------------

_DATA_FILES = {
    "fused_graphs_full.pt": "/home/user/workspace/PolicyGraphAudit-RT/data/processed/fused_graphs_full.pt",
    "discrepancy_labels_full.parquet": "/home/user/workspace/PolicyGraphAudit-RT/data/processed/discrepancy_labels_full.parquet",
    "sdk_registry.json": "/home/user/workspace/PolicyGraphAudit-RT/data/processed/sdk_registry.json",
    "m5_test_metrics_masked.json": "/home/user/workspace/PolicyGraphAudit-RT/reports/m5_test_metrics_masked.json",
    "m5_ablation_table_masked.csv": "/home/user/workspace/PolicyGraphAudit-RT/reports/m5_ablation_table_masked.csv",
    "m5_mask_prob_sensitivity.csv": "/home/user/workspace/PolicyGraphAudit-RT/reports/m5_mask_prob_sensitivity.csv",
    "m5_model_card.md": "/home/user/workspace/PolicyGraphAudit-RT/reports/m5_model_card.md",
}

print("\n=== M7 Dashboard Smoke Test ===\n")
print("--- Data Files ---")
missing_files = []
for name, path in _DATA_FILES.items():
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  {PASS} {name} ({size:,} bytes)")
    else:
        print(f"  {FAIL} {name} MISSING at {path}")
        missing_files.append(name)
        errors.append(f"Missing data file: {name}")

# ---- 2. Load data files ---------------------------------------------------

print("\n--- Data Loading ---")
try:
    import pandas as pd
    df = pd.read_parquet(_DATA_FILES["discrepancy_labels_full.parquet"])
    print(f"  {PASS} discrepancy_labels_full.parquet ({len(df)} rows, {df['app_id'].nunique()} apps)")
except Exception as e:
    print(f"  {FAIL} discrepancy_labels_full.parquet: {e}")
    errors.append(f"parquet load error: {e}")

try:
    import json
    with open(_DATA_FILES["sdk_registry.json"]) as f:
        sdks = json.load(f)
    print(f"  {PASS} sdk_registry.json ({len(sdks)} SDKs)")
except Exception as e:
    print(f"  {FAIL} sdk_registry.json: {e}")
    errors.append(f"sdk_registry load error: {e}")

try:
    import torch
    graphs = torch.load(_DATA_FILES["fused_graphs_full.pt"], map_location="cpu", weights_only=False)
    print(f"  {PASS} fused_graphs_full.pt ({len(graphs)} graphs)")
except Exception as e:
    print(f"  {FAIL} fused_graphs_full.pt: {e}")
    errors.append(f"graphs load error: {e}")

try:
    import json
    with open(_DATA_FILES["m5_test_metrics_masked.json"]) as f:
        metrics = json.load(f)
    print(f"  {PASS} m5_test_metrics_masked.json (macro_f1={metrics.get('macro_f1', '?'):.4f})")
except Exception as e:
    print(f"  {FAIL} m5_test_metrics_masked.json: {e}")
    errors.append(f"metrics load error: {e}")

# ---- 3. Module imports ----------------------------------------------------

print("\n--- Module Imports ---")
_MODULES = [
    "src.m7_dashboard.app",
    "src.m7_dashboard.tabs.overview",
    "src.m7_dashboard.tabs.graph_explorer",
    "src.m7_dashboard.tabs.discrepancy_atlas",
    "src.m7_dashboard.tabs.sdk_leaderboard",
    "src.m7_dashboard.tabs.model_card",
    "src.m7_dashboard.tabs.audit_reports",
]

imported = {}
for mod_name in _MODULES:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        imported[mod_name] = mod
        print(f"  {PASS} {mod_name}")
    except Exception as e:
        print(f"  {FAIL} {mod_name}: {e}")
        traceback.print_exc()
        errors.append(f"import error {mod_name}: {e}")

# ---- 4. Tab layout renders ------------------------------------------------

print("\n--- Tab Layout Renders ---")
_TAB_MODULES = {
    "Tab 1: Overview": "src.m7_dashboard.tabs.overview",
    "Tab 2: Graph Explorer": "src.m7_dashboard.tabs.graph_explorer",
    "Tab 3: Discrepancy Atlas": "src.m7_dashboard.tabs.discrepancy_atlas",
    "Tab 4: SDK Risk Leaderboard": "src.m7_dashboard.tabs.sdk_leaderboard",
    "Tab 5: Model Card": "src.m7_dashboard.tabs.model_card",
    "Tab 6: Audit Reports": "src.m7_dashboard.tabs.audit_reports",
}

for tab_name, mod_name in _TAB_MODULES.items():
    if mod_name not in imported:
        print(f"  {SKIP} {tab_name} (module not imported)")
        continue
    try:
        t0 = time.time()
        result = imported[mod_name].layout()
        elapsed = time.time() - t0
        print(f"  {PASS} {tab_name} (rendered in {elapsed:.2f}s)")
    except Exception as e:
        print(f"  {FAIL} {tab_name}: {e}")
        traceback.print_exc()
        errors.append(f"layout error {tab_name}: {e}")

# ---- 5. App layout --------------------------------------------------------

print("\n--- App Layout ---")
if "src.m7_dashboard.app" in imported:
    try:
        app_mod = imported["src.m7_dashboard.app"]
        layout = app_mod.app.layout
        assert layout is not None
        print(f"  {PASS} app.layout is not None")
    except Exception as e:
        print(f"  {FAIL} app.layout: {e}")
        errors.append(f"app layout error: {e}")

# ---- 6. Optional files note -----------------------------------------------

_OPTIONAL = {
    "reports/audits/ directory": "/home/user/workspace/PolicyGraphAudit-RT/reports/audits",
    "m5_training_curves_masked.png": "/home/user/workspace/PolicyGraphAudit-RT/reports/m5_training_curves_masked.png",
}
print("\n--- Optional Files ---")
for name, path in _OPTIONAL.items():
    exists = os.path.exists(path)
    status = PASS if exists else SKIP
    note = "" if exists else " (will use placeholder — not an error)"
    print(f"  {status} {name}{note}")

# ---- Summary --------------------------------------------------------------

print("\n=== Summary ===")
if errors:
    print(f"  {len(errors)} error(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  All checks passed.")
    print("\nDashboard smoke test passed")
