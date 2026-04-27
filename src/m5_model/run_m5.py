"""
run_m5.py — Full M5 pipeline runner.

Runs:
1. train_model() — full HeteroAuditGNN
2. run_all_baselines() — 3 ablation baselines
3. save_ablation_table() — compile results
4. print summary

Usage:
    python -m src.m5_model.run_m5
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

_PROJ_ROOT = Path(__file__).resolve().parents[2]


def main():
    total_t0 = time.time()

    # 1. Train full model
    print("=" * 60)
    print("Step 1: Train full HeteroAuditGNN")
    print("=" * 60)
    from src.m5_model.train import train_model, save_training_curves
    from src.m5_model.dataset import app_level_split, load_audit_dataset

    metrics, model, splits = train_model(verbose=True)
    train_graphs, val_graphs, test_graphs = splits

    history = metrics.pop("history")
    save_training_curves(history)

    # Save test metrics
    metrics_path = _PROJ_ROOT / "reports" / "m5_test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(metrics_path), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Test metrics saved to {metrics_path}")

    # 2. Run baselines on same split
    print()
    print("=" * 60)
    print("Step 2: Run ablation baselines")
    print("=" * 60)
    from src.m5_model.baselines import run_all_baselines
    baseline_results = run_all_baselines(train_graphs, val_graphs, test_graphs)

    # 3. Compile ablation table
    print()
    print("=" * 60)
    print("Step 3: Compile ablation table")
    print("=" * 60)
    from src.m5_model.ablation import save_ablation_table, print_ablation_summary
    df = save_ablation_table()
    print_ablation_summary(df)

    # Final summary
    total_time = time.time() - total_t0
    print()
    print("=" * 60)
    print("M5 PIPELINE COMPLETE")
    print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Full model macro F1: {metrics['macro_f1']:.4f}")
    print(f"  UNDECLARED_COLLECTION F1: "
          f"{metrics['per_class_f1'].get('UNDECLARED_COLLECTION', 0):.4f}")
    print("=" * 60)

    return {
        "full_model": metrics,
        "baselines": baseline_results,
        "total_runtime_sec": total_time,
    }


if __name__ == "__main__":
    main()
