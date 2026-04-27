"""
ablation.py — Compile ablation table from saved JSON results.

Produces:
- reports/m5_ablation_table.csv
- reports/m5_ablation_table.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_PROJ_ROOT = Path(__file__).resolve().parents[2]
_REPORTS_DIR = _PROJ_ROOT / "reports"
_BASELINE_DIR = _REPORTS_DIR / "baselines"

CLASS_NAMES = [
    "CONSISTENT",
    "POLICY_LABEL_MISMATCH",
    "OVER_DISCLOSURE",
    "UNDECLARED_COLLECTION",
]

MODEL_ORDER = [
    "tsne_pca_clustering",
    "text_only_logreg",
    "policy_only_gnn",
    "full_hetero_gnn",
]


def _load_result(name: str) -> dict | None:
    """Load a saved JSON result for a model/baseline."""
    if name == "full_hetero_gnn":
        path = _REPORTS_DIR / "m5_test_metrics.json"
    else:
        path = _BASELINE_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(str(path)) as f:
        return json.load(f)


def build_ablation_table() -> pd.DataFrame:
    """
    Build the ablation comparison table.

    Returns
    -------
    pd.DataFrame with columns:
        model, macro_f1, f1_consistent, f1_policy_label_mismatch,
        f1_over_disclosure, f1_undeclared_collection, params, runtime_sec
    """
    rows = []
    for name in MODEL_ORDER:
        result = _load_result(name)
        if result is None:
            print(f"  [WARNING] No result found for {name}, skipping.")
            continue

        per_class = result.get("per_class_f1", {})
        rows.append({
            "model": name,
            "macro_f1": round(result.get("macro_f1", 0.0), 4),
            "f1_consistent": round(per_class.get("CONSISTENT", 0.0), 4),
            "f1_policy_label_mismatch": round(
                per_class.get("POLICY_LABEL_MISMATCH", 0.0), 4),
            "f1_over_disclosure": round(per_class.get("OVER_DISCLOSURE", 0.0), 4),
            "f1_undeclared_collection": round(
                per_class.get("UNDECLARED_COLLECTION", 0.0), 4),
            "params": result.get("params", 0),
            "runtime_sec": round(result.get("runtime_sec", 0.0), 1),
        })

    df = pd.DataFrame(rows)
    return df


def save_ablation_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build and save ablation table to CSV and Markdown."""
    if df is None:
        df = build_ablation_table()

    csv_path = _REPORTS_DIR / "m5_ablation_table.csv"
    md_path = _REPORTS_DIR / "m5_ablation_table.md"

    df.to_csv(str(csv_path), index=False)
    print(f"  Ablation table (CSV) saved to {csv_path}")

    # Markdown table
    md_lines = [
        "# M5 Ablation Table\n",
        "| Model | Macro F1 | F1 Consistent | F1 Pol/Label | F1 Over-Disc | F1 Undecl | Params | Runtime (s) |",
        "|-------|----------|--------------|-------------|------------|--------|--------|-------------|",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['model']} "
            f"| {row['macro_f1']:.4f} "
            f"| {row['f1_consistent']:.4f} "
            f"| {row['f1_policy_label_mismatch']:.4f} "
            f"| {row['f1_over_disclosure']:.4f} "
            f"| {row['f1_undeclared_collection']:.4f} "
            f"| {int(row['params']):,} "
            f"| {row['runtime_sec']:.1f} |"
        )

    with open(str(md_path), "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"  Ablation table (Markdown) saved to {md_path}")

    return df


def print_ablation_summary(df: pd.DataFrame):
    """Print a human-readable summary comparing baselines to full model."""
    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print()

    full_row = df[df["model"] == "full_hetero_gnn"]
    if full_row.empty:
        return
    full_f1 = float(full_row["macro_f1"].iloc[0])

    print("  Comparison vs full_hetero_gnn:")
    for _, row in df[df["model"] != "full_hetero_gnn"].iterrows():
        delta = full_f1 - float(row["macro_f1"])
        sign = "+" if delta > 0 else ""
        print(f"    {row['model']:<28s} delta = {sign}{delta:+.4f}")


if __name__ == "__main__":
    df = save_ablation_table()
    print_ablation_summary(df)
