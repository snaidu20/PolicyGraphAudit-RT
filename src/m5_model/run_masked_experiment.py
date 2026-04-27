"""
run_masked_experiment.py — End-to-end M5 masked experiment runner.

Runs:
1. train_masked_model (mask_prob=0.30)
2. All baselines on masked splits
3. Mask-probability sensitivity sweep {0.0, 0.15, 0.30, 0.50}
4. Ablation tables (masked + comparison vs unmasked)

Usage
-----
    cd PolicyGraphAudit-RT
    python -m src.m5_model.run_masked_experiment
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJ_ROOT))

from src.m5_model.dataset import CLASS_NAMES, app_level_split, load_audit_dataset
from src.m5_model.edge_masking import apply_masking_to_graphs
from src.m5_model.train_masked import train_masked_model, save_training_curves, _METRICS_OUT
from src.m5_model.baselines_masked import run_all_baselines

SEED = 42
MASK_PROB = 0.30
_REPORTS = _PROJ_ROOT / "reports"
_REPORTS.mkdir(parents=True, exist_ok=True)


def run_sensitivity_sweep(
    train_graphs, val_graphs, test_graphs,
    probs=(0.0, 0.15, 0.30, 0.50),
):
    """Run full_hetero_gnn for each mask probability, return list of result dicts."""
    print()
    print("=" * 60)
    print("Mask-probability sensitivity sweep")
    print("=" * 60)
    rows = []
    for prob in probs:
        print(f"\n--- mask_prob = {prob} ---")
        save_path = _PROJ_ROOT / "data" / "processed" / f"m5_best_model_masked_p{int(prob*100):02d}.pt"
        metrics, _, _ = train_masked_model(
            mask_prob=prob,
            save_path=save_path,
            verbose=False,
        )
        row = {
            "mask_prob": prob,
            "macro_f1": metrics["macro_f1"],
            "f1_CONSISTENT":            metrics["per_class_f1"]["CONSISTENT"],
            "f1_POLICY_LABEL_MISMATCH": metrics["per_class_f1"]["POLICY_LABEL_MISMATCH"],
            "f1_OVER_DISCLOSURE":       metrics["per_class_f1"]["OVER_DISCLOSURE"],
            "f1_UNDECLARED_COLLECTION": metrics["per_class_f1"]["UNDECLARED_COLLECTION"],
            "epochs_trained": metrics["epochs_trained"],
            "runtime_sec": metrics["runtime_sec"],
        }
        rows.append(row)
        print(f"  macro_f1={row['macro_f1']:.4f}  "
              f"UNDECL={row['f1_UNDECLARED_COLLECTION']:.4f}  "
              f"epochs={row['epochs_trained']}")
    return rows


def write_sensitivity_csv(rows: list, out_path: Path):
    header = ("mask_prob,macro_f1,f1_CONSISTENT,f1_POLICY_LABEL_MISMATCH,"
              "f1_OVER_DISCLOSURE,f1_UNDECLARED_COLLECTION,epochs_trained,runtime_sec")
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['mask_prob']},{r['macro_f1']:.4f},"
            f"{r['f1_CONSISTENT']:.4f},{r['f1_POLICY_LABEL_MISMATCH']:.4f},"
            f"{r['f1_OVER_DISCLOSURE']:.4f},{r['f1_UNDECLARED_COLLECTION']:.4f},"
            f"{r['epochs_trained']},{r['runtime_sec']:.1f}"
        )
    out_path.write_text("\n".join(lines))
    print(f"  Sensitivity CSV saved to {out_path}")


def write_ablation_table_masked(baseline_results: dict, full_metrics: dict, out_path_csv: Path, out_path_md: Path):
    """Write ablation table for masked results."""
    models_order = ["tsne_pca_clustering", "text_only_logreg", "policy_only_gnn", "full_hetero_gnn_masked"]
    rows_data = {
        "tsne_pca_clustering": baseline_results.get("tsne_pca_clustering", {}),
        "text_only_logreg":    baseline_results.get("text_only_logreg", {}),
        "policy_only_gnn":     baseline_results.get("policy_only_gnn", {}),
        "full_hetero_gnn_masked": full_metrics,
    }

    header_csv = ("model,macro_f1,f1_CONSISTENT,f1_POLICY_LABEL_MISMATCH,"
                  "f1_OVER_DISCLOSURE,f1_UNDECLARED_COLLECTION,params,runtime_sec")
    lines_csv = [header_csv]

    md_lines = [
        "# M5 Ablation Table (Edge-Masked Evaluation, mask_prob=0.30)\n",
        "| Model | Macro F1 | F1 Consistent | F1 Pol/Label | F1 Over-Disc | F1 Undecl | Params | Runtime (s) |",
        "|-------|----------|--------------|-------------|------------|--------|--------|-------------|",
    ]

    for model_name in models_order:
        r = rows_data.get(model_name, {})
        f1     = r.get("macro_f1", 0.0)
        pcf    = r.get("per_class_f1", {})
        f1_c   = pcf.get("CONSISTENT", 0.0)
        f1_pl  = pcf.get("POLICY_LABEL_MISMATCH", 0.0)
        f1_od  = pcf.get("OVER_DISCLOSURE", 0.0)
        f1_uc  = pcf.get("UNDECLARED_COLLECTION", 0.0)
        params = r.get("params", 0)
        rt     = r.get("runtime_sec", 0.0)

        lines_csv.append(
            f"{model_name},{f1:.4f},{f1_c:.4f},{f1_pl:.4f},{f1_od:.4f},{f1_uc:.4f},{params},{rt:.1f}"
        )
        display = model_name.replace("_masked", " (masked)")
        md_lines.append(
            f"| {display} | {f1:.4f} | {f1_c:.4f} | {f1_pl:.4f} | {f1_od:.4f} | {f1_uc:.4f} | {params:,} | {rt:.1f} |"
        )

    out_path_csv.write_text("\n".join(lines_csv))
    out_path_md.write_text("\n".join(md_lines))
    print(f"  Ablation CSV → {out_path_csv}")
    print(f"  Ablation MD  → {out_path_md}")


def write_comparison_table(unmasked_metrics_path: Path, masked_metrics_path: Path,
                           baseline_unmasked_dir: Path, baseline_masked_dir: Path,
                           out_path: Path):
    """Write side-by-side UNMASKED vs MASKED comparison table."""
    def _load_json(p: Path) -> dict:
        if p.exists():
            return json.loads(p.read_text())
        return {}

    # Unmasked results
    um_full   = _load_json(unmasked_metrics_path)
    um_base   = {
        "tsne_pca_clustering": _load_json(baseline_unmasked_dir / "tsne_pca_clustering.json"),
        "text_only_logreg":    _load_json(baseline_unmasked_dir / "text_only_logreg.json"),
        "policy_only_gnn":     _load_json(baseline_unmasked_dir / "policy_only_gnn.json"),
    }

    # Masked results
    m_full  = _load_json(masked_metrics_path)
    m_base  = {
        "tsne_pca_clustering": _load_json(baseline_masked_dir / "tsne_pca_clustering.json"),
        "text_only_logreg":    _load_json(baseline_masked_dir / "text_only_logreg.json"),
        "policy_only_gnn":     _load_json(baseline_masked_dir / "policy_only_gnn.json"),
    }

    lines = [
        "# M5 Unmasked vs. Masked Comparison Table\n",
        "> **Unmasked** (mask_prob=0.0): trivial tautology — model reconstructs own supervision signal (F1=1.0).",
        "> **Masked** (mask_prob=0.30): defensible evaluation — 30% of label-determining edges hidden.",
        "",
        "| Model | Unmasked Macro F1 | Masked Macro F1 | Delta |",
        "|-------|-------------------|-----------------|-------|",
    ]

    def _row(model_display, um_f1, m_f1):
        delta = m_f1 - um_f1
        sign  = "+" if delta >= 0 else ""
        return f"| {model_display} | {um_f1:.4f} | {m_f1:.4f} | {sign}{delta:.4f} |"

    lines.append(_row(
        "tsne_pca_clustering",
        um_base["tsne_pca_clustering"].get("macro_f1", 0.0),
        m_base["tsne_pca_clustering"].get("macro_f1", 0.0),
    ))
    lines.append(_row(
        "text_only_logreg",
        um_base["text_only_logreg"].get("macro_f1", 0.0),
        m_base["text_only_logreg"].get("macro_f1", 0.0),
    ))
    lines.append(_row(
        "policy_only_gnn",
        um_base["policy_only_gnn"].get("macro_f1", 0.0),
        m_base["policy_only_gnn"].get("macro_f1", 0.0),
    ))
    lines.append(_row(
        "**full_hetero_gnn** ← headline",
        um_full.get("macro_f1", 1.0),
        m_full.get("macro_f1", 0.0),
    ))

    lines += [
        "",
        "## Notes",
        "",
        "- The unmasked full_hetero_gnn achieves F1=1.0000 because it can trivially recover",
        "  the discrepancy label from DECLARES_COLLECTS, DECLARES_SHARES, and COLLECTS_DATATYPE",
        "  edges — the exact edges used to compute the label.",
        "- Under 30% masking, the model must infer labels from policy text (MENTIONS edges) and",
        "  remaining graph context, making this a genuine generalisation test.",
        "- The gap (Δ) for the full model quantifies how much of the prior result was circular.",
        "- Policy-only and text baselines are largely unaffected by masking (they never used",
        "  structural label-determining edges), serving as consistency checks.",
    ]

    out_path.write_text("\n".join(lines))
    print(f"  Comparison table → {out_path}")


def main():
    t_start = time.time()
    print("=" * 60)
    print("M5 Masked Experiment — Full Run")
    print(f"  mask_prob={MASK_PROB}, seed={SEED}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset and split (same seed as unmasked run)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading dataset and splitting...")
    graphs, _ = load_audit_dataset()
    train_graphs, val_graphs, test_graphs = app_level_split(graphs, seed=SEED)
    print(f"  Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # Apply masking
    train_m, ts = apply_masking_to_graphs(train_graphs, mask_prob=MASK_PROB, seed=SEED)
    val_m,   vs = apply_masking_to_graphs(val_graphs,   mask_prob=MASK_PROB, seed=SEED)
    test_m,  es = apply_masking_to_graphs(test_graphs,  mask_prob=MASK_PROB, seed=SEED)
    print(f"  Train removed {ts['total_edges_removed']} edges ({ts['total_pairs_masked']} pairs)")
    print(f"  Val   removed {vs['total_edges_removed']} edges ({vs['total_pairs_masked']} pairs)")
    print(f"  Test  removed {es['total_edges_removed']} edges ({es['total_pairs_masked']} pairs)")

    # ------------------------------------------------------------------
    # 2. Train full_hetero_gnn on masked graphs
    # ------------------------------------------------------------------
    print("\n[2/4] Training full_hetero_gnn (masked)...")
    full_metrics, model, _ = train_masked_model(
        mask_prob=MASK_PROB,
        verbose=True,
    )
    history = full_metrics.pop("history", {})

    # Save training curves
    from src.m5_model.train_masked import _CURVES_OUT
    save_training_curves(history, _CURVES_OUT)

    # Save metrics JSON
    _METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(_METRICS_OUT), "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  Full model metrics → {_METRICS_OUT}")

    # ------------------------------------------------------------------
    # 3. Run baselines on masked splits
    # ------------------------------------------------------------------
    print("\n[3/4] Running baselines on masked splits...")
    baseline_results = run_all_baselines(train_m, val_m, test_m)

    # ------------------------------------------------------------------
    # 4. Sensitivity sweep
    # ------------------------------------------------------------------
    print("\n[4/4] Sensitivity sweep...")
    sweep_rows = run_sensitivity_sweep(train_graphs, val_graphs, test_graphs,
                                       probs=(0.0, 0.15, 0.30, 0.50))
    sweep_csv_path = _REPORTS / "m5_mask_prob_sensitivity.csv"
    write_sensitivity_csv(sweep_rows, sweep_csv_path)

    # ------------------------------------------------------------------
    # 5. Write ablation tables
    # ------------------------------------------------------------------
    print("\n[Tables] Generating ablation tables...")
    write_ablation_table_masked(
        baseline_results,
        full_metrics,
        _REPORTS / "m5_ablation_table_masked.csv",
        _REPORTS / "m5_ablation_table_masked.md",
    )

    write_comparison_table(
        unmasked_metrics_path   = _PROJ_ROOT / "reports" / "m5_test_metrics.json",
        masked_metrics_path     = _METRICS_OUT,
        baseline_unmasked_dir   = _PROJ_ROOT / "reports" / "baselines",
        baseline_masked_dir     = _PROJ_ROOT / "reports" / "baselines_masked",
        out_path                = _REPORTS / "m5_ablation_comparison.md",
    )

    total_runtime = time.time() - t_start

    # ------------------------------------------------------------------
    # 6. Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  full_hetero_gnn masked macro F1: {full_metrics['macro_f1']:.4f}")
    print(f"  Per-class F1:")
    for cls in CLASS_NAMES:
        f1_val = full_metrics["per_class_f1"].get(cls, 0.0)
        print(f"    {cls:<30s}: {f1_val:.4f}")
    print(f"\n  Sensitivity sweep:")
    for row in sweep_rows:
        print(f"    mask_prob={row['mask_prob']:.2f} → macro_f1={row['macro_f1']:.4f}")
    print(f"\n  Total runtime: {total_runtime:.1f}s ({total_runtime/60:.1f} min)")

    return full_metrics, baseline_results, sweep_rows


if __name__ == "__main__":
    main()
