"""
smoke_test.py — M3 smoke test.

- Loads Play DSL sample
- Builds graphs for top-10 apps by realInstalls
- Prints stats for each
- Saves all graphs to data/interim/sample_label_runtime_graphs.pkl

Expected runtime: < 1 minute.
"""

from __future__ import annotations

import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution (same as other M3 modules)
# ---------------------------------------------------------------------------

def _resolve_data_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
        candidate = parent.parent / "data"
        if candidate.is_dir():
            return candidate
    return Path("data")


OUTPUT_PATH = _resolve_data_root() / "interim" / "sample_label_runtime_graphs.pkl"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_smoke_test(n_apps: int = 10) -> dict:
    """
    Build graphs for the top-n apps by realInstalls and return aggregate stats.

    Returns
    -------
    dict with keys: app_ids, per_app_stats, aggregate
    """
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load Play DSL
    # ------------------------------------------------------------------
    from m3_runtime_graph.load_play_dsl import load_play_dsl
    print(f"Loading Play DSL sample...")
    df = load_play_dsl()
    n_total_rows  = len(df)
    n_unique_apps = df["appId"].nunique()
    print(f"  {n_total_rows} rows | {n_unique_apps} unique appIds")

    # Select top-n apps by realInstalls
    top_apps = (
        df.groupby("appId")["realInstalls"]
        .max()
        .nlargest(n_apps)
        .index
        .tolist()
    )
    print(f"  Top {n_apps} apps: {top_apps[:3]} ...")

    # ------------------------------------------------------------------
    # 2. Build graphs
    # ------------------------------------------------------------------
    from m3_runtime_graph.build_graph import build_label_runtime_graph

    graphs: dict[str, object] = {}
    per_app_stats: list[dict] = []
    failed: list[str] = []

    for i, app_id in enumerate(top_apps, 1):
        result = build_label_runtime_graph(app_id, df)
        if result is None:
            print(f"  [{i:2d}] {app_id:50s}  — NO DATA")
            failed.append(app_id)
            continue

        G, stats = result
        graphs[app_id] = (G, stats)
        per_app_stats.append({"appId": app_id, **stats})

        # Node type breakdown
        nt_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
        et_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))

        print(
            f"  [{i:2d}] {app_id:50s}  "
            f"nodes={stats['n_nodes']:3d}  edges={stats['n_edges']:3d}  "
            f"sdks={stats['n_inferred_sdks']:2d}  "
            f"dt={stats['n_data_types']:2d}  "
            f"genre={stats['genreId']}"
        )

    # ------------------------------------------------------------------
    # 3. Aggregate stats
    # ------------------------------------------------------------------
    if per_app_stats:
        def _avg(key: str) -> float:
            vals = [s[key] for s in per_app_stats if key in s]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        # Node type averages across all built graphs
        all_nt: Counter = Counter()
        all_et: Counter = Counter()
        for app_id, (G, _) in graphs.items():
            for _, d in G.nodes(data=True):
                all_nt[d["node_type"]] += 1
            for _, _, d in G.edges(data=True):
                all_et[d["edge_type"]] += 1

        n_built = len(graphs)
        avg_nodes_by_type = {nt: round(c / n_built, 1) for nt, c in all_nt.items()}
        avg_edges_by_type = {et: round(c / n_built, 1) for et, c in all_et.items()}

        aggregate = {
            "n_apps_built":           n_built,
            "n_apps_failed":          len(failed),
            "avg_nodes":              _avg("n_nodes"),
            "avg_edges":              _avg("n_edges"),
            "avg_label_collects":     _avg("n_label_decl_collects"),
            "avg_label_shares":       _avg("n_label_decl_shares"),
            "avg_label_purposes":     _avg("n_label_purposes"),
            "avg_inferred_sdks":      _avg("n_inferred_sdks"),
            "avg_data_types":         _avg("n_data_types"),
            "avg_nodes_by_type":      avg_nodes_by_type,
            "avg_edges_by_type":      avg_edges_by_type,
        }
    else:
        aggregate = {}

    # ------------------------------------------------------------------
    # 4. Save graphs
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as fh:
        pickle.dump(graphs, fh)
    print(f"\nSaved {len(graphs)} graphs to {OUTPUT_PATH}")

    elapsed = time.time() - t0
    print(f"\nSmoke test complete in {elapsed:.1f}s")
    print("\nAggregate stats (mean over built apps):")
    for k, v in aggregate.items():
        print(f"  {k}: {v}")

    return {
        "n_total_rows":   n_total_rows,
        "n_unique_apps":  n_unique_apps,
        "app_ids":        top_apps,
        "per_app_stats":  per_app_stats,
        "aggregate":      aggregate,
        "elapsed_s":      round(elapsed, 2),
    }


if __name__ == "__main__":
    results = run_smoke_test(n_apps=10)
