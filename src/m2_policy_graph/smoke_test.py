"""
smoke_test.py — End-to-end smoke test for the M2 Policy Graph Builder.

Loads 5 random policies from data/raw/princeton_ppc/policies/,
builds a graph for each, prints stats, and saves all graphs to
data/interim/sample_policy_graphs.pkl.

Should complete in under 2 minutes (classifier already cached on disk).

Usage:
  python -m m2_policy_graph.smoke_test
  # or
  python src/m2_policy_graph/smoke_test.py
"""

from __future__ import annotations

import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Resolve project root
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
        if (p.parent / "data").is_dir() and (p.parent / "src").is_dir():
            return p.parent
    raise RuntimeError("Cannot locate project root. Run from within PolicyGraphAudit-RT/")


PROJECT_ROOT = _find_project_root()
POLICIES_DIR = PROJECT_ROOT / "data" / "raw" / "princeton_ppc" / "policies"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "sample_policy_graphs.pkl"


def load_policy(path: Path) -> str:
    """Read a policy markdown file and return its text."""
    return path.read_text(encoding="utf-8", errors="replace")


def pick_policies(n: int = 5, seed: int = 7) -> List[Path]:
    """Pick n random policy files from POLICIES_DIR."""
    all_files = sorted(POLICIES_DIR.glob("*.md"))
    if not all_files:
        raise FileNotFoundError(f"No .md files found in {POLICIES_DIR}")
    rng = random.Random(seed)
    return rng.sample(all_files, min(n, len(all_files)))


def run_smoke_test(n: int = 5, seed: int = 7) -> None:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from m2_policy_graph.build_graph import build_policy_graph
    from m2_policy_graph.classifier import _get_singleton

    print("=" * 60)
    print("M2 Policy Graph Builder — Smoke Test")
    print("=" * 60)

    t0 = time.time()

    # Load classifier once (shared across all policies)
    print("\nLoading classifier...")
    clf = _get_singleton()
    print(f"  Classifier loaded in {time.time() - t0:.1f}s")

    # Pick policies
    policy_files = pick_policies(n, seed)
    print(f"\nSelected {len(policy_files)} policies:")
    for f in policy_files:
        print(f"  {f.name}")

    # Build graphs
    results: List[Tuple[str, nx.MultiDiGraph, Dict[str, Any]]] = []
    print()

    for idx, path in enumerate(policy_files, 1):
        t1 = time.time()
        policy_id = path.stem
        text = load_policy(path)
        G, stats = build_policy_graph(text, policy_id, classifier=clf)
        elapsed = time.time() - t1

        print(f"[{idx}/{len(policy_files)}] {policy_id}")
        print(f"  chars={len(text):,}  segments={stats['n_segments']}")
        print(f"  nodes={stats['n_nodes']}  edges={stats['n_edges']}")
        print(f"  data_types={stats['n_data_types']}  "
              f"purposes={stats['n_purposes']}  "
              f"third_parties={stats['n_third_parties']}")
        print(f"  edge breakdown: {stats['edge_counts']}")

        # Sample top segment categories
        seg_cats = [
            d["opp115_category"]
            for _, d in G.nodes(data=True)
            if d.get("node_type") == "PolicySegment"
        ]
        from collections import Counter
        cat_counts = Counter(seg_cats).most_common(3)
        print(f"  top OPP-115 categories: {cat_counts}")
        print(f"  time: {elapsed:.1f}s")
        print()

        results.append((policy_id, G, stats))

    # Aggregate summary
    total_elapsed = time.time() - t0
    print("=" * 60)
    print("Summary across 5 policies:")
    avg_nodes = sum(s["n_nodes"] for _, _, s in results) / len(results)
    avg_edges = sum(s["n_edges"] for _, _, s in results) / len(results)
    avg_segs = sum(s["n_segments"] for _, _, s in results) / len(results)
    avg_dt = sum(s["n_data_types"] for _, _, s in results) / len(results)
    avg_pu = sum(s["n_purposes"] for _, _, s in results) / len(results)
    avg_tp = sum(s["n_third_parties"] for _, _, s in results) / len(results)

    # Aggregate edge counts
    edge_totals: Dict[str, int] = {}
    for _, _, s in results:
        for et, cnt in s["edge_counts"].items():
            edge_totals[et] = edge_totals.get(et, 0) + cnt
    avg_edges_by_type = {et: round(v / len(results), 1) for et, v in edge_totals.items()}

    print(f"  avg nodes:        {avg_nodes:.1f}")
    print(f"  avg edges:        {avg_edges:.1f}")
    print(f"  avg segments:     {avg_segs:.1f}")
    print(f"  avg data_types:   {avg_dt:.1f}")
    print(f"  avg purposes:     {avg_pu:.1f}")
    print(f"  avg third_parties:{avg_tp:.1f}")
    print(f"  avg edges by type:{avg_edges_by_type}")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print("=" * 60)

    # Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graphs": [(pid, G) for pid, G, _ in results],
        "stats": [{"policy_id": pid, **s} for pid, _, s in results],
        "summary": {
            "avg_nodes": avg_nodes,
            "avg_edges": avg_edges,
            "avg_segments": avg_segs,
            "avg_data_types": avg_dt,
            "avg_purposes": avg_pu,
            "avg_third_parties": avg_tp,
            "avg_edges_by_type": avg_edges_by_type,
            "total_elapsed_s": round(total_elapsed, 1),
        },
    }
    with open(OUTPUT_PATH, "wb") as fh:
        pickle.dump(payload, fh, protocol=5)
    print(f"\nSaved graphs to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_smoke_test(n=5)
