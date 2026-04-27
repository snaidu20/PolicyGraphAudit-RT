"""
smoke_test.py — End-to-end smoke test for M4 heterogeneous graph fusion.

Steps
-----
1. Fetch privacy policies (with cache) — skips already-cached.
2. Pick up to 20 apps with both a successful policy fetch AND ≥3 Play DSL rows.
3. For each: build the fused graph (M2 + M3), compute discrepancy labels,
   convert to PyG HeteroData.
4. Save list of HeteroData objects to data/processed/fused_graphs.pt.
5. Save combined discrepancy labels to data/processed/discrepancy_labels.parquet.
6. Print summary statistics.

Target: ≥15/20 apps successfully fused.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent
_ROOT = _SRC.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("m4_smoke")
log.setLevel(logging.INFO)

_DATA_ROOT = _ROOT / "data"
_PROCESSED_DIR = _DATA_ROOT / "processed"
_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FUSED_GRAPHS_PATH = _PROCESSED_DIR / "fused_graphs.pt"
DISC_LABELS_PATH  = _PROCESSED_DIR / "discrepancy_labels.parquet"

# Target number of test apps
N_TEST_APPS = 20
SUCCESS_TARGET = 15  # ≥15/20 required


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _fmt_pct(num, denom) -> str:
    if denom == 0:
        return "N/A"
    return f"{100*num/denom:.1f}%"


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Fetch policies
    # ------------------------------------------------------------------
    _print_section("Step 1: Policy Fetch")
    from m4_fusion.fetch_policies import fetch_all_policies, load_policies_index, load_policy_text

    fetch_counts = fetch_all_policies(min_rows=3, timeout=10.0, delay=1.0)
    index = load_policies_index()

    total_indexed = len(index)
    n_ok = sum(1 for v in index.values() if v.get("status") == "ok")
    n_fail = total_indexed - n_ok

    print(f"  Indexed apps      : {total_indexed}")
    print(f"  Successfully fetched : {n_ok} ({_fmt_pct(n_ok, total_indexed)})")
    print(f"  Failed/skipped    : {n_fail}")
    print(f"  Fetch counts      : {fetch_counts}")

    if n_ok < 150:
        print(f"\n  WARNING: Only {n_ok} successful fetches — below 150 prototype threshold.")
        print("  Continuing with what we have.")
    else:
        print(f"\n  ✓ Prototype threshold (≥150) met: {n_ok} policies fetched.")

    # ------------------------------------------------------------------
    # Step 2: Load Play DSL and select top 20 qualifying apps
    # ------------------------------------------------------------------
    _print_section("Step 2: App Selection")
    from m3_runtime_graph.load_play_dsl import load_play_dsl, list_unique_apps

    play_df = load_play_dsl()
    qualifying_apps = list_unique_apps(min_rows=3)
    print(f"  Qualifying apps (≥3 Play DSL rows): {len(qualifying_apps)}")

    # Select apps with successful policy fetch, sorted by policy text length
    # (longer policies = richer M2 graphs) for best smoke-test outcomes
    candidate_apps = []
    for app_id in qualifying_apps:
        entry = index.get(app_id)
        if entry and entry.get("status") == "ok":
            cached_path = entry.get("cached_path", "")
            text_len = Path(cached_path).stat().st_size if cached_path and Path(cached_path).exists() else 0
            candidate_apps.append((app_id, text_len))

    # Sort by text length descending (richest policies first)
    candidate_apps.sort(key=lambda x: x[1], reverse=True)
    test_apps = [app_id for app_id, _ in candidate_apps[:N_TEST_APPS]]

    print(f"  Apps with successful policy fetch : {len(candidate_apps)}")
    print(f"  Selected for smoke test           : {len(test_apps)}")

    if not test_apps:
        print("  ERROR: No apps available for testing. Run with --force-refetch to retry.")
        return

    # ------------------------------------------------------------------
    # Step 3: Build fused graphs
    # ------------------------------------------------------------------
    _print_section("Step 3: Graph Fusion")

    from m4_fusion.fuse import fuse_app_graphs
    from m4_fusion.discrepancy_labels import (
        compute_discrepancy_labels, save_discrepancy_labels, ALL_DISCREPANCY_TYPES
    )
    from m4_fusion.to_pyg import nx_to_pyg, get_embedder
    from m2_policy_graph.classifier import _get_singleton as get_opp115

    # Pre-load singletons
    print("  Loading OPP-115 classifier ...")
    opp115 = get_opp115()
    print("  Loading sentence-transformer embedder ...")
    embedder = get_embedder()

    all_hetero_data = []
    all_disc_dfs: List[pd.DataFrame] = []
    all_fuse_stats = []
    n_success = 0
    n_fail_fuse = 0

    with tqdm(total=len(test_apps), desc="Fusing graphs", unit="app") as pbar:
        for app_id in test_apps:
            pbar.set_description(f"Fusing {app_id[:30]}")

            policy_text = load_policy_text(app_id)
            if not policy_text:
                log.warning("No cached policy text for %s — skipping.", app_id)
                n_fail_fuse += 1
                pbar.update(1)
                continue

            # Fuse M2 + M3
            try:
                result = fuse_app_graphs(
                    app_id=app_id,
                    policy_text=policy_text,
                    play_df=play_df,
                    opp115_classifier=opp115,
                )
            except Exception as exc:
                log.warning("fuse_app_graphs failed for %s: %s", app_id, exc)
                n_fail_fuse += 1
                pbar.update(1)
                continue

            if result is None:
                log.warning("fuse_app_graphs returned None for %s.", app_id)
                n_fail_fuse += 1
                pbar.update(1)
                continue

            G_fused, fuse_stats = result
            all_fuse_stats.append(fuse_stats)

            # Compute discrepancy labels
            try:
                disc_df = compute_discrepancy_labels(G_fused, app_id)
            except Exception as exc:
                log.warning("discrepancy_labels failed for %s: %s", app_id, exc)
                disc_df = pd.DataFrame()

            all_disc_dfs.append(disc_df)

            # Convert to PyG HeteroData
            try:
                hetero = nx_to_pyg(
                    G_fused,
                    discrepancy_df=disc_df if len(disc_df) > 0 else None,
                    embedder=embedder,
                )
                hetero.app_id = app_id  # store for identification
                all_hetero_data.append(hetero)
                n_success += 1
            except Exception as exc:
                log.warning("nx_to_pyg failed for %s: %s", app_id, exc)
                n_fail_fuse += 1

            pbar.update(1)

    # ------------------------------------------------------------------
    # Step 4: Save outputs
    # ------------------------------------------------------------------
    _print_section("Step 4: Saving Outputs")

    if all_hetero_data:
        torch.save(all_hetero_data, FUSED_GRAPHS_PATH)
        size_mb = FUSED_GRAPHS_PATH.stat().st_size / 1e6
        print(f"  Saved {len(all_hetero_data)} HeteroData objects → {FUSED_GRAPHS_PATH}")
        print(f"  File size: {size_mb:.2f} MB")

    if all_disc_dfs:
        combined_disc = pd.concat(
            [df for df in all_disc_dfs if len(df) > 0], ignore_index=True
        )
        save_discrepancy_labels(combined_disc, path=DISC_LABELS_PATH)
        print(f"  Saved {len(combined_disc)} discrepancy label rows → {DISC_LABELS_PATH}")

    # ------------------------------------------------------------------
    # Step 5: Print statistics
    # ------------------------------------------------------------------
    _print_section("Step 5: Summary Statistics")

    print(f"\n  FUSION RESULTS")
    print(f"  ─────────────────────────────────────")
    print(f"  Apps attempted             : {len(test_apps)}")
    print(f"  Successfully fused         : {n_success} / {len(test_apps)}")
    target_met = "✓" if n_success >= SUCCESS_TARGET else "✗"
    print(f"  Target (≥{SUCCESS_TARGET}/20)              : {target_met} ({n_success})")

    if all_fuse_stats:
        print(f"\n  GRAPH SIZE STATISTICS (avg over {len(all_fuse_stats)} apps)")
        print(f"  ─────────────────────────────────────")

        # Avg nodes / edges by type
        all_nt_counts: Dict[str, List[int]] = defaultdict(list)
        all_et_counts: Dict[str, List[int]] = defaultdict(list)
        all_node_counts = []
        all_edge_counts = []

        for st in all_fuse_stats:
            all_node_counts.append(st["n_merged_nodes"])
            all_edge_counts.append(st["n_merged_edges"])
            for nt, cnt in st.get("n_nodes_by_type", {}).items():
                all_nt_counts[nt].append(cnt)
            for et, cnt in st.get("n_edges_by_type", {}).items():
                all_et_counts[et].append(cnt)

        print(f"  Avg total nodes : {sum(all_node_counts)/len(all_node_counts):.1f}")
        print(f"  Avg total edges : {sum(all_edge_counts)/len(all_edge_counts):.1f}")
        print(f"  Min total nodes : {min(all_node_counts)}")
        print(f"  Max total nodes : {max(all_node_counts)}")
        print(f"  Min total edges : {min(all_edge_counts)}")
        print(f"  Max total edges : {max(all_edge_counts)}")

        print(f"\n  Nodes by type (avg):")
        for nt in sorted(all_nt_counts):
            vals = all_nt_counts[nt]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"    {nt:<20}: {avg:.1f}")

        print(f"\n  Edges by type (avg):")
        for et in sorted(all_et_counts):
            vals = all_et_counts[et]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"    {et:<30}: {avg:.1f}")

    # Discrepancy label distribution
    if all_disc_dfs:
        combined = pd.concat(
            [df for df in all_disc_dfs if len(df) > 0], ignore_index=True
        )
        if len(combined) > 0:
            print(f"\n  DISCREPANCY LABEL DISTRIBUTION ({len(combined)} rows total)")
            print(f"  ─────────────────────────────────────")
            dist = combined["discrepancy_type"].value_counts()
            total = len(combined)
            for disc_type in ALL_DISCREPANCY_TYPES:
                count = dist.get(disc_type, 0)
                print(f"    {disc_type:<30}: {count:>5} ({_fmt_pct(count, total)})")

    # HeteroData stats (first graph)
    if all_hetero_data:
        print(f"\n  PYG HETERODATA — FIRST GRAPH ({all_hetero_data[0].app_id})")
        print(f"  ─────────────────────────────────────")
        hd = all_hetero_data[0]
        for nt in hd.node_types:
            n = hd[nt].x.shape[0] if hasattr(hd[nt], "x") else 0
            feat_dim = hd[nt].x.shape[1] if hasattr(hd[nt], "x") and n > 0 else 0
            print(f"    {nt:<20}: {n} nodes, {feat_dim}-dim features")
        for et in hd.edge_types:
            ei = hd[et].edge_index
            print(f"    {str(et):<45}: {ei.shape[1]} edges")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_smoke_test()
