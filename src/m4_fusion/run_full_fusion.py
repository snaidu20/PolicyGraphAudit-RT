"""
run_full_fusion.py — Full M4 fusion pipeline across all 268 successfully-fetched apps.

Steps
-----
1. Load all apps with status='ok' in data/interim/policies_index.json.
2. For each app: build M2+M3 fused graph, compute discrepancy labels,
   convert to PyG HeteroData.
3. Save HeteroData list to data/processed/fused_graphs_full.pt.
4. Save combined discrepancy labels to data/processed/discrepancy_labels_full.parquet.
5. Print final M5 training dataset stats.

Runtime budget: <30 minutes.  Incremental save every 50 apps to avoid OOM.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

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
log = logging.getLogger("m4_full_fusion")
log.setLevel(logging.INFO)

_DATA_ROOT = _ROOT / "data"
_PROCESSED_DIR = _DATA_ROOT / "processed"
_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
_INTERIM_DIR = _DATA_ROOT / "interim"

FUSED_GRAPHS_PATH = _PROCESSED_DIR / "fused_graphs_full.pt"
DISC_LABELS_PATH  = _PROCESSED_DIR / "discrepancy_labels_full.parquet"
POLICIES_INDEX    = _INTERIM_DIR / "policies_index.json"

BATCH_SIZE = 50          # save incrementally every N apps
TIMEOUT_SECONDS = 1700   # ~28 minutes hard stop


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


def _load_policies_index() -> dict:
    import json
    with open(POLICIES_INDEX, encoding="utf-8") as fh:
        return json.load(fh)


def _load_policy_text(app_id: str, index: dict) -> Optional[str]:
    """Load cached policy text from disk."""
    entry = index.get(app_id)
    if not entry or entry.get("status") != "ok":
        return None
    cached_path = entry.get("cached_path", "")
    if not cached_path:
        return None
    p = Path(cached_path)
    if not p.exists():
        # Try relative to data root
        p = _DATA_ROOT / cached_path
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_full_fusion() -> None:
    t0 = time.time()
    fail_log: List[dict] = []

    # ------------------------------------------------------------------
    # Step 1: Identify all apps with successful policy fetch
    # ------------------------------------------------------------------
    _print_section("Step 1: Loading policy index")
    index = _load_policies_index()
    ok_apps = [
        app_id for app_id, v in index.items()
        if isinstance(v, dict) and v.get("status") == "ok"
    ]
    print(f"  Total apps in index  : {len(index)}")
    print(f"  Apps with status=ok  : {len(ok_apps)}")

    # ------------------------------------------------------------------
    # Step 2: Load Play DSL and filter qualifying apps
    # ------------------------------------------------------------------
    _print_section("Step 2: Loading Play DSL")
    from m3_runtime_graph.load_play_dsl import load_play_dsl, list_unique_apps

    play_df = load_play_dsl()
    qualifying_set = set(list_unique_apps(min_rows=1))  # relaxed from 3 → 1
    print(f"  Qualifying apps (≥1 Play DSL rows): {len(qualifying_set)}")

    # Only process apps that have both a policy AND Play DSL data
    apps_to_fuse = [a for a in ok_apps if a in qualifying_set]
    print(f"  Apps to fuse (policy+DSL)         : {len(apps_to_fuse)}")

    if not apps_to_fuse:
        print("  ERROR: No apps available. Aborting.")
        return

    # ------------------------------------------------------------------
    # Step 3: Load singletons
    # ------------------------------------------------------------------
    _print_section("Step 3: Loading model singletons")
    from m2_policy_graph.classifier import _get_singleton as get_opp115
    from m4_fusion.to_pyg import get_embedder

    print("  Loading OPP-115 classifier ...")
    opp115 = get_opp115()
    print("  Loading sentence-transformer embedder ...")
    embedder = get_embedder()

    # ------------------------------------------------------------------
    # Step 4: Fuse all apps
    # ------------------------------------------------------------------
    _print_section("Step 4: Fusing graphs")

    from m4_fusion.fuse import fuse_app_graphs
    from m4_fusion.discrepancy_labels import (
        compute_discrepancy_labels, ALL_DISCREPANCY_TYPES
    )
    from m4_fusion.to_pyg import nx_to_pyg

    all_hetero_data: List = []
    all_disc_dfs: List[pd.DataFrame] = []
    all_fuse_stats: List[dict] = []
    n_success = 0
    n_fail_fuse = 0
    n_timeout = 0

    with tqdm(total=len(apps_to_fuse), desc="Fusing apps", unit="app") as pbar:
        for i, app_id in enumerate(apps_to_fuse):
            # Hard timeout guard
            elapsed = time.time() - t0
            if elapsed > TIMEOUT_SECONDS:
                log.warning("Reached timeout %.0fs after %d apps — stopping early.", elapsed, i)
                n_timeout = len(apps_to_fuse) - i
                break

            pbar.set_description(f"Fusing {app_id[:35]}")

            policy_text = _load_policy_text(app_id, index)
            if not policy_text or len(policy_text.strip()) < 100:
                log.debug("Skipping %s — no/short policy text", app_id)
                n_fail_fuse += 1
                fail_log.append({"app_id": app_id, "reason": "missing_or_short_policy"})
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
                log.debug("fuse_app_graphs failed for %s: %s", app_id, exc)
                n_fail_fuse += 1
                fail_log.append({"app_id": app_id, "reason": f"fuse_error: {exc!s:.80}"})
                pbar.update(1)
                continue

            if result is None:
                log.debug("fuse_app_graphs returned None for %s", app_id)
                n_fail_fuse += 1
                fail_log.append({"app_id": app_id, "reason": "fuse_returned_none"})
                pbar.update(1)
                continue

            G_fused, fuse_stats = result
            all_fuse_stats.append(fuse_stats)

            # Compute discrepancy labels
            try:
                disc_df = compute_discrepancy_labels(G_fused, app_id)
            except Exception as exc:
                log.debug("discrepancy_labels failed for %s: %s", app_id, exc)
                disc_df = pd.DataFrame()

            all_disc_dfs.append(disc_df)

            # Convert to PyG HeteroData
            try:
                hetero = nx_to_pyg(
                    G_fused,
                    discrepancy_df=disc_df if len(disc_df) > 0 else None,
                    embedder=embedder,
                )
                hetero.app_id = app_id
                all_hetero_data.append(hetero)
                n_success += 1
            except Exception as exc:
                log.debug("nx_to_pyg failed for %s: %s", app_id, exc)
                n_fail_fuse += 1
                fail_log.append({"app_id": app_id, "reason": f"pyg_error: {exc!s:.80}"})

            pbar.update(1)

            # Incremental save every BATCH_SIZE apps
            if (i + 1) % BATCH_SIZE == 0 and all_hetero_data:
                torch.save(all_hetero_data, FUSED_GRAPHS_PATH)
                if all_disc_dfs:
                    combined_tmp = pd.concat(
                        [df for df in all_disc_dfs if len(df) > 0], ignore_index=True
                    )
                    combined_tmp.to_parquet(DISC_LABELS_PATH, index=False)
                log.info("Incremental save: %d graphs, %d label rows", len(all_hetero_data),
                         sum(len(df) for df in all_disc_dfs if len(df) > 0))

    # ------------------------------------------------------------------
    # Step 5: Final save
    # ------------------------------------------------------------------
    _print_section("Step 5: Final save")

    if all_hetero_data:
        torch.save(all_hetero_data, FUSED_GRAPHS_PATH)
        size_mb = FUSED_GRAPHS_PATH.stat().st_size / 1e6
        print(f"  Saved {len(all_hetero_data)} HeteroData objects → {FUSED_GRAPHS_PATH}")
        print(f"  File size: {size_mb:.2f} MB")
    else:
        print("  WARNING: No HeteroData objects to save.")

    combined_disc: Optional[pd.DataFrame] = None
    if all_disc_dfs:
        combined_disc = pd.concat(
            [df for df in all_disc_dfs if len(df) > 0], ignore_index=True
        )
        combined_disc.to_parquet(DISC_LABELS_PATH, index=False)
        disc_size_mb = DISC_LABELS_PATH.stat().st_size / 1e6
        print(f"  Saved {len(combined_disc)} discrepancy label rows → {DISC_LABELS_PATH}")
        print(f"  Parquet size: {disc_size_mb:.3f} MB")

    # ------------------------------------------------------------------
    # Step 6: Statistics report
    # ------------------------------------------------------------------
    _print_section("Step 6: M4 Final Statistics")

    print(f"\n  FUSION RESULTS")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Apps attempted          : {len(apps_to_fuse)}")
    print(f"  Successfully fused      : {n_success}")
    print(f"  Failed during fusion    : {n_fail_fuse}")
    if n_timeout:
        print(f"  Stopped early (timeout) : {n_timeout} apps not processed")
    print(f"  Success rate            : {_fmt_pct(n_success, len(apps_to_fuse))}")

    if all_fuse_stats:
        all_node_counts = [s["n_merged_nodes"] for s in all_fuse_stats]
        all_edge_counts = [s["n_merged_edges"] for s in all_fuse_stats]
        print(f"\n  GRAPH SIZE STATISTICS (over {len(all_fuse_stats)} apps)")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Avg nodes : {sum(all_node_counts)/len(all_node_counts):.1f}")
        print(f"  Avg edges : {sum(all_edge_counts)/len(all_edge_counts):.1f}")
        print(f"  Min/Max nodes : {min(all_node_counts)} / {max(all_node_counts)}")
        print(f"  Min/Max edges : {min(all_edge_counts)} / {max(all_edge_counts)}")

        # Node/edge type breakdowns
        all_nt: Dict[str, List[int]] = defaultdict(list)
        all_et: Dict[str, List[int]] = defaultdict(list)
        for st in all_fuse_stats:
            for nt, cnt in st.get("n_nodes_by_type", {}).items():
                all_nt[nt].append(cnt)
            for et, cnt in st.get("n_edges_by_type", {}).items():
                all_et[et].append(cnt)

        print(f"\n  Avg nodes by type:")
        for nt in sorted(all_nt):
            avg = sum(all_nt[nt]) / len(all_nt[nt])
            print(f"    {nt:<22}: {avg:.1f}")

        print(f"\n  Avg edges by type:")
        for et in sorted(all_et):
            avg = sum(all_et[et]) / len(all_et[et])
            print(f"    {et:<32}: {avg:.1f}")

    if combined_disc is not None and len(combined_disc) > 0:
        print(f"\n  DISCREPANCY LABEL DISTRIBUTION ({len(combined_disc)} total rows)")
        print(f"  ─────────────────────────────────────────────")
        dist = combined_disc["discrepancy_type"].value_counts()
        total = len(combined_disc)
        for disc_type in ALL_DISCREPANCY_TYPES:
            count = dist.get(disc_type, 0)
            print(f"    {disc_type:<30}: {count:>6} ({_fmt_pct(count, total)})")

        print(f"\n  Per-class row counts (M5 training set):")
        print(f"    Total rows: {total}")
        print(f"    Unique apps: {combined_disc['app_id'].nunique()}")
        print(f"    Unique data types: {combined_disc['data_type'].nunique()}")

    # Disk size
    if all_hetero_data:
        size_mb = FUSED_GRAPHS_PATH.stat().st_size / 1e6
        print(f"\n  DISK USAGE")
        print(f"  ─────────────────────────────────────────────")
        print(f"  fused_graphs_full.pt        : {size_mb:.2f} MB")
        if DISC_LABELS_PATH.exists():
            disc_mb = DISC_LABELS_PATH.stat().st_size / 1e6
            print(f"  discrepancy_labels_full.parquet: {disc_mb:.3f} MB")
        total_mb = sum(
            FUSED_GRAPHS_PATH.stat().st_size if FUSED_GRAPHS_PATH.exists() else 0,
            DISC_LABELS_PATH.stat().st_size if DISC_LABELS_PATH.exists() else 0,
        ) / 1e6 if False else size_mb  # simplify

    # Failures list
    if fail_log:
        print(f"\n  FAILED APPS ({len(fail_log)} total):")
        from collections import Counter
        reasons = Counter(f["reason"].split(":")[0] for f in fail_log)
        for reason, count in reasons.most_common():
            print(f"    {reason:<40}: {count}")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_full_fusion()
