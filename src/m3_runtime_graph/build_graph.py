"""
build_graph.py — Per-app Privacy Label + Runtime Evidence graph builder (M3).

Graph schema (per configs/schema.yaml):
  Nodes:   App, PrivacyLabel, DataType, Purpose, SDK, ThirdParty
  Edges:
    (App)          -[HAS_LABEL]->          (PrivacyLabel)
    (PrivacyLabel) -[DECLARES_COLLECTS]->  (DataType)
    (PrivacyLabel) -[DECLARES_SHARES]->    (DataType)
    (PrivacyLabel) -[DECLARES_PURPOSE]->   (Purpose)
    (App)          -[CONTAINS_SDK]->       (SDK)
    (SDK)          -[COLLECTS_DATATYPE]->  (DataType)
    (SDK)          -[OPERATED_BY]->        (ThirdParty)

Public API
----------
build_label_runtime_graph(appId, play_df) -> tuple[MultiDiGraph, dict] | None
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import pandas as pd

from m3_runtime_graph.vocab_map import map_data_type, map_purpose, _DATATYPE_SENTINELS
from m3_runtime_graph.trackers import infer_sdks_for_app

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node / edge type constants (match schema.yaml exactly)
# ---------------------------------------------------------------------------

NT_APP          = "App"
NT_LABEL        = "PrivacyLabel"
NT_DATA_TYPE    = "DataType"
NT_PURPOSE      = "Purpose"
NT_SDK          = "SDK"
NT_THIRD_PARTY  = "ThirdParty"

ET_HAS_LABEL          = "HAS_LABEL"
ET_DECLARES_COLLECTS  = "DECLARES_COLLECTS"
ET_DECLARES_SHARES    = "DECLARES_SHARES"
ET_DECLARES_PURPOSE   = "DECLARES_PURPOSE"
ET_CONTAINS_SDK       = "CONTAINS_SDK"
ET_COLLECTS_DATATYPE  = "COLLECTS_DATATYPE"
ET_OPERATED_BY        = "OPERATED_BY"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_bucket(real_installs: float) -> str:
    """Classify raw realInstalls count into logarithmic bucket string."""
    n = int(real_installs)
    if n < 100:
        return "<100"
    if n < 1_000:
        return "100-999"
    if n < 10_000:
        return "1K-9.9K"
    if n < 100_000:
        return "10K-99K"
    if n < 1_000_000:
        return "100K-999K"
    if n < 10_000_000:
        return "1M-9.9M"
    return "10M+"


def _add_node_once(
    G: nx.MultiDiGraph,
    node_id: str,
    **attrs: Any,
) -> None:
    """Add node only if not already present (idempotent)."""
    if not G.has_node(node_id):
        G.add_node(node_id, **attrs)


# Per-graph edge deduplication set — reset inside build_label_runtime_graph
_EDGE_SEEN: set[tuple[str, str, str]] = set()


def _add_edge_once(
    G: nx.MultiDiGraph,
    u: str,
    v: str,
    edge_type: str,
) -> bool:
    """Add a single directed edge; skip if identical (u, v, edge_type) exists."""
    key = (u, v, edge_type)
    if key in _EDGE_SEEN:
        return False
    _EDGE_SEEN.add(key)
    G.add_edge(u, v, edge_type=edge_type)
    return True


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_label_runtime_graph(
    appId: str,
    play_df: pd.DataFrame,
) -> Optional[Tuple[nx.MultiDiGraph, Dict[str, Any]]]:
    """
    Build a typed MultiDiGraph for the Privacy Label + Runtime Evidence side
    of a single Android app.

    Parameters
    ----------
    appId : str
        Android package name (e.g. 'com.example.myapp').
    play_df : pd.DataFrame
        Full or pre-filtered Play DSL DataFrame (load_play_dsl output).
        Will be filtered to rows matching appId.

    Returns
    -------
    (G, stats) if data exists for the app, else None.

    stats dict keys
    ---------------
    n_app, n_label_decl_collects, n_label_decl_shares, n_label_purposes,
    n_inferred_sdks, n_data_types, n_nodes, n_edges,
    privacy_policy_url, developer, title, genreId, installs_bucket
    """
    global _EDGE_SEEN
    _EDGE_SEEN = set()  # reset per-app deduplication set

    app_df = play_df[play_df["appId"] == appId].copy()
    if app_df.empty:
        log.warning("No Play DSL rows for appId=%s", appId)
        return None

    G = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Extract app metadata from first row
    # ------------------------------------------------------------------
    meta = app_df.iloc[0]
    title      = str(meta.get("appTitle", ""))
    developer  = str(meta.get("developer", ""))
    genre_id   = str(meta.get("genreId", ""))
    privacy_url = str(meta.get("privacyPolicy", ""))
    real_installs = float(meta.get("realInstalls", 0) or 0)
    inst_bucket = _install_bucket(real_installs)

    # ------------------------------------------------------------------
    # App node
    # ------------------------------------------------------------------
    app_node = f"app:{appId}"
    _add_node_once(
        G, app_node,
        node_type=NT_APP,
        package_name=appId,
        title=title,
        developer=developer,
        genreId=genre_id,
        installs_bucket=inst_bucket,
    )

    # ------------------------------------------------------------------
    # PrivacyLabel node
    # ------------------------------------------------------------------
    label_node = f"label:{appId}"
    _add_node_once(
        G, label_node,
        node_type=NT_LABEL,
        platform="google_play",
        app_id=appId,
    )
    _add_edge_once(G, app_node, label_node, ET_HAS_LABEL)

    # ------------------------------------------------------------------
    # DataType and Purpose nodes from label rows
    # ------------------------------------------------------------------
    n_decl_collects = 0
    n_decl_shares   = 0
    n_purposes      = 0
    seen_dt_collect: set[str] = set()
    seen_dt_share:   set[str] = set()
    seen_purposes:   set[str] = set()
    declared_canonical_purposes: set[str] = set()

    # Only process actual data rows (not security practice rows)
    data_rows = app_df[app_df["type"].isin(["dataCollected", "dataShared"])]

    for _, row in data_rows.iterrows():
        raw_dt   = str(row.get("dataType", "") or "")
        raw_type = str(row.get("type", "") or "")
        raw_pu   = str(row.get("purpose", "") or "")

        # Skip sentinel values
        if raw_dt in _DATATYPE_SENTINELS:
            continue

        canonical_dt = map_data_type(raw_dt)

        # DataType node + label edge
        if canonical_dt:
            dt_node = f"datatype:{canonical_dt}"
            _add_node_once(G, dt_node, node_type=NT_DATA_TYPE, name=canonical_dt)

            if raw_type == "dataCollected" and canonical_dt not in seen_dt_collect:
                seen_dt_collect.add(canonical_dt)
                _add_edge_once(G, label_node, dt_node, ET_DECLARES_COLLECTS)
                n_decl_collects += 1

            elif raw_type == "dataShared" and canonical_dt not in seen_dt_share:
                seen_dt_share.add(canonical_dt)
                _add_edge_once(G, label_node, dt_node, ET_DECLARES_SHARES)
                n_decl_shares += 1

        # Purpose node + label edge
        if raw_pu and raw_pu not in ("N/A", "No data", "nan"):
            canonical_pu = map_purpose(raw_pu)
            if canonical_pu and canonical_pu not in seen_purposes:
                seen_purposes.add(canonical_pu)
                declared_canonical_purposes.add(canonical_pu)
                pu_node = f"purpose:{canonical_pu}"
                _add_node_once(G, pu_node, node_type=NT_PURPOSE, name=canonical_pu)
                _add_edge_once(G, label_node, pu_node, ET_DECLARES_PURPOSE)
                n_purposes += 1

    # ------------------------------------------------------------------
    # Inferred SDKs via category-level priors
    # ------------------------------------------------------------------
    inferred_sdks = infer_sdks_for_app(genre_id, declared_canonical_purposes)
    n_inferred = len(inferred_sdks)

    for sdk in inferred_sdks:
        sdk_node = f"sdk:{sdk['tracker_id']}"
        _add_node_once(
            G, sdk_node,
            node_type=NT_SDK,
            tracker_id=sdk["tracker_id"],
            name=sdk["name"],
            category=sdk["category"],
            owner=sdk["owner_company"],
            jurisdiction=sdk["jurisdiction"],
            inferred=True,
        )
        _add_edge_once(G, app_node, sdk_node, ET_CONTAINS_SDK)

        # SDK → DataType edges (from Yale cross-reference)
        for dt_canon in sdk.get("collects_data_types", []):
            dt_node = f"datatype:{dt_canon}"
            _add_node_once(G, dt_node, node_type=NT_DATA_TYPE, name=dt_canon)
            _add_edge_once(G, sdk_node, dt_node, ET_COLLECTS_DATATYPE)

        # SDK → ThirdParty (owner company)
        owner = sdk["owner_company"].strip()
        if owner:
            tp_key  = owner.lower().replace(" ", "_")
            tp_node = f"thirdparty:{tp_key}"
            _add_node_once(
                G, tp_node,
                node_type=NT_THIRD_PARTY,
                name=owner,
                jurisdiction=sdk["jurisdiction"],
            )
            _add_edge_once(G, sdk_node, tp_node, ET_OPERATED_BY)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    n_data_types = len(
        [n for n, d in G.nodes(data=True) if d.get("node_type") == NT_DATA_TYPE]
    )

    stats: Dict[str, Any] = {
        "n_app":                 1,
        "n_label_decl_collects": n_decl_collects,
        "n_label_decl_shares":   n_decl_shares,
        "n_label_purposes":      n_purposes,
        "n_inferred_sdks":       n_inferred,
        "n_data_types":          n_data_types,
        "n_nodes":               G.number_of_nodes(),
        "n_edges":               G.number_of_edges(),
        "privacy_policy_url":    privacy_url,
        "developer":             developer,
        "title":                 title,
        "genreId":               genre_id,
        "installs_bucket":       inst_bucket,
    }

    return G, stats


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from m3_runtime_graph.load_play_dsl import load_play_dsl

    df = load_play_dsl(max_apps=5)
    sample_id = df["appId"].iloc[0]
    print(f"Building graph for {sample_id} ...")
    result = build_label_runtime_graph(sample_id, df)
    if result is None:
        print("No data.")
        sys.exit(1)

    G, stats = result
    print("Stats:", stats)
    print("Node type breakdown:")
    from collections import Counter
    ct = Counter(d["node_type"] for _, d in G.nodes(data=True))
    for nt, cnt in ct.items():
        print(f"  {nt}: {cnt}")
    print("Edge type breakdown:")
    et = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    for et_name, cnt in et.items():
        print(f"  {et_name}: {cnt}")
