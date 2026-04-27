"""
fuse.py — M4 graph fusion: stitches M2 policy graph + M3 label/runtime graph
          into a single per-app heterogeneous MultiDiGraph.

Fusion strategy
---------------
1. Build M2 policy graph from fetched policy text.
2. Build M3 label+runtime graph from Play DSL.
3. Re-ID all nodes using canonical_entities.canonical_id, so that overlapping
   DataType / Purpose / ThirdParty nodes collapse into shared vocabulary nodes.
4. Merge the two node/edge sets into one MultiDiGraph.
5. Add the (App)-[HAS_POLICY]->(Policy) anchor edge tying both subgraphs.
6. Return merged graph + stats.

Public API
----------
fuse_app_graphs(app_id, policy_text, play_df) -> (MultiDiGraph, dict) | None
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import pandas as pd

# Ensure src is importable
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from m4_fusion.canonical_entities import remap_m2_node, remap_m3_node

log = logging.getLogger(__name__)

# Edge type used for the fusion anchor
ET_HAS_POLICY = "HAS_POLICY"


# ---------------------------------------------------------------------------
# Graph re-ID helpers
# ---------------------------------------------------------------------------

def _remap_graph(
    source_graph: nx.MultiDiGraph,
    app_id: str,
    remap_fn,
) -> nx.MultiDiGraph:
    """
    Return a new MultiDiGraph with all node IDs replaced by canonical IDs.

    Also stores 'node_type' and 'canonical_id' attributes on every node.
    """
    G_new = nx.MultiDiGraph()

    # Build old_id -> (canonical_id, node_type) mapping
    id_map: Dict[str, str] = {}
    for old_id, attrs in source_graph.nodes(data=True):
        cid, nt = remap_fn(old_id, app_id)
        id_map[old_id] = cid
        merged_attrs = dict(attrs)
        merged_attrs["node_type"] = nt
        merged_attrs["canonical_id"] = cid
        # Only add if not already present (shared nodes may exist from other graph)
        if not G_new.has_node(cid):
            G_new.add_node(cid, **merged_attrs)

    # Copy edges with remapped IDs
    for u, v, edge_attrs in source_graph.edges(data=True):
        new_u = id_map.get(u)
        new_v = id_map.get(v)
        if new_u and new_v:
            G_new.add_edge(new_u, new_v, **edge_attrs)

    return G_new


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------

def fuse_app_graphs(
    app_id: str,
    policy_text: str,
    play_df: pd.DataFrame,
    opp115_classifier=None,
) -> Optional[Tuple[nx.MultiDiGraph, Dict[str, Any]]]:
    """
    Build and merge the M2 policy graph + M3 label/runtime graph for one app.

    Parameters
    ----------
    app_id : str
        Android package name.
    policy_text : str
        Raw policy text (already stripped of HTML).
    play_df : pd.DataFrame
        Full Play DSL DataFrame (will be filtered to this appId internally).
    opp115_classifier : OPP115Classifier, optional
        Pre-loaded M2 classifier singleton (avoids repeated disk loads).

    Returns
    -------
    (merged_graph, stats) or None if M3 build fails.

    stats keys
    ----------
    n_nodes_by_type, n_edges_by_type, n_collapsed_entities,
    n_m2_nodes, n_m3_nodes, n_merged_nodes, n_merged_edges
    """
    # ------------------------------------------------------------------
    # Lazy imports to avoid circular imports and long load times at module level
    # ------------------------------------------------------------------
    from m2_policy_graph.build_graph import build_policy_graph
    from m3_runtime_graph.build_graph import build_label_runtime_graph

    # ------------------------------------------------------------------
    # Step 1: Build M2 policy graph
    # ------------------------------------------------------------------
    policy_id = f"{app_id}_policy"
    try:
        G_policy, m2_stats = build_policy_graph(
            policy_text, policy_id, classifier=opp115_classifier
        )
    except Exception as exc:
        log.warning("M2 build failed for %s: %s", app_id, exc)
        G_policy = nx.MultiDiGraph()
        m2_stats = {}

    # ------------------------------------------------------------------
    # Step 2: Build M3 label+runtime graph
    # ------------------------------------------------------------------
    m3_result = build_label_runtime_graph(app_id, play_df)
    if m3_result is None:
        log.warning("M3 build returned None for %s", app_id)
        return None
    G_runtime, m3_stats = m3_result

    # ------------------------------------------------------------------
    # Step 3: Re-ID both graphs to canonical IDs
    # ------------------------------------------------------------------
    G_policy_canon = _remap_graph(G_policy, app_id, remap_m2_node)
    G_runtime_canon = _remap_graph(G_runtime, app_id, remap_m3_node)

    n_m2 = G_policy_canon.number_of_nodes()
    n_m3 = G_runtime_canon.number_of_nodes()

    # ------------------------------------------------------------------
    # Step 4: Merge by node union (shared nodes collapse naturally)
    # ------------------------------------------------------------------
    G_merged = nx.MultiDiGraph()

    # Add M3 nodes first (has App/Label anchors)
    for node_id, attrs in G_runtime_canon.nodes(data=True):
        G_merged.add_node(node_id, **attrs)
    for u, v, edge_attrs in G_runtime_canon.edges(data=True):
        G_merged.add_edge(u, v, **edge_attrs)

    # Add M2 nodes — shared DataType/Purpose/ThirdParty nodes already exist
    n_collapsed = 0
    for node_id, attrs in G_policy_canon.nodes(data=True):
        if G_merged.has_node(node_id):
            n_collapsed += 1
            # Merge attributes: update existing node with any new keys from M2
            existing = G_merged.nodes[node_id]
            for k, v in attrs.items():
                if k not in existing:
                    existing[k] = v
        else:
            G_merged.add_node(node_id, **attrs)

    for u, v, edge_attrs in G_policy_canon.edges(data=True):
        G_merged.add_edge(u, v, **edge_attrs)

    # ------------------------------------------------------------------
    # Step 5: Add HAS_POLICY anchor edge  (App) -> (Policy)
    # ------------------------------------------------------------------
    app_node_id = f"App::{app_id}"
    policy_node_id = f"Policy::{app_id}_policy::{app_id}_policy"

    # Ensure the Policy node exists even if M2 produced an empty graph
    if not G_merged.has_node(policy_node_id):
        G_merged.add_node(
            policy_node_id,
            node_type="Policy",
            canonical_id=policy_node_id,
            doc_id=policy_id,
            length_chars=len(policy_text),
        )

    if G_merged.has_node(app_node_id) and G_merged.has_node(policy_node_id):
        G_merged.add_edge(app_node_id, policy_node_id, edge_type=ET_HAS_POLICY)

    # ------------------------------------------------------------------
    # Step 6: Compute stats
    # ------------------------------------------------------------------
    node_type_counts = Counter(
        d.get("node_type", "Unknown") for _, d in G_merged.nodes(data=True)
    )
    edge_type_counts = Counter(
        d.get("edge_type", "Unknown") for _, _, d in G_merged.edges(data=True)
    )

    stats: Dict[str, Any] = {
        "app_id": app_id,
        "n_m2_nodes": n_m2,
        "n_m3_nodes": n_m3,
        "n_merged_nodes": G_merged.number_of_nodes(),
        "n_merged_edges": G_merged.number_of_edges(),
        "n_collapsed_entities": n_collapsed,
        "n_nodes_by_type": dict(node_type_counts),
        "n_edges_by_type": dict(edge_type_counts),
        "m2_stats": m2_stats,
        "m3_stats": {k: v for k, v in m3_stats.items() if not isinstance(v, pd.DataFrame)},
    }

    log.info(
        "Fused %s: %d nodes (%d collapsed), %d edges",
        app_id,
        G_merged.number_of_nodes(),
        n_collapsed,
        G_merged.number_of_edges(),
    )
    return G_merged, stats
