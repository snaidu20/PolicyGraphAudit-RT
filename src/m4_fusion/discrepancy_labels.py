"""
discrepancy_labels.py — Rule-derived weak-supervision labels for M5 training.

Computes per-(appId, dataType) discrepancy labels by traversing the fused
heterogeneous graph produced by M4.

IMPORTANT: These labels are *rule-derived* (i.e., weak supervision), not
human-annotated ground truth.  They encode four logical conditions comparing
what an app's privacy label declares vs. what the policy text mentions vs.
what inferred SDKs imply about data collection.  Treat label noise (especially
for OVER_DISCLOSURE and POLICY_LABEL_MISMATCH) as expected in a weakly
supervised setting.

Discrepancy types (from configs/schema.yaml)
--------------------------------------------
UNDECLARED_COLLECTION
    SDK implies collection of this data type, but neither the label nor the
    policy mentions it.  Indicates potential hidden data harvesting.

POLICY_LABEL_MISMATCH
    Label declares collection XOR policy mentions collection — one source says
    yes, the other says nothing.  Indicates inconsistency between regulatory
    disclosures.

OVER_DISCLOSURE
    Policy mentions the data type but the label does not declare collection and
    no SDK implies collection.  Less severe — may be boilerplate or future-
    proofing language in the policy.

CONSISTENT
    Both the label declares collection AND the policy mentions it.  No
    discrepancy detected.

Public API
----------
compute_discrepancy_labels(graph, app_id) -> pd.DataFrame
save_discrepancy_labels(df, path) -> None
load_discrepancy_labels(path) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discrepancy class constants
# ---------------------------------------------------------------------------

UNDECLARED_COLLECTION = "UNDECLARED_COLLECTION"
POLICY_LABEL_MISMATCH = "POLICY_LABEL_MISMATCH"
OVER_DISCLOSURE = "OVER_DISCLOSURE"
CONSISTENT = "CONSISTENT"

ALL_DISCREPANCY_TYPES = [
    CONSISTENT,
    UNDECLARED_COLLECTION,
    POLICY_LABEL_MISMATCH,
    OVER_DISCLOSURE,
]

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_output_path() -> Path:
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent.parent / "data" / "processed" / "discrepancy_labels.parquet",
        Path("data") / "processed" / "discrepancy_labels.parquet",
    ]:
        if candidate.parent.exists():
            return candidate
    return Path("data/processed/discrepancy_labels.parquet")


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

def _collect_edges_by_type(
    G: nx.MultiDiGraph,
    edge_type: str,
) -> set[tuple[str, str]]:
    """Return set of (src, dst) tuples for edges of the given type."""
    return {
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == edge_type
    }


def _get_nodes_by_type(G: nx.MultiDiGraph, node_type: str) -> set[str]:
    """Return set of node IDs with the given node_type attribute."""
    return {
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == node_type
    }


# ---------------------------------------------------------------------------
# Main label computation
# ---------------------------------------------------------------------------

def compute_discrepancy_labels(
    graph: nx.MultiDiGraph,
    app_id: str,
) -> pd.DataFrame:
    """
    Compute rule-derived discrepancy labels for all DataType nodes reachable
    from this app in the fused graph.

    NOTE: These are *rule-derived weak supervision labels*, not human
    annotations.  Noise is expected, particularly for OVER_DISCLOSURE (policy
    boilerplate) and POLICY_LABEL_MISMATCH (vocabulary mapping gaps).

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Fused per-app graph from fuse_app_graphs().
    app_id : str
        Android package name for this app.

    Returns
    -------
    pd.DataFrame with columns:
        app_id, data_type, label_collects, label_shares,
        policy_mentions, runtime_implies,
        discrepancy_type
    One row per (app_id, data_type) pair.
    """
    G = graph

    # ------------------------------------------------------------------
    # Build edge lookup sets (all edge types we need)
    # ------------------------------------------------------------------
    # (PrivacyLabel) -[DECLARES_COLLECTS]-> (DataType)
    label_collects_edges = _collect_edges_by_type(G, "DECLARES_COLLECTS")
    # (PrivacyLabel) -[DECLARES_SHARES]-> (DataType)
    label_shares_edges = _collect_edges_by_type(G, "DECLARES_SHARES")
    # (PolicySegment) -[MENTIONS]-> (DataType)
    policy_mentions_edges = _collect_edges_by_type(G, "MENTIONS")
    # (SDK) -[COLLECTS_DATATYPE]-> (DataType)
    sdk_collects_edges = _collect_edges_by_type(G, "COLLECTS_DATATYPE")
    # (App) -[CONTAINS_SDK]-> (SDK)
    app_sdks_edges = _collect_edges_by_type(G, "CONTAINS_SDK")

    # Canonical app node ID
    app_node = f"App::{app_id}"

    # ------------------------------------------------------------------
    # Collect all DataType nodes in the graph
    # ------------------------------------------------------------------
    all_data_type_nodes = _get_nodes_by_type(G, "DataType")

    # Determine which DataType nodes are reachable from this app:
    # Any DataType that has at least one relevant edge pointing to it
    # (from label, policy, or SDK for this app's subgraph)
    label_nodes = {v for u, v in label_collects_edges | label_shares_edges}
    policy_dt_nodes = {v for u, v in policy_mentions_edges}
    # SDKs belonging to this app
    this_app_sdk_nodes = {v for u, v in app_sdks_edges if u == app_node}
    sdk_dt_nodes = {
        v for u, v in sdk_collects_edges if u in this_app_sdk_nodes
    }

    # Union of all reachable data types
    reachable_dt = (
        (all_data_type_nodes & (label_nodes | policy_dt_nodes | sdk_dt_nodes))
    )

    if not reachable_dt:
        log.debug("No reachable DataType nodes for app %s", app_id)
        return pd.DataFrame(columns=[
            "app_id", "data_type",
            "label_collects", "label_shares", "policy_mentions", "runtime_implies",
            "discrepancy_type",
        ])

    # ------------------------------------------------------------------
    # Build sets for fast lookup
    # ------------------------------------------------------------------
    # DataTypes declared as collected
    label_collects_dt = {v for _, v in label_collects_edges}
    # DataTypes declared as shared
    label_shares_dt = {v for _, v in label_shares_edges}
    # DataTypes mentioned in policy
    policy_mentions_dt = {v for _, v in policy_mentions_edges}
    # DataTypes implied by SDKs belonging to this app
    runtime_implies_dt = sdk_dt_nodes

    # ------------------------------------------------------------------
    # Classify each DataType
    # ------------------------------------------------------------------
    rows = []
    for dt_node in sorted(reachable_dt):
        # Get canonical data type name (strip prefix for readability)
        dt_attrs = G.nodes.get(dt_node, {})
        dt_name = dt_attrs.get("name", dt_node.split("::", 1)[-1] if "::" in dt_node else dt_node)

        lc = dt_node in label_collects_dt
        ls = dt_node in label_shares_dt
        pm = dt_node in policy_mentions_dt
        ri = dt_node in runtime_implies_dt

        # Apply discrepancy logic (rule-derived weak supervision).
        # Order matters: check most specific conditions first.
        if lc and pm:
            disc_type = CONSISTENT
        elif ri and not lc and not pm:
            disc_type = UNDECLARED_COLLECTION
        elif pm and not lc and not ri:
            # Policy mentions the data type, but neither label nor SDK implies it
            disc_type = OVER_DISCLOSURE
        elif lc and not pm:
            # Label declares collection but policy is silent → mismatch
            disc_type = POLICY_LABEL_MISMATCH
        else:
            # Residual cases (e.g. only label_shares but no other signal)
            disc_type = CONSISTENT

        rows.append({
            "app_id": app_id,
            "data_type": dt_name,
            "data_type_node": dt_node,
            "label_collects": lc,
            "label_shares": ls,
            "policy_mentions": pm,
            "runtime_implies": ri,
            "discrepancy_type": disc_type,
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_discrepancy_labels(
    df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Save discrepancy labels DataFrame to Parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Combined labels from all apps (concatenate per-app DataFrames).
    path : Path, optional
        Output path.  Defaults to data/processed/discrepancy_labels.parquet.

    Returns
    -------
    Path where the file was saved.
    """
    out_path = Path(path) if path else _resolve_output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Saved %d discrepancy label rows to %s", len(df), out_path)
    return out_path


def load_discrepancy_labels(path: Optional[Path] = None) -> pd.DataFrame:
    """Load previously saved discrepancy labels from Parquet."""
    load_path = Path(path) if path else _resolve_output_path()
    if not load_path.exists():
        raise FileNotFoundError(f"Discrepancy labels not found at {load_path}")
    return pd.read_parquet(load_path)
