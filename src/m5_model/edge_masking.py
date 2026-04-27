"""
edge_masking.py — Label-determining edge masking for M5 non-circular evaluation.

MOTIVATION
----------
The M4 fusion pipeline computes discrepancy labels (CONSISTENT, UNDECLARED_COLLECTION,
POLICY_LABEL_MISMATCH, OVER_DISCLOSURE) from three structural edge types:

    (PrivacyLabel) -[DECLARES_COLLECTS]-> (DataType)
    (PrivacyLabel) -[DECLARES_SHARES]->   (DataType)
    (SDK)          -[COLLECTS_DATATYPE]-> (DataType)

These are the *exact* edges that determine the label.  If the GNN sees all of
them during training, it can reconstruct the rule deterministically → macro F1
= 1.0000.  This is methodological circularity, not genuine generalisation.

FIX (Setup A — recommended)
----------------------------
- Train:  30 % of label-determining edges masked per (App, DataType) pair.
- Test:   SAME 30 % masked using the same deterministic per-pair seed so the
          evaluation is on realistically-incomplete graphs.
- Ground truth labels: always the ORIGINAL M4 labels (unmasked).
          The model must infer the label from policy text + remaining structure.

Policy-side edges [(PolicySegment)-[MENTIONS]->(DataType)] are deliberately
KEPT — these are the signal source the model should leverage.

Public API
----------
mask_label_determining_edges(graph, mask_prob, seed) -> (HeteroData, dict)
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData

# Edge types whose *target* DataType nodes determine the discrepancy label.
# Defined by the rule in discrepancy_labels.py:
#   DECLARES_COLLECTS  → label_collects flag
#   DECLARES_SHARES    → label_shares flag
#   COLLECTS_DATATYPE  → runtime_implies flag
# MENTIONS is intentionally excluded — it is the POLICY side we want to keep.
LABEL_DETERMINING_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("PrivacyLabel", "DECLARES_COLLECTS", "DataType"),
    ("PrivacyLabel", "DECLARES_SHARES",   "DataType"),
    ("SDK",          "COLLECTS_DATATYPE", "DataType"),
]


def _deterministic_mask(pair_key: str, edge_type: str, mask_prob: float, seed: int) -> bool:
    """
    Deterministically decide whether to mask a specific (pair, edge_type) combination.
    Returns True if the edge should be MASKED (removed).

    Uses a seeded hash so the mask is identical across train / test invocations.
    """
    h = hash((pair_key, edge_type, seed)) % 10_000
    return (h / 10_000) < mask_prob


def mask_label_determining_edges(
    graph: HeteroData,
    mask_prob: float = 0.30,
    seed: int = 42,
) -> Tuple[HeteroData, dict]:
    """
    Mask label-determining edges in a per-app HeteroData graph.

    For each (App, DataType) labeled pair:
    - With probability ``mask_prob`` (determined deterministically by seed),
      removes DECLARES_COLLECTS, DECLARES_SHARES, and COLLECTS_DATATYPE edges
      pointing to that DataType from their respective source node types.

    CRITICALLY: Policy-side MENTIONS edges are NOT masked.  The model must
    predict from text evidence + remaining structural context.

    Parameters
    ----------
    graph : HeteroData
        A per-app fused heterogeneous graph from M4.  Must have
        ``graph.discrepancy_pairs`` (tensor [n_pairs, 2]) and
        ``graph.discrepancy_labels`` (tensor [n_pairs]).
    mask_prob : float
        Per-pair, per-edge-type masking probability in [0, 1].
        0.30 is the primary experimental setting.
    seed : int
        Global seed for deterministic masking.  MUST be the same value
        at train and test time so the evaluation is on the same masked graph.

    Returns
    -------
    masked_graph : HeteroData
        A deep copy of ``graph`` with the selected edges removed.
        ``discrepancy_labels`` and ``discrepancy_pairs`` are unchanged —
        the original M4 labels are always the ground truth.
    stats : dict
        {
          "n_pairs_with_masking": int,   # pairs where ≥1 edge was removed
          "n_edges_removed": int,        # total edges removed across all types
          "masked_pairs": list[int],     # pair indices (into discrepancy_pairs)
          "mask_log": list[dict],        # per-pair detail for inspection
        }
    """
    masked_graph = copy.deepcopy(graph)

    n_pairs = masked_graph.discrepancy_pairs.shape[0] if (
        hasattr(masked_graph, "discrepancy_pairs") and
        masked_graph.discrepancy_pairs.numel() > 0
    ) else 0

    if n_pairs == 0:
        return masked_graph, {
            "n_pairs_with_masking": 0,
            "n_edges_removed": 0,
            "masked_pairs": [],
            "mask_log": [],
        }

    # Get DataType local indices for each pair
    pairs_np = masked_graph.discrepancy_pairs.numpy()   # [n_pairs, 2]
    dt_locals = pairs_np[:, 1]  # DataType local index per pair

    # Build a canonical pair key using the graph's DataType node IDs if available
    dt_node_ids: Optional[List[str]] = None
    if "DataType" in masked_graph.node_types:
        dt_store = masked_graph["DataType"]
        if hasattr(dt_store, "node_ids"):
            dt_node_ids = dt_store.node_ids

    # -------------------------------------------------------------------
    # For each edge type we might mask, build a DataType→[src_indices] map
    # -------------------------------------------------------------------
    # edge_index shape: [2, n_edges]  where row 0 = src, row 1 = dst (DataType)
    n_edges_removed = 0
    masked_pair_set = set()
    mask_log = []

    for src_type, rel, dst_type in LABEL_DETERMINING_EDGE_TYPES:
        # Skip if this edge type is absent in the graph
        edge_key = (src_type, rel, dst_type)
        if edge_key not in masked_graph.edge_types:
            continue

        ei: torch.Tensor = masked_graph[src_type, rel, dst_type].edge_index  # [2, E]
        if ei.shape[1] == 0:
            continue

        ei_np = ei.numpy()  # shape [2, E]
        dst_indices = ei_np[1]  # DataType local indices per edge

        # Determine which DataType local indices should be masked for this edge type
        keep_mask = np.ones(ei_np.shape[1], dtype=bool)

        for pair_idx, dt_local in enumerate(dt_locals):
            # Build a deterministic key for this (app, datatype, edge_type) triple
            if dt_node_ids is not None and dt_local < len(dt_node_ids):
                pair_key = dt_node_ids[dt_local]
            else:
                # Fallback: use graph app id + datatype local index
                app_ids = getattr(masked_graph.get("App", {}), "node_ids", None)
                app_str = app_ids[0] if (app_ids and len(app_ids) > 0) else "unknown"
                pair_key = f"{app_str}::dt{dt_local}"

            should_mask = _deterministic_mask(pair_key, rel, mask_prob, seed)
            if not should_mask:
                continue

            # Find all edges in this type that point to dt_local
            edge_positions = np.where(dst_indices == dt_local)[0]
            if len(edge_positions) == 0:
                continue

            keep_mask[edge_positions] = False
            masked_pair_set.add(pair_idx)
            n_edges_removed += len(edge_positions)
            mask_log.append({
                "pair_idx": int(pair_idx),
                "dt_local": int(dt_local),
                "edge_type": rel,
                "n_edges_removed": len(edge_positions),
            })

        # Apply the keep mask — filter edge_index
        kept_indices = np.where(keep_mask)[0]
        new_ei = torch.from_numpy(ei_np[:, kept_indices])
        masked_graph[src_type, rel, dst_type].edge_index = new_ei

        # Also handle reverse edges if they exist (rev_ prefix convention from model.py)
        rev_key = (dst_type, f"rev_{rel}", src_type)
        if rev_key in masked_graph.edge_types:
            rev_ei: torch.Tensor = masked_graph[rev_key].edge_index
            rev_np = rev_ei.numpy()
            # For reverse, row 0 = dst (DataType), row 1 = src
            dst_in_rev = rev_np[0]
            rev_keep = keep_mask  # same positions map because reverse was built by flip
            # Rebuild: keep only positions where the forward edge was kept
            rev_kept = torch.from_numpy(rev_np[:, kept_indices])
            masked_graph[rev_key].edge_index = rev_kept

    stats = {
        "n_pairs_with_masking": len(masked_pair_set),
        "n_edges_removed": n_edges_removed,
        "masked_pairs": sorted(masked_pair_set),
        "mask_log": mask_log,
    }
    return masked_graph, stats


def apply_masking_to_graphs(
    graphs: List[HeteroData],
    mask_prob: float = 0.30,
    seed: int = 42,
) -> Tuple[List[HeteroData], dict]:
    """
    Apply mask_label_determining_edges to a list of graphs.

    Parameters
    ----------
    graphs : list of HeteroData
    mask_prob : float
    seed : int

    Returns
    -------
    masked_graphs : list[HeteroData]
    summary : dict with aggregate stats
    """
    masked_graphs = []
    total_pairs_masked = 0
    total_edges_removed = 0

    for g in graphs:
        mg, stats = mask_label_determining_edges(g, mask_prob=mask_prob, seed=seed)
        masked_graphs.append(mg)
        total_pairs_masked += stats["n_pairs_with_masking"]
        total_edges_removed += stats["n_edges_removed"]

    summary = {
        "n_graphs": len(graphs),
        "mask_prob": mask_prob,
        "seed": seed,
        "total_pairs_masked": total_pairs_masked,
        "total_edges_removed": total_edges_removed,
    }
    return masked_graphs, summary
