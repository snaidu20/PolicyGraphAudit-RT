"""
to_pyg.py — Convert a fused NetworkX MultiDiGraph to a PyTorch Geometric
            HeteroData object ready for M5 GNN training.

Approach
--------
Node features
~~~~~~~~~~~~~
- DataType, Purpose, ThirdParty, PolicySegment:
    sentence-transformer embedding of node ``name`` or ``text_snippet``
    (model: all-MiniLM-L6-v2, 384-dim float32).
- App:
    One-hot encoding of ``genreId`` (Google Play category).
- Policy, PrivacyLabel, SDK, Endpoint:
    Zero-vector placeholder (384-dim) — structural nodes without rich text.

Edge indices
~~~~~~~~~~~~
For every (src_type, edge_type, dst_type) triple in the schema, we collect
the (src_local_idx, dst_local_idx) pairs and store as ``edge_index`` [2, E].

Discrepancy labels
~~~~~~~~~~~~~~~~~~
``data['App'].discrepancy_labels`` is a dense int64 tensor of shape
[n_apps, n_data_types, 4] where the last dimension encodes a one-hot
discrepancy class (CONSISTENT / UNDECLARED_COLLECTION / POLICY_LABEL_MISMATCH
/ OVER_DISCLOSURE).  These are the M5 training targets.

Public API
----------
nx_to_pyg(merged_graph, discrepancy_df=None, embedder=None) -> HeteroData
get_embedder() -> SentenceTransformer  (cached singleton)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_NODE_TYPES = [
    "App", "Policy", "PolicySegment", "DataType", "Purpose",
    "ThirdParty", "PrivacyLabel", "SDK", "Endpoint",
]

# (src_type, edge_type, dst_type) — canonical schema edge triples
SCHEMA_EDGE_TRIPLES = [
    ("Policy",        "HAS_SEGMENT",        "PolicySegment"),
    ("PolicySegment", "MENTIONS",            "DataType"),
    ("PolicySegment", "FOR_PURPOSE",         "Purpose"),
    ("PolicySegment", "SHARED_WITH",         "ThirdParty"),
    ("PrivacyLabel",  "DECLARES_COLLECTS",   "DataType"),
    ("PrivacyLabel",  "DECLARES_PURPOSE",    "Purpose"),
    ("PrivacyLabel",  "DECLARES_SHARES",     "DataType"),
    ("SDK",           "COLLECTS_DATATYPE",   "DataType"),
    ("SDK",           "OPERATED_BY",         "ThirdParty"),
    ("App",           "CONTAINS_SDK",        "SDK"),
    ("App",           "REACHES_ENDPOINT",    "Endpoint"),
    ("Endpoint",      "RESOLVES_TO",         "ThirdParty"),
    ("App",           "HAS_POLICY",          "Policy"),
    ("App",           "HAS_LABEL",           "PrivacyLabel"),
]

# Embed dim for sentence-transformer model
_EMBED_DIM = 384

# One-hot categories for App.genreId
_GENRE_CATEGORIES = [
    "GAME", "FAMILY", "TOOLS", "LIFESTYLE", "EDUCATION", "HEALTH_AND_FITNESS",
    "MEDICAL", "FINANCE", "BUSINESS", "SHOPPING", "SOCIAL", "COMMUNICATION",
    "ENTERTAINMENT", "PHOTOGRAPHY", "MAPS_AND_NAVIGATION", "TRAVEL_AND_LOCAL",
    "NEWS_AND_MAGAZINES", "FOOD_AND_DRINK", "SPORTS", "PRODUCTIVITY",
    "MUSIC_AND_AUDIO", "BOOKS_AND_REFERENCE", "AUTO_AND_VEHICLES",
    "HOUSE_AND_HOME", "EVENTS", "BEAUTY", "VIDEO_PLAYERS", "DATING",
    "PARENTING", "ART_AND_DESIGN", "COMICS", "WEATHER", "PERSONALIZATION",
    "OTHER",
]
_GENRE_IDX = {g: i for i, g in enumerate(_GENRE_CATEGORIES)}

# Discrepancy type order (matches discrepancy_labels.py)
DISCREPANCY_TYPES = [
    "CONSISTENT",
    "UNDECLARED_COLLECTION",
    "POLICY_LABEL_MISMATCH",
    "OVER_DISCLOSURE",
]
_DISC_IDX = {d: i for i, d in enumerate(DISCREPANCY_TYPES)}

# ---------------------------------------------------------------------------
# Sentence-transformer singleton
# ---------------------------------------------------------------------------

_embedder_singleton = None


def get_embedder():
    """Lazy-load and cache the all-MiniLM-L6-v2 sentence-transformer."""
    global _embedder_singleton
    if _embedder_singleton is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading sentence-transformer all-MiniLM-L6-v2 ...")
        _embedder_singleton = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder_singleton


def _embed_texts(texts: List[str], embedder) -> np.ndarray:
    """
    Encode a list of strings to [N, 384] float32 array.
    Falls back to zeros for empty inputs.
    """
    if not texts:
        return np.zeros((0, _EMBED_DIM), dtype=np.float32)
    vecs = embedder.encode(texts, show_progress_bar=False, batch_size=64)
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Genre one-hot
# ---------------------------------------------------------------------------

def _genre_onehot(genre_str: str) -> np.ndarray:
    vec = np.zeros(len(_GENRE_CATEGORIES), dtype=np.float32)
    key = str(genre_str).upper().strip() if genre_str else ""
    # Partial match: GAME covers GAME_ACTION, GAME_ARCADE, etc.
    idx = _GENRE_IDX.get(key, None)
    if idx is None:
        for cat in _GENRE_CATEGORIES:
            if key.startswith(cat):
                idx = _GENRE_IDX[cat]
                break
    if idx is None:
        idx = _GENRE_IDX["OTHER"]
    vec[idx] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def nx_to_pyg(
    merged_graph: nx.MultiDiGraph,
    discrepancy_df: Optional[pd.DataFrame] = None,
    embedder=None,
) -> HeteroData:
    """
    Convert a fused M4 NetworkX MultiDiGraph to a PyG HeteroData object.

    Parameters
    ----------
    merged_graph : nx.MultiDiGraph
        Output of fuse_app_graphs().
    discrepancy_df : pd.DataFrame, optional
        Output of compute_discrepancy_labels().  If provided, discrepancy
        class labels are attached as ``data['App'].discrepancy_labels``.
    embedder : SentenceTransformer, optional
        Pre-loaded embedder.  Loaded on first call if None.

    Returns
    -------
    HeteroData
        - data[node_type].x             : node feature tensor
        - data[node_type].node_ids      : list of canonical node IDs
        - data[src, rel, dst].edge_index: edge connectivity [2, E]
        - data['App'].discrepancy_labels: [n_apps, n_dtypes, 4] int64 tensor
    """
    G = merged_graph
    if embedder is None:
        embedder = get_embedder()

    data = HeteroData()

    # ------------------------------------------------------------------
    # Step 1: Partition nodes by type; build local index
    # ------------------------------------------------------------------
    # node_type -> list of (canonical_id, attrs_dict)
    nodes_by_type: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)

    for node_id, attrs in G.nodes(data=True):
        nt = attrs.get("node_type", "Unknown")
        nodes_by_type[nt].append((node_id, attrs))

    # Build canonical_id -> local_index per type (for edge remapping)
    node_to_local: Dict[str, int] = {}  # canonical_id -> local idx in its type
    node_to_type: Dict[str, str] = {}   # canonical_id -> node_type

    for nt, node_list in nodes_by_type.items():
        for local_idx, (nid, _) in enumerate(node_list):
            node_to_local[nid] = local_idx
            node_to_type[nid] = nt

    # ------------------------------------------------------------------
    # Step 2: Build node features per type
    # ------------------------------------------------------------------
    for nt, node_list in nodes_by_type.items():
        node_ids = [nid for nid, _ in node_list]
        attrs_list = [attrs for _, attrs in node_list]

        if nt == "App":
            # One-hot genre encoding
            feats = np.stack([
                _genre_onehot(a.get("genreId", ""))
                for a in attrs_list
            ], axis=0)

        elif nt in ("DataType", "Purpose", "ThirdParty", "PolicySegment"):
            # Sentence-transformer embedding of name/text
            texts = []
            for a in attrs_list:
                text = a.get("name") or a.get("text_snippet") or ""
                # For PolicySegment, use text_snippet if available
                if nt == "PolicySegment":
                    text = a.get("text_snippet") or a.get("name") or ""
                texts.append(str(text))
            feats = _embed_texts(texts, embedder)

        else:
            # Policy, PrivacyLabel, SDK, Endpoint → zero placeholder
            feats = np.zeros((len(node_ids), _EMBED_DIM), dtype=np.float32)

        data[nt].x = torch.tensor(feats, dtype=torch.float32)
        data[nt].node_ids = node_ids

    # ------------------------------------------------------------------
    # Step 3: Build edge_index tensors per edge type triple
    # ------------------------------------------------------------------
    # Group actual edges by (src_type, edge_type, dst_type)
    edge_buckets: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)

    for u, v, edge_attrs in G.edges(data=True):
        et = edge_attrs.get("edge_type", "UNKNOWN")
        src_type = node_to_type.get(u)
        dst_type = node_to_type.get(v)
        if src_type is None or dst_type is None:
            continue
        src_local = node_to_local.get(u)
        dst_local = node_to_local.get(v)
        if src_local is None or dst_local is None:
            continue
        edge_buckets[(src_type, et, dst_type)].append((src_local, dst_local))

    for (src_type, et, dst_type), pairs in edge_buckets.items():
        if not pairs:
            continue
        src_idx = [p[0] for p in pairs]
        dst_idx = [p[1] for p in pairs]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[src_type, et, dst_type].edge_index = edge_index

    # ------------------------------------------------------------------
    # Step 4: Attach discrepancy labels as M5 training targets
    # ------------------------------------------------------------------
    if discrepancy_df is not None and len(discrepancy_df) > 0:
        app_nodes = nodes_by_type.get("App", [])
        dt_nodes = nodes_by_type.get("DataType", [])

        n_apps = len(app_nodes)
        n_dts = len(dt_nodes)
        n_classes = len(DISCREPANCY_TYPES)

        if n_apps > 0 and n_dts > 0:
            label_tensor = torch.zeros(
                (n_apps, n_dts, n_classes), dtype=torch.long
            )

            # Map data_type_node -> local DataType index
            dt_node_to_local = {
                nid: i for i, (nid, _) in enumerate(dt_nodes)
            }
            # Map app_id -> local App index
            app_id_to_local = {
                attrs.get("package_name", nid.replace("App::", "")): i
                for i, (nid, attrs) in enumerate(app_nodes)
            }

            for _, row in discrepancy_df.iterrows():
                app_idx = app_id_to_local.get(row["app_id"], -1)
                dt_node = row.get("data_type_node", "")
                dt_idx = dt_node_to_local.get(dt_node, -1)
                disc_class = _DISC_IDX.get(row.get("discrepancy_type", ""), 0)

                if app_idx >= 0 and dt_idx >= 0:
                    label_tensor[app_idx, dt_idx, disc_class] = 1

            data["App"].discrepancy_labels = label_tensor

    return data
