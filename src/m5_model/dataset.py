"""
dataset.py — M5 dataset loading and splitting utilities.

Loads the 268 HeteroData objects from M4 fusion and attaches discrepancy
labels from the parquet file, then splits by app ID (never splitting a
single app's pairs across train/val/test).

Public API
----------
load_audit_dataset() -> tuple[list[HeteroData], pd.DataFrame]
app_level_split(graphs, ratio, seed) -> tuple[list, list, list]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# Canonical class encoding (matches to_pyg.py DISCREPANCY_TYPES order)
CLASS_NAMES = [
    "CONSISTENT",
    "POLICY_LABEL_MISMATCH",
    "OVER_DISCLOSURE",
    "UNDECLARED_COLLECTION",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# Paths relative to project root
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_GRAPHS_PATH = _PROJ_ROOT / "data" / "processed" / "fused_graphs_full.pt"
_LABELS_PATH = _PROJ_ROOT / "data" / "processed" / "discrepancy_labels_full.parquet"


def _get_app_package(graph: HeteroData) -> str:
    """Return the package name for the single App node in a per-app graph."""
    node_ids = graph["App"].node_ids
    if not node_ids:
        return ""
    return node_ids[0].replace("App::", "")


def load_audit_dataset(
    graphs_path: Path | None = None,
    labels_path: Path | None = None,
) -> Tuple[List[HeteroData], pd.DataFrame]:
    """
    Load fused HeteroData graphs and discrepancy labels, then attach labels
    to each graph.

    Returns
    -------
    graphs : list[HeteroData]
        268 graphs, each with `discrepancy_labels` (int64 tensor of class
        indices, shape [n_pairs]) and `discrepancy_pairs` (int64 tensor of
        shape [n_pairs, 2] giving [app_local_idx, datatype_local_idx]).
    labels_df : pd.DataFrame
        Raw labels parquet for downstream use.
    """
    gp = graphs_path or _GRAPHS_PATH
    lp = labels_path or _LABELS_PATH

    graphs: List[HeteroData] = torch.load(str(gp), weights_only=False)
    labels_df: pd.DataFrame = pd.read_parquet(str(lp))

    # Build lookup: app_id -> list of (data_type_node, class_idx)
    app_to_pairs: dict[str, list] = {}
    for _, row in labels_df.iterrows():
        aid = row["app_id"]
        cls = CLASS_TO_IDX.get(str(row["discrepancy_type"]), 0)
        app_to_pairs.setdefault(aid, []).append((row["data_type_node"], cls))

    for graph in graphs:
        pkg = _get_app_package(graph)
        pairs_for_app = app_to_pairs.get(pkg, [])

        if not pairs_for_app or "DataType" not in graph.node_types:
            # Attach empty tensors so collation still works
            graph.discrepancy_labels = torch.zeros(0, dtype=torch.long)
            graph.discrepancy_pairs = torch.zeros((0, 2), dtype=torch.long)
            continue

        dt_node_ids: list[str] = graph["DataType"].node_ids
        dt_id_to_local = {nid: i for i, nid in enumerate(dt_node_ids)}

        app_local = 0  # each graph has exactly one App node
        pair_indices: list[list[int]] = []
        class_indices: list[int] = []

        for dt_node, cls in pairs_for_app:
            dt_local = dt_id_to_local.get(dt_node, -1)
            if dt_local >= 0:
                pair_indices.append([app_local, dt_local])
                class_indices.append(cls)

        if pair_indices:
            graph.discrepancy_pairs = torch.tensor(pair_indices, dtype=torch.long)
            graph.discrepancy_labels = torch.tensor(class_indices, dtype=torch.long)
        else:
            graph.discrepancy_labels = torch.zeros(0, dtype=torch.long)
            graph.discrepancy_pairs = torch.zeros((0, 2), dtype=torch.long)

    return graphs, labels_df


def app_level_split(
    graphs: List[HeteroData],
    ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[List[HeteroData], List[HeteroData], List[HeteroData]]:
    """
    Split graphs by appId so no app's pairs appear in more than one split.

    Parameters
    ----------
    graphs : list[HeteroData]
    ratio : (train, val, test) — must sum to 1.0
    seed : random seed for reproducibility

    Returns
    -------
    train_graphs, val_graphs, test_graphs
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Only keep graphs that have at least one labeled pair
    labeled = [g for g in graphs if g.discrepancy_labels.numel() > 0]

    # Shuffle app order
    indices = list(range(len(labeled)))
    rng.shuffle(indices)
    shuffled = [labeled[i] for i in indices]

    n = len(shuffled)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return train, val, test
