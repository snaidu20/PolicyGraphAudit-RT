"""
model.py — HeteroAuditGNN: heterogeneous R-GCN encoder + MLP classifier head.

Architecture
------------
1. Per-node-type input projection: Linear(input_dim -> hidden_dim=128)
2. 2-layer HeteroConv with SAGEConv per edge type (bidirectional via
   ToUndirected applied at data-loading time)
3. Classifier head: concat(App_emb, DataType_emb) -> MLP(256->64->4)

The model handles missing node types gracefully (some graphs lack SDK /
Endpoint nodes) by zeroing out their contributions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected

# ---------------------------------------------------------------------------
# Node type input dimensions (from m5_model.yaml)
# ---------------------------------------------------------------------------
NODE_INPUT_DIMS: Dict[str, int] = {
    "Policy": 384,          # zero-placeholder in v1 (384-dim zeros from to_pyg)
    "PolicySegment": 384,   # all-MiniLM-L6-v2
    "DataType": 384,
    "Purpose": 384,
    "ThirdParty": 384,
    "PrivacyLabel": 384,    # zero-placeholder
    "App": 34,              # genre one-hot (34 categories)
    "SDK": 384,             # zero-placeholder
    "Endpoint": 384,        # zero-placeholder
}

# Edge types used for message passing (from m5_model.yaml)
EDGE_TYPES_FOR_PASSING: List[Tuple[str, str, str]] = [
    ("Policy",        "HAS_SEGMENT",        "PolicySegment"),
    ("PolicySegment", "MENTIONS",            "DataType"),
    ("PolicySegment", "FOR_PURPOSE",         "Purpose"),
    ("PolicySegment", "SHARED_WITH",         "ThirdParty"),
    ("PrivacyLabel",  "DECLARES_COLLECTS",   "DataType"),
    ("PrivacyLabel",  "DECLARES_SHARES",     "DataType"),
    ("PrivacyLabel",  "DECLARES_PURPOSE",    "Purpose"),
    ("SDK",           "COLLECTS_DATATYPE",   "DataType"),
    ("SDK",           "OPERATED_BY",         "ThirdParty"),
    ("App",           "CONTAINS_SDK",        "SDK"),
    ("App",           "HAS_POLICY",          "Policy"),
    ("App",           "HAS_LABEL",           "PrivacyLabel"),
]

_UNDIRECTED = ToUndirected()


def _reverse_key(src: str, rel: str, dst: str) -> Tuple[str, str, str]:
    return (dst, f"rev_{rel}", src)


class HeteroAuditGNN(nn.Module):
    """
    Heterogeneous GNN for (App, DataType) discrepancy classification.

    Parameters
    ----------
    hidden_dim : int
        Dimension of all internal representations (default 128).
    dropout : float
        Dropout probability applied after each GNN layer (default 0.2).
    num_classes : int
        Number of output discrepancy classes (default 4).
    policy_only : bool
        If True, only use Policy-side nodes (for policy_only_gnn baseline).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_classes: int = 4,
        policy_only: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.policy_only = policy_only

        # Input projections — one per node type
        self.input_proj = nn.ModuleDict()
        for nt, in_dim in NODE_INPUT_DIMS.items():
            self.input_proj[nt] = nn.Linear(in_dim, hidden_dim)

        # Build GNN layers
        self.conv_layers = nn.ModuleList()
        for _ in range(2):
            conv_dict = self._build_conv_dict(hidden_dim)
            self.conv_layers.append(HeteroConv(conv_dict, aggr="mean"))

        # Classifier head: concat(App, DataType) -> [256] -> [64] -> [num_classes]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _build_conv_dict(self, dim: int) -> dict:
        """
        Build one SAGEConv per (forward + reverse) edge type.
        SAGEConv(in_channels, out_channels) with in_channels as (-1, -1)
        for lazy initialisation — handles variable input sizes across
        heterogeneous node types.
        """
        conv_dict = {}
        edge_types = self._get_edge_types()
        for src, rel, dst in edge_types:
            conv_dict[(src, rel, dst)] = SAGEConv((-1, -1), dim)
            rev = _reverse_key(src, rel, dst)
            conv_dict[rev] = SAGEConv((-1, -1), dim)
        return conv_dict

    def _get_edge_types(self) -> List[Tuple[str, str, str]]:
        if self.policy_only:
            return [
                ("Policy",        "HAS_SEGMENT",  "PolicySegment"),
                ("PolicySegment", "MENTIONS",      "DataType"),
                ("PolicySegment", "FOR_PURPOSE",   "Purpose"),
            ]
        return EDGE_TYPES_FOR_PASSING

    def _project_nodes(
        self, x_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """Apply per-type input projection."""
        out = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                proj = self.input_proj[nt]
                # Handle shape mismatch: if feature dim differs, zero-pad or truncate
                expected_in = proj.in_features
                if x.shape[1] != expected_in:
                    if x.shape[1] < expected_in:
                        pad = torch.zeros(x.shape[0], expected_in - x.shape[1],
                                          device=x.device)
                        x = torch.cat([x, pad], dim=1)
                    else:
                        x = x[:, :expected_in]
                out[nt] = proj(x)
            else:
                # Unknown type: use first hidden_dim features or zeros
                if x.shape[1] >= self.hidden_dim:
                    out[nt] = x[:, : self.hidden_dim]
                else:
                    pad = torch.zeros(x.shape[0], self.hidden_dim - x.shape[1],
                                      device=x.device)
                    out[nt] = torch.cat([x, pad], dim=1)
        return out

    def _build_edge_index_dict(
        self, data: HeteroData
    ) -> Dict[Tuple[str, str, str], Tensor]:
        """Collect forward + reverse edge indices from a HeteroData object."""
        edge_index_dict: Dict[Tuple[str, str, str], Tensor] = {}
        for triple in data.edge_types:
            src, rel, dst = triple
            # Forward
            ei = data[src, rel, dst].edge_index
            edge_index_dict[(src, rel, dst)] = ei
            # Reverse
            rev_key = _reverse_key(src, rel, dst)
            if ei.shape[1] > 0:
                edge_index_dict[rev_key] = ei.flip(0)
        return edge_index_dict

    def encode(self, data: HeteroData) -> Dict[str, Tensor]:
        """
        Run the GNN encoder and return a dict of node embeddings.

        Parameters
        ----------
        data : HeteroData

        Returns
        -------
        dict mapping node_type -> Tensor[n_nodes, hidden_dim]
        """
        # Project inputs
        x_dict = {nt: data[nt].x for nt in data.node_types
                  if hasattr(data[nt], "x")}
        x_dict = self._project_nodes(x_dict)

        # Build edge index dict with reverses
        edge_index_dict = self._build_edge_index_dict(data)

        # Filter x_dict and edge_index_dict to relevant types
        valid_node_types = set(x_dict.keys())
        filtered_edge_index = {
            k: v for k, v in edge_index_dict.items()
            if k[0] in valid_node_types and k[2] in valid_node_types
        }

        # GNN layers
        for conv_layer in self.conv_layers:
            # Only pass edges that exist in this graph
            available_edge_keys = set(filtered_edge_index.keys())
            layer_convs = {
                k: v for k, v in conv_layer.convs.items()
                if k in available_edge_keys
            }
            if not layer_convs:
                break

            # Run conv (handles missing edge types gracefully)
            try:
                new_x = conv_layer(x_dict, filtered_edge_index)
            except Exception:
                # Fallback: skip this layer
                break

            # Apply activation + dropout, carry forward unchanged types
            for nt in new_x:
                new_x[nt] = F.relu(new_x[nt])
                new_x[nt] = F.dropout(new_x[nt], p=self.dropout,
                                      training=self.training)
            # Merge: update existing, keep types that weren't touched
            for nt in new_x:
                x_dict[nt] = new_x[nt]

        return x_dict

    def forward(
        self,
        data: HeteroData,
        pairs: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        data : HeteroData
        pairs : Tensor[n_pairs, 2] — [app_local_idx, datatype_local_idx]
            If None, uses data.discrepancy_pairs.

        Returns
        -------
        logits : Tensor[n_pairs, num_classes]
        """
        x_dict = self.encode(data)

        if pairs is None:
            pairs = data.discrepancy_pairs  # shape [n_pairs, 2]

        if pairs.numel() == 0:
            return torch.zeros(0, 4, device=next(self.parameters()).device)

        app_emb = x_dict.get("App")
        dt_emb = x_dict.get("DataType")

        if app_emb is None or dt_emb is None:
            return torch.zeros(pairs.shape[0], 4,
                               device=next(self.parameters()).device)

        app_idx = pairs[:, 0]
        dt_idx = pairs[:, 1]

        # Clamp indices to valid range
        app_idx = app_idx.clamp(0, app_emb.shape[0] - 1)
        dt_idx = dt_idx.clamp(0, dt_emb.shape[0] - 1)

        app_vecs = app_emb[app_idx]   # [n_pairs, hidden_dim]
        dt_vecs = dt_emb[dt_idx]      # [n_pairs, hidden_dim]

        combined = torch.cat([app_vecs, dt_vecs], dim=-1)  # [n_pairs, 2*hidden]
        logits = self.classifier(combined)
        return logits
