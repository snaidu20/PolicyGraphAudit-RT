"""
inference.py — M6 inference helper for PolicyGraphAudit-RT.

Public API
----------
score_app(appId: str) -> dict
    Loads fused_graphs_full.pt and m5_best_model_masked.pt, runs the
    full_hetero_gnn on the named app's graph, and returns a structured
    result dict with predictions, risk score and evidence fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_GRAPHS_PATH = _ROOT / "data" / "processed" / "fused_graphs_full.pt"
_MODEL_PATH  = _ROOT / "data" / "processed" / "m5_best_model_masked.pt"
_LABELS_PATH = _ROOT / "data" / "processed" / "discrepancy_labels_full.parquet"
_META_PATH   = _ROOT / "data" / "raw" / "play_data_safety" / "sample_5000.json"

# Risk weight per discrepancy class
_RISK_WEIGHTS = {
    "UNDECLARED_COLLECTION":  1.0,
    "POLICY_LABEL_MISMATCH":  0.7,
    "OVER_DISCLOSURE":        0.3,
    "CONSISTENT":             0.0,
}

CLASS_NAMES = [
    "CONSISTENT",
    "POLICY_LABEL_MISMATCH",
    "OVER_DISCLOSURE",
    "UNDECLARED_COLLECTION",
]


# ---------------------------------------------------------------------------
# Lazy-cached loaders
# ---------------------------------------------------------------------------
_GRAPHS_CACHE: list | None = None
_MODEL_CACHE: Any | None = None
_LABELS_CACHE: pd.DataFrame | None = None
_META_CACHE: dict[str, dict] | None = None


def _load_graphs() -> list:
    global _GRAPHS_CACHE
    if _GRAPHS_CACHE is None:
        _GRAPHS_CACHE = torch.load(str(_GRAPHS_PATH), weights_only=False)
    return _GRAPHS_CACHE


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import sys
        sys.path.insert(0, str(_ROOT / "src"))
        from m5_model.model import HeteroAuditGNN
        model = HeteroAuditGNN(hidden_dim=128, dropout=0.2, num_classes=4)
        state = torch.load(str(_MODEL_PATH), weights_only=False, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _MODEL_CACHE = model
    return _MODEL_CACHE


def _load_labels() -> pd.DataFrame:
    global _LABELS_CACHE
    if _LABELS_CACHE is None:
        _LABELS_CACHE = pd.read_parquet(str(_LABELS_PATH))
    return _LABELS_CACHE


def _load_metadata() -> dict[str, dict]:
    """Build appId -> metadata dict from play data safety JSONL."""
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    meta: dict[str, dict] = {}
    with open(_META_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = row.get("appId", "")
            if aid and aid not in meta:
                meta[aid] = {
                    "title":            row.get("appTitle", aid),
                    "developer":        row.get("developer", "Unknown"),
                    "genreId":          row.get("genreId", "Unknown"),
                    "installs_bucket":  row.get("installs", "Unknown"),
                    "privacyPolicy_url": row.get("privacyPolicy", ""),
                }
    _META_CACHE = meta
    return meta


# ---------------------------------------------------------------------------
# Core: score_app
# ---------------------------------------------------------------------------

def score_app(appId: str) -> dict:
    """
    Run inference for a single app and return a structured audit result.

    Parameters
    ----------
    appId : str
        Android package name (e.g. 'com.example.app').

    Returns
    -------
    dict with keys:
        appId, app_metadata, predictions, risk_score, summary_counts
    """
    graphs = _load_graphs()
    model  = _load_model()
    labels = _load_labels()
    meta   = _load_metadata()

    # Find the graph for this appId
    graph = None
    for g in graphs:
        if g.app_id == appId:
            graph = g
            break
    if graph is None:
        raise ValueError(f"appId '{appId}' not found in fused_graphs_full.pt")

    # Attach discrepancy_pairs from labels parquet (same logic as dataset.py)
    app_labels = labels[labels["app_id"] == appId]

    if app_labels.empty or "DataType" not in graph.node_types:
        # No labeled pairs: return empty predictions
        return _empty_result(appId, meta)

    dt_node_ids: list[str] = graph["DataType"].node_ids
    dt_id_to_local = {nid: i for i, nid in enumerate(dt_node_ids)}

    pair_indices: list[list[int]] = []
    row_data: list[dict] = []
    for _, row in app_labels.iterrows():
        dt_node = row["data_type_node"]
        dt_local = dt_id_to_local.get(dt_node, -1)
        if dt_local < 0:
            continue
        pair_indices.append([0, dt_local])
        row_data.append({
            "data_type":       row["data_type"],
            "data_type_node":  dt_node,
            "has_label_collects": bool(row.get("label_collects", False)),
            "has_label_shares":   bool(row.get("label_shares", False)),
            "has_policy_mentions": bool(row.get("policy_mentions", False)),
            "has_sdk_collects":   bool(row.get("runtime_implies", False)),
        })

    if not pair_indices:
        return _empty_result(appId, meta)

    pairs_tensor = torch.tensor(pair_indices, dtype=torch.long)

    # Run model (full graph — no masking for inference)
    with torch.no_grad():
        logits = model(graph, pairs=pairs_tensor)  # [n_pairs, 4]
        probs  = F.softmax(logits, dim=-1)         # [n_pairs, 4]
        pred_classes = probs.argmax(dim=-1)        # [n_pairs]

    # Build SDK name lookup from graph node_ids
    sdk_names: list[str] = []
    if "SDK" in graph.node_types:
        for nid in graph["SDK"].node_ids:
            # node_id format: SDK::<app_id>::<sdk_name>
            parts = nid.split("::")
            sdk_names.append(parts[-1] if len(parts) >= 2 else nid)

    # Assemble predictions
    predictions: list[dict] = []
    summary_counts = {c: 0 for c in CLASS_NAMES}
    total_weight = 0.0

    for i, rd in enumerate(row_data):
        cls_idx  = int(pred_classes[i].item())
        cls_name = CLASS_NAMES[cls_idx]
        conf     = float(probs[i, cls_idx].item())

        summary_counts[cls_name] += 1
        total_weight += _RISK_WEIGHTS[cls_name]

        predictions.append({
            "data_type":       rd["data_type"],
            "predicted_class": cls_name,
            "confidence":      conf,
            "evidence": {
                "has_label_collects":  rd["has_label_collects"],
                "has_label_shares":    rd["has_label_shares"],
                "has_policy_mentions": rd["has_policy_mentions"],
                "has_sdk_collects":    rd["has_sdk_collects"],
                "sdks_involved":       sdk_names,
            },
        })

    # Risk score: weighted average
    n = len(predictions)
    risk_score = round(total_weight / n, 4) if n > 0 else 0.0

    app_meta = meta.get(appId, {
        "title":            appId,
        "developer":        "Unknown",
        "genreId":          "Unknown",
        "installs_bucket":  "Unknown",
        "privacyPolicy_url": "",
    })

    return {
        "appId":        appId,
        "app_metadata": app_meta,
        "predictions":  predictions,
        "risk_score":   risk_score,
        "summary_counts": summary_counts,
    }


def _empty_result(appId: str, meta: dict) -> dict:
    app_meta = meta.get(appId, {
        "title":            appId,
        "developer":        "Unknown",
        "genreId":          "Unknown",
        "installs_bucket":  "Unknown",
        "privacyPolicy_url": "",
    })
    return {
        "appId":        appId,
        "app_metadata": app_meta,
        "predictions":  [],
        "risk_score":   0.0,
        "summary_counts": {c: 0 for c in CLASS_NAMES},
    }


def list_all_app_ids() -> list[str]:
    """Return all appIds available in the fused graphs."""
    return [g.app_id for g in _load_graphs()]


def score_all_apps() -> list[dict]:
    """Run score_app on every graph that has labeled pairs. Returns list of results."""
    results = []
    for app_id in list_all_app_ids():
        try:
            r = score_app(app_id)
            if r["predictions"]:
                results.append(r)
        except Exception:
            pass
    return results
