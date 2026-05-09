"""
baselines_masked.py — Ablation baselines evaluated on MASKED test graphs.

Mirrors baselines.py but all baselines are evaluated on the same masked
splits used by train_masked.py (mask_prob=0.30, seed=42).

This enables apples-to-apples comparison between the full model and baselines
under the non-circular evaluation protocol.

Baseline mapping
----------------
1. tsne_pca_clustering  — Policy text only; structural edges never used.
   Re-evaluated on masked test split for consistency.

2. text_only_logreg     — Policy/label/datatype text embeddings only.
   The features never included structural edges, so masking has no effect on
   features; we re-evaluate on the masked test split labels.

3. policy_only_gnn      — GNN restricted to Policy/PolicySegment/DataType edges.
   MENTIONS edges are NOT masked (by design), so policy-only GNN is unaffected
   by masking — but we re-evaluate on the masked test split for consistency.

4. full_hetero_gnn      — Trained by train_masked.py; metrics loaded from JSON.

Outputs
-------
    reports/baselines_masked/<name>.json
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import HeteroData

from src.m5_model.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    app_level_split,
    load_audit_dataset,
)
from src.m5_model.model import HeteroAuditGNN
from src.m5_model.edge_masking import apply_masking_to_graphs
from src.m5_model.train_masked import (
    _compute_class_weights,
    _evaluate,
    _iter_batches,
)

# Reproducibility
SEED = 42
MASK_PROB = 0.30
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_PROJ_ROOT    = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _PROJ_ROOT / "reports" / "baselines_masked"
_BASELINE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared feature helpers (identical to baselines.py)
# ---------------------------------------------------------------------------

def _get_app_policy_emb(graph: HeteroData) -> np.ndarray:
    if "PolicySegment" in graph.node_types:
        x = graph["PolicySegment"].x.numpy()
        if x.shape[0] > 0:
            return x.mean(axis=0)
    if "Policy" in graph.node_types:
        x = graph["Policy"].x.numpy()
        if x.shape[0] > 0:
            return x.mean(axis=0)
    return np.zeros(384, dtype=np.float32)


def _get_app_label_onehot(graph: HeteroData, n_classes: int = 4) -> np.ndarray:
    if "PrivacyLabel" in graph.node_types:
        x = graph["PrivacyLabel"].x.numpy()
        if x.shape[0] > 0:
            return (x.mean(axis=0)[:n_classes]
                    if x.shape[1] >= n_classes else np.zeros(n_classes))
    return np.zeros(n_classes, dtype=np.float32)


def _get_datatype_emb(graph: HeteroData, dt_local_idx: int) -> np.ndarray:
    if "DataType" in graph.node_types:
        x = graph["DataType"].x.numpy()
        if dt_local_idx < x.shape[0]:
            return x[dt_local_idx]
    return np.zeros(384, dtype=np.float32)


def _extract_pairs(graphs: List[HeteroData]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feature: concat(policy_emb[384], label_emb[4], datatype_emb[384]) = 772-d.
    Note: PrivacyLabel features are node *attribute* embeddings (text/zeros),
    NOT derived from DECLARES_COLLECTS/SHARES edges — so masking those edges
    does not affect these features.
    """
    rows_X, rows_y = [], []
    for g in graphs:
        if g.discrepancy_labels.numel() == 0:
            continue
        policy_emb = _get_app_policy_emb(g)
        label_emb  = _get_app_label_onehot(g)
        pairs  = g.discrepancy_pairs.numpy()
        labels = g.discrepancy_labels.numpy()
        for (app_idx, dt_idx), cls in zip(pairs, labels):
            dt_emb = _get_datatype_emb(g, dt_idx)
            feat   = np.concatenate([policy_emb, label_emb, dt_emb])
            rows_X.append(feat)
            rows_y.append(cls)
    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Baseline 1: tsne_pca_clustering (topic-modeling baseline)
# ---------------------------------------------------------------------------

def run_tsne_pca_clustering(
    train_graphs: List[HeteroData],
    test_graphs: List[HeteroData],
) -> dict:
    """Policy text embeddings only — not affected by edge masking."""
    print("  [Baseline 1] tsne_pca_clustering (masked test split)...")
    t0 = time.time()

    # Gather per-pair (policy_emb, label) for training
    train_embs_labeled, train_labels_flat = [], []
    for g in train_graphs:
        if g.discrepancy_labels.numel() == 0:
            continue
        emb  = _get_app_policy_emb(g)
        lbls = g.discrepancy_labels.numpy()
        for lbl in lbls:
            train_embs_labeled.append(emb)
            train_labels_flat.append(lbl)

    train_embs_labeled = np.array(train_embs_labeled)
    train_labels_flat  = np.array(train_labels_flat)

    pca = PCA(n_components=10, random_state=SEED)
    train_embs_pca = pca.fit_transform(train_embs_labeled)

    km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    km.fit(train_embs_pca)
    train_clusters = km.labels_

    cluster_majority = {}
    for c in range(4):
        mask = train_clusters == c
        if mask.sum() == 0:
            cluster_majority[c] = 0
        else:
            cluster_majority[c] = int(
                np.bincount(train_labels_flat[mask],
                            minlength=len(CLASS_NAMES)).argmax()
            )

    test_preds, test_labels = [], []
    for g in test_graphs:
        if g.discrepancy_labels.numel() == 0:
            continue
        emb       = _get_app_policy_emb(g)
        emb_pca   = pca.transform(emb.reshape(1, -1))
        cluster   = km.predict(emb_pca)[0]
        pred_cls  = cluster_majority[cluster]
        lbls      = g.discrepancy_labels.numpy()
        for lbl in lbls:
            test_preds.append(pred_cls)
            test_labels.append(lbl)

    test_preds  = np.array(test_preds)
    test_labels = np.array(test_labels)

    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    report   = classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    runtime = time.time() - t0
    result  = {
        "model": "tsne_pca_clustering",
        "mask_prob": MASK_PROB,
        "macro_f1": float(macro_f1),
        "per_class_f1": {
            cls: float(report.get(cls, {}).get("f1-score", 0.0))
            for cls in CLASS_NAMES
        },
        "classification_report": report,
        "params": 0,
        "runtime_sec": runtime,
    }
    out_path = _BASELINE_DIR / "tsne_pca_clustering.json"
    with open(str(out_path), "w") as f:
        json.dump(result, f, indent=2)
    print(f"    macro_f1={macro_f1:.4f}, saved to {out_path}")
    return result


# ---------------------------------------------------------------------------
# Baseline 2: text_only_logreg
# ---------------------------------------------------------------------------

def run_text_only_logreg(
    train_graphs: List[HeteroData],
    val_graphs:   List[HeteroData],
    test_graphs:  List[HeteroData],
) -> dict:
    """LogReg on text features — edge masking doesn't affect these features."""
    print("  [Baseline 2] text_only_logreg (masked splits)...")
    t0 = time.time()

    X_train, y_train = _extract_pairs(train_graphs)
    X_val,   y_val   = _extract_pairs(val_graphs)
    X_test,  y_test  = _extract_pairs(test_graphs)

    if X_train.shape[0] == 0:
        print("    No training pairs found, skipping.")
        return {"model": "text_only_logreg", "macro_f1": 0.0, "params": 0,
                "mask_prob": MASK_PROB, "runtime_sec": 0.0}

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=SEED,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(X_train, y_train)
    test_preds = clf.predict(X_test)
    macro_f1   = f1_score(y_test, test_preds, average="macro", zero_division=0)
    report     = classification_report(
        y_test, test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    runtime  = time.time() - t0
    n_params = sum(arr.size for arr in clf.coef_) + clf.intercept_.size

    result = {
        "model": "text_only_logreg",
        "mask_prob": MASK_PROB,
        "macro_f1": float(macro_f1),
        "per_class_f1": {
            cls: float(report.get(cls, {}).get("f1-score", 0.0))
            for cls in CLASS_NAMES
        },
        "classification_report": report,
        "params": int(n_params),
        "runtime_sec": runtime,
    }
    out_path = _BASELINE_DIR / "text_only_logreg.json"
    with open(str(out_path), "w") as f:
        json.dump(result, f, indent=2)
    print(f"    macro_f1={macro_f1:.4f}, saved to {out_path}")
    return result


# ---------------------------------------------------------------------------
# Baseline 3: policy_only_gnn (on masked splits)
# ---------------------------------------------------------------------------

def run_policy_only_gnn(
    train_graphs: List[HeteroData],
    val_graphs:   List[HeteroData],
    test_graphs:  List[HeteroData],
) -> dict:
    """
    GNN restricted to Policy/PolicySegment/DataType/Purpose edges.
    MENTIONS edges are NOT masked, so this baseline is structurally identical
    to its unmasked counterpart — re-evaluated on masked test split for
    apples-to-apples comparison.
    """
    print("  [Baseline 3] policy_only_gnn (masked splits)...")
    t0 = time.time()

    save_path = _PROJ_ROOT / "data" / "processed" / "m5_policy_only_gnn_masked.pt"

    def _filter_policy_only(graphs):
        keep_types = {"Policy", "PolicySegment", "DataType", "Purpose"}
        filtered = []
        for g in graphs:
            if "DataType" not in g.node_types or "PolicySegment" not in g.node_types:
                continue
            fg = HeteroData()
            for nt in g.node_types:
                if nt in keep_types and hasattr(g[nt], "x"):
                    fg[nt].x        = g[nt].x
                    fg[nt].node_ids = g[nt].node_ids
            for src, rel, dst in g.edge_types:
                if src in keep_types and dst in keep_types:
                    fg[src, rel, dst].edge_index = g[src, rel, dst].edge_index
            fg.discrepancy_labels = g.discrepancy_labels
            fg.discrepancy_pairs  = g.discrepancy_pairs
            # Proxy App node from mean PolicySegment
            if "PolicySegment" in fg.node_types and fg["PolicySegment"].x.shape[0] > 0:
                mean_emb     = fg["PolicySegment"].x.mean(dim=0, keepdim=True)
                fg["App"].x  = mean_emb
                fg["App"].node_ids = g["App"].node_ids
            else:
                fg["App"].x  = torch.zeros(1, 384)
                fg["App"].node_ids = g["App"].node_ids
            filtered.append(fg)
        return filtered

    train_f = _filter_policy_only(train_graphs)
    val_f   = _filter_policy_only(val_graphs)
    test_f  = _filter_policy_only(test_graphs)

    if not train_f:
        print("    No valid policy-only graphs, skipping.")
        return {"model": "policy_only_gnn", "macro_f1": 0.0, "params": 0,
                "mask_prob": MASK_PROB, "runtime_sec": 0.0}

    model        = HeteroAuditGNN(hidden_dim=128, dropout=0.2, policy_only=True)
    class_weights = _compute_class_weights(train_f).to("cpu")
    criterion    = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1      = -1.0
    best_state       = None
    patience_counter = 0
    patience         = 8

    for epoch in range(1, 51):
        model.train()
        for batch in _iter_batches(train_f, 8, shuffle=True):
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, requires_grad=True)
            n = 0
            for data in batch:
                if data.discrepancy_labels.numel() == 0:
                    continue
                logits = model(data)
                if logits.numel() == 0:
                    continue
                loss       = criterion(logits, data.discrepancy_labels)
                batch_loss = batch_loss + loss
                n         += data.discrepancy_labels.shape[0]
            if n > 0:
                (batch_loss / n).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        val_f1, _, _ = _evaluate(model, val_f)
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            best_state       = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), str(save_path))
    test_f1, test_preds, test_labels = _evaluate(model, test_f)

    report = {}
    if len(test_preds) > 0:
        report = classification_report(
            test_labels, test_preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

    runtime  = time.time() - t0
    from torch.nn.parameter import UninitializedParameter
    n_params = sum(p.numel() for p in model.parameters()
                   if not isinstance(p, UninitializedParameter))

    result = {
        "model": "policy_only_gnn",
        "mask_prob": MASK_PROB,
        "macro_f1": float(test_f1),
        "per_class_f1": {
            cls: float(report.get(cls, {}).get("f1-score", 0.0))
            for cls in CLASS_NAMES
        },
        "classification_report": report,
        "params": n_params,
        "runtime_sec": runtime,
    }
    out_path = _BASELINE_DIR / "policy_only_gnn.json"
    with open(str(out_path), "w") as f:
        json.dump(result, f, indent=2)
    print(f"    macro_f1={test_f1:.4f}, saved to {out_path}")
    return result


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    train_graphs: List[HeteroData],
    val_graphs:   List[HeteroData],
    test_graphs:  List[HeteroData],
) -> dict:
    """Run all three baselines on masked splits and return results dict."""
    print()
    print("=" * 60)
    print("Running ablation baselines (masked graphs)")
    print("=" * 60)

    results = {}
    results["tsne_pca_clustering"] = run_tsne_pca_clustering(
        train_graphs, test_graphs
    )
    results["text_only_logreg"] = run_text_only_logreg(
        train_graphs, val_graphs, test_graphs
    )
    results["policy_only_gnn"] = run_policy_only_gnn(
        train_graphs, val_graphs, test_graphs
    )
    return results


if __name__ == "__main__":
    graphs, _ = load_audit_dataset()
    train, val, test = app_level_split(graphs, seed=SEED)

    # Apply masking before running baselines
    train_m, _ = apply_masking_to_graphs(train, mask_prob=MASK_PROB, seed=SEED)
    val_m,   _ = apply_masking_to_graphs(val,   mask_prob=MASK_PROB, seed=SEED)
    test_m,  _ = apply_masking_to_graphs(test,  mask_prob=MASK_PROB, seed=SEED)

    run_all_baselines(train_m, val_m, test_m)
