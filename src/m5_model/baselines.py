"""
baselines.py — Ablation baseline models for M5.

Baseline 1: tsne_pca_clustering (topic-modeling baseline)
    PCA(10) + KMeans(4) on policy text embeddings → majority-class prediction.

Baseline 2: text_only_logreg
    LogisticRegression on concat(policy_emb, label_one_hot, datatype_emb)
    per (App, DataType) pair. No graph structure.

Baseline 3: policy_only_gnn
    Same HeteroAuditGNN architecture but restricted to Policy-side nodes only.
    Predict from DataType embedding after message passing.

All baselines use the same train/val/test split as the full model.
Results saved to reports/baselines/<name>.json.
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
from src.m5_model.train import (
    _compute_class_weights,
    _evaluate,
    _iter_batches,
    train_model,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

_PROJ_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _PROJ_ROOT / "reports" / "baselines"
_BASELINE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_app_policy_emb(graph: HeteroData) -> np.ndarray:
    """
    Return a single 384-d policy embedding for the app in this graph.
    Mean-pool PolicySegment embeddings if available, else zeros.
    """
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
    """One-hot encode the PrivacyLabel node if present (placeholder)."""
    if "PrivacyLabel" in graph.node_types:
        x = graph["PrivacyLabel"].x.numpy()
        if x.shape[0] > 0:
            return x.mean(axis=0)[:n_classes] if x.shape[1] >= n_classes else np.zeros(n_classes)
    return np.zeros(n_classes, dtype=np.float32)


def _get_datatype_emb(graph: HeteroData, dt_local_idx: int) -> np.ndarray:
    """Return the 384-d embedding of a specific DataType node."""
    if "DataType" in graph.node_types:
        x = graph["DataType"].x.numpy()
        if dt_local_idx < x.shape[0]:
            return x[dt_local_idx]
    return np.zeros(384, dtype=np.float32)


def _extract_pairs(
    graphs: List[HeteroData],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels for text-based baselines.
    Returns (X, y) where each row is one (App, DataType) pair.
    Feature: concat(policy_emb[384], label_emb[4], datatype_emb[384]) = 772-d
    """
    rows_X, rows_y = [], []
    for g in graphs:
        if g.discrepancy_labels.numel() == 0:
            continue
        policy_emb = _get_app_policy_emb(g)
        label_emb = _get_app_label_onehot(g)
        pairs = g.discrepancy_pairs.numpy()
        labels = g.discrepancy_labels.numpy()
        for (app_idx, dt_idx), cls in zip(pairs, labels):
            dt_emb = _get_datatype_emb(g, dt_idx)
            feat = np.concatenate([policy_emb, label_emb, dt_emb])
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
    """
    Topic-modeling baseline: PCA(10) + KMeans(4) on policy text embeddings.
    For each (App, DataType) test pair, predict the majority discrepancy class
    within the cluster of the app's policy.
    """
    print("  [Baseline 1] tsne_pca_clustering (topic-modeling baseline)...")
    t0 = time.time()

    # Build policy embeddings for all train apps
    train_embs = np.array([_get_app_policy_emb(g) for g in train_graphs])
    train_labels_per_app = [
        g.discrepancy_labels.numpy() for g in train_graphs
        if g.discrepancy_labels.numel() > 0
    ]
    # Flatten labels per app for cluster majority voting
    train_embs_labeled = []
    train_labels_flat = []
    for g in train_graphs:
        if g.discrepancy_labels.numel() == 0:
            continue
        emb = _get_app_policy_emb(g)
        lbls = g.discrepancy_labels.numpy()
        for lbl in lbls:
            train_embs_labeled.append(emb)
            train_labels_flat.append(lbl)

    train_embs_labeled = np.array(train_embs_labeled)
    train_labels_flat = np.array(train_labels_flat)

    # PCA
    pca = PCA(n_components=10, random_state=SEED)
    train_embs_pca = pca.fit_transform(train_embs_labeled)

    # KMeans
    km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    km.fit(train_embs_pca)
    train_clusters = km.labels_

    # Majority class per cluster
    cluster_majority = {}
    for c in range(4):
        mask = train_clusters == c
        if mask.sum() == 0:
            cluster_majority[c] = 0
        else:
            cluster_majority[c] = int(
                np.bincount(train_labels_flat[mask], minlength=len(CLASS_NAMES)).argmax()
            )

    # Predict on test set
    test_preds, test_labels = [], []
    for g in test_graphs:
        if g.discrepancy_labels.numel() == 0:
            continue
        emb = _get_app_policy_emb(g)
        emb_pca = pca.transform(emb.reshape(1, -1))
        cluster = km.predict(emb_pca)[0]
        pred_class = cluster_majority[cluster]
        lbls = g.discrepancy_labels.numpy()
        for lbl in lbls:
            test_preds.append(pred_class)
            test_labels.append(lbl)

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    report = classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    runtime = time.time() - t0
    result = {
        "model": "tsne_pca_clustering",
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
    val_graphs: List[HeteroData],
    test_graphs: List[HeteroData],
) -> dict:
    """
    Logistic regression on concat(policy_emb, label_emb, datatype_emb).
    No graph structure used.
    """
    print("  [Baseline 2] text_only_logreg...")
    t0 = time.time()

    X_train, y_train = _extract_pairs(train_graphs)
    X_val, y_val = _extract_pairs(val_graphs)
    X_test, y_test = _extract_pairs(test_graphs)

    if X_train.shape[0] == 0:
        print("    No training pairs found, skipping.")
        return {"model": "text_only_logreg", "macro_f1": 0.0, "params": 0,
                "runtime_sec": 0.0}

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=SEED,
        solver="lbfgs",
        C=1.0,
    )
    clf.fit(X_train, y_train)

    test_preds = clf.predict(X_test)
    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    report = classification_report(
        y_test, test_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    runtime = time.time() - t0
    n_params = sum(arr.size for arr in clf.coef_) + clf.intercept_.size

    result = {
        "model": "text_only_logreg",
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
# Baseline 3: policy_only_gnn
# ---------------------------------------------------------------------------

def run_policy_only_gnn(
    train_graphs: List[HeteroData],
    val_graphs: List[HeteroData],
    test_graphs: List[HeteroData],
) -> dict:
    """
    Same HeteroAuditGNN but restricted to Policy/PolicySegment/DataType/Purpose
    node types. No App, SDK, PrivacyLabel, etc.
    """
    print("  [Baseline 3] policy_only_gnn...")
    t0 = time.time()

    save_path = _PROJ_ROOT / "data" / "processed" / "m5_policy_only_gnn.pt"

    def _filter_policy_only(graphs):
        """Keep only policy-side node types and edges."""
        keep_types = {"Policy", "PolicySegment", "DataType", "Purpose"}
        filtered = []
        for g in graphs:
            if "DataType" not in g.node_types or "PolicySegment" not in g.node_types:
                continue
            from torch_geometric.data import HeteroData
            fg = HeteroData()
            for nt in g.node_types:
                if nt in keep_types and hasattr(g[nt], "x"):
                    fg[nt].x = g[nt].x
                    fg[nt].node_ids = g[nt].node_ids
            for src, rel, dst in g.edge_types:
                if src in keep_types and dst in keep_types:
                    fg[src, rel, dst].edge_index = g[src, rel, dst].edge_index
            # Attach labels — predict DataType from policy side
            fg.discrepancy_labels = g.discrepancy_labels
            fg.discrepancy_pairs = g.discrepancy_pairs
            # For policy-only, the "App" embedding must come from somewhere:
            # We use a mean-pooled PolicySegment embedding as the App proxy.
            # We attach it as an App node with the mean embedding.
            if "PolicySegment" in fg.node_types and fg["PolicySegment"].x.shape[0] > 0:
                mean_emb = fg["PolicySegment"].x.mean(dim=0, keepdim=True)
                fg["App"].x = mean_emb
                fg["App"].node_ids = g["App"].node_ids
            else:
                fg["App"].x = torch.zeros(1, 384)
                fg["App"].node_ids = g["App"].node_ids
            filtered.append(fg)
        return filtered

    train_f = _filter_policy_only(train_graphs)
    val_f = _filter_policy_only(val_graphs)
    test_f = _filter_policy_only(test_graphs)

    if not train_f:
        print("    No valid policy-only graphs, skipping.")
        return {"model": "policy_only_gnn", "macro_f1": 0.0, "params": 0,
                "runtime_sec": 0.0}

    # Re-use train_model but with policy_only=True and these filtered graphs
    # We manually train here for full control
    model = HeteroAuditGNN(hidden_dim=128, dropout=0.2, policy_only=True)
    class_weights = _compute_class_weights(train_f).to("cpu")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0
    patience = 8

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
                loss = criterion(logits, data.discrepancy_labels)
                batch_loss = batch_loss + loss
                n += data.discrepancy_labels.shape[0]
            if n > 0:
                (batch_loss / n).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        val_f1, _, _ = _evaluate(model, val_f)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
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

    runtime = time.time() - t0
    from torch.nn.parameter import UninitializedParameter
    n_params = sum(p.numel() for p in model.parameters()
                   if not isinstance(p, UninitializedParameter))

    result = {
        "model": "policy_only_gnn",
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
    val_graphs: List[HeteroData],
    test_graphs: List[HeteroData],
) -> dict:
    """Run all three baselines and return results dict."""
    print()
    print("=" * 60)
    print("Running ablation baselines")
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
    run_all_baselines(train, val, test)
