"""
train.py — Training loop for HeteroAuditGNN.

Usage
-----
    python -m src.m5_model.train

Outputs
-------
- data/processed/m5_best_model.pt
- reports/m5_training_curves.png
- reports/m5_test_metrics.json
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import HeteroData

from src.m5_model.dataset import CLASS_NAMES, app_level_split, load_audit_dataset
from src.m5_model.model import HeteroAuditGNN

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parents[2]
_MODEL_OUT = _PROJ_ROOT / "data" / "processed" / "m5_best_model.pt"
_CURVES_OUT = _PROJ_ROOT / "reports" / "m5_training_curves.png"
_METRICS_OUT = _PROJ_ROOT / "reports" / "m5_test_metrics.json"
_PROJ_ROOT / "reports" / "baselines"


def _iter_batches(graphs: List[HeteroData], batch_size: int, shuffle: bool = False):
    """Yield mini-batches (lists of HeteroData) without PyG collation."""
    indices = list(range(len(graphs)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [graphs[i] for i in indices[start:start + batch_size]]


def _evaluate(model: HeteroAuditGNN, graphs: List[HeteroData], device: str = "cpu"):
    """Run evaluation, return (macro_f1, all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            if data.discrepancy_labels.numel() == 0:
                continue
            logits = model(data)
            if logits.numel() == 0:
                continue
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(data.discrepancy_labels.cpu())
    if not all_preds:
        return 0.0, [], []
    preds_cat = torch.cat(all_preds).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    macro_f1 = f1_score(labels_cat, preds_cat, average="macro", zero_division=0)
    return macro_f1, preds_cat, labels_cat


def _compute_class_weights(train_graphs) -> torch.Tensor:
    """Compute balanced class weights from training set labels."""
    all_labels = []
    for g in train_graphs:
        if g.discrepancy_labels.numel() > 0:
            all_labels.append(g.discrepancy_labels.numpy())
    if not all_labels:
        return torch.ones(len(CLASS_NAMES))
    y = np.concatenate(all_labels)
    classes = np.arange(len(CLASS_NAMES))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    hidden_dim: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    epochs: int = 50,
    patience: int = 8,
    device: str = "cpu",
    policy_only: bool = False,
    save_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train HeteroAuditGNN, return dict with metrics and history.

    Parameters
    ----------
    policy_only : bool
        If True, trains the policy_only_gnn ablation baseline.
    save_path : Path | None
        Where to save best model weights. If None, uses default path.
    """
    t0 = time.time()

    # Load data
    if verbose:
        print("Loading dataset...")
    graphs, _ = load_audit_dataset()
    train_graphs, val_graphs, test_graphs = app_level_split(graphs, seed=SEED)

    if verbose:
        print(f"  Train: {len(train_graphs)} apps, "
              f"Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # We iterate graphs individually (no PyG batch collation — pairs have variable sizes)

    # Class weights
    class_weights = _compute_class_weights(train_graphs).to(device)
    if verbose:
        print(f"  Class weights: {class_weights.tolist()}")

    # Model
    model = HeteroAuditGNN(
        hidden_dim=hidden_dim,
        dropout=0.2,
        num_classes=len(CLASS_NAMES),
        policy_only=policy_only,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training history
    history = {
        "train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []
    }
    best_val_f1 = -1.0
    patience_counter = 0
    best_state = None

    if verbose:
        print("Training...")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss, epoch_preds, epoch_labels = 0.0, [], []
        n_batches = 0

        for batch in _iter_batches(train_graphs, batch_size, shuffle=True):
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, requires_grad=True)
            n_pairs_total = 0

            for data in batch:
                data = data.to(device)
                if data.discrepancy_labels.numel() == 0:
                    continue
                logits = model(data)
                if logits.numel() == 0:
                    continue
                labels = data.discrepancy_labels
                loss = criterion(logits, labels)
                batch_loss = batch_loss + loss
                n_pairs_total += labels.shape[0]

                with torch.no_grad():
                    epoch_preds.append(logits.argmax(dim=-1).cpu())
                    epoch_labels.append(labels.cpu())

            if n_pairs_total > 0:
                batch_loss = batch_loss / n_pairs_total
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Train F1
        if epoch_preds:
            train_preds = torch.cat(epoch_preds).numpy()
            train_labels_cat = torch.cat(epoch_labels).numpy()
            train_f1 = f1_score(train_labels_cat, train_preds,
                                average="macro", zero_division=0)
        else:
            train_f1 = 0.0

        # --- Validate ---
        val_f1, val_preds, val_labels_cat = _evaluate(model, val_graphs, device)

        # Val loss
        model.eval()
        val_loss_total, val_n = 0.0, 0
        with torch.no_grad():
            for data in val_graphs:
                data = data.to(device)
                if data.discrepancy_labels.numel() == 0:
                    continue
                logits = model(data)
                if logits.numel() == 0:
                    continue
                loss = criterion(logits, data.discrepancy_labels)
                val_loss_total += loss.item()
                val_n += 1
        val_loss = val_loss_total / max(val_n, 1)

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={avg_loss:.4f} train_f1={train_f1:.4f} | "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (patience={patience})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model
    out_path = save_path or _MODEL_OUT
    torch.save(model.state_dict(), str(out_path))
    if verbose:
        print(f"  Best model saved to {out_path}")

    # --- Test evaluation ---
    test_f1, test_preds, test_labels_np = _evaluate(model, test_graphs, device)

    if len(test_preds) > 0:
        report = classification_report(
            test_labels_np, test_preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(test_labels_np, test_preds,
                              labels=list(range(len(CLASS_NAMES)))).tolist()
    else:
        report = {}
        cm = []

    runtime = time.time() - t0
    from torch.nn.parameter import UninitializedParameter
    n_params = sum(p.numel() for p in model.parameters()
                   if not isinstance(p, UninitializedParameter))

    metrics = {
        "model": "full_hetero_gnn" if not policy_only else "policy_only_gnn",
        "macro_f1": float(test_f1),
        "per_class_f1": {
            cls: float(report.get(cls, {}).get("f1-score", 0.0))
            for cls in CLASS_NAMES
        },
        "classification_report": report,
        "confusion_matrix": cm,
        "best_val_f1": float(best_val_f1),
        "epochs_trained": len(history["train_loss"]),
        "params": n_params,
        "runtime_sec": runtime,
        "history": history,
    }

    return metrics, model, (train_graphs, val_graphs, test_graphs)


def save_training_curves(history: dict, out_path: Path = _CURVES_OUT):
    """Save loss + macro F1 training curves to PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_f1"], label="Train Macro F1")
    axes[1].plot(epochs, history["val_f1"], label="Val Macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1 Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("M5 HeteroAuditGNN — Training Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {out_path}")


def main():
    print("=" * 60)
    print("M5 — HeteroAuditGNN Training")
    print("=" * 60)

    metrics, model, splits = train_model(verbose=True)
    history = metrics.pop("history")

    # Save training curves
    save_training_curves(history)

    # Save metrics
    _METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(str(_METRICS_OUT), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Test metrics saved to {_METRICS_OUT}")

    # Print summary
    print()
    print("=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"  Params   : {metrics['params']:,}")
    print(f"  Runtime  : {metrics['runtime_sec']:.1f}s")
    print()
    print("  Per-class F1:")
    for cls, f1 in metrics["per_class_f1"].items():
        print(f"    {cls:<30s}: {f1:.4f}")
    print()
    print("  Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print("   ", row)

    return metrics


if __name__ == "__main__":
    main()
