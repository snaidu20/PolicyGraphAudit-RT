"""
classifier.py — OPP-115 multi-label segment classifier.

Uses sentence-transformers/all-MiniLM-L6-v2 embeddings + sklearn
OneVsRestClassifier(LogisticRegression).  No fine-tuning of the transformer.

Public API:
  train_and_save(output_path)        Train, evaluate, save model.
  load_classifier(model_path)        Load saved model.
  predict_categories(text, model)    Return [(category_name, prob), ...].

Run this module directly to train: python -m m2_policy_graph.classifier
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# OPP-115 category names (index 0–9, from Wilson et al. 2016)
OPP115_CATEGORIES: List[str] = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "User Access, Edit and Deletion",
    "Data Retention",
    "Data Security",
    "Policy Change",
    "Do Not Track",
    "International and Specific Audiences",
    "Other",
]

# Short slug versions for compact display
OPP115_SLUGS: List[str] = [
    "first_party_collection",
    "third_party_sharing",
    "user_choice_control",
    "user_access_edit_deletion",
    "data_retention",
    "data_security",
    "policy_change",
    "do_not_track",
    "international_audiences",
    "other",
]

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_opp115_dataset() -> Tuple[List[str], List[List[int]]]:
    """
    Load the alzoubi36/opp_115 dataset from HuggingFace.

    Returns (texts, labels) where labels[i] is a list of ints 0–9.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("alzoubi36/opp_115", trust_remote_code=True)

    texts: List[str] = []
    labels: List[List[int]] = []

    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        for row in ds[split]:
            text = row.get("text", "") or row.get("segment_text", "")
            label = row.get("label", row.get("labels", []))
            if isinstance(label, int):
                label = [label]
            elif not isinstance(label, list):
                label = list(label)
            if text:
                texts.append(text)
                labels.append(label)

    return texts, labels


def _embed(texts: List[str], model_name: str = _EMBED_MODEL_NAME) -> np.ndarray:
    """Embed a list of texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


# ---------------------------------------------------------------------------
# Public training function
# ---------------------------------------------------------------------------

def train_and_save(
    output_path: Path | str,
    model_name: str = _EMBED_MODEL_NAME,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict:
    """
    Train the OPP-115 classifier and save to output_path (.pkl).

    Returns a dict with val_f1_macro, test_f1_macro, per_category_f1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading OPP-115 dataset from HuggingFace...")
    texts, labels = _load_opp115_dataset()
    print(f"  {len(texts)} segments loaded.")

    print(f"Embedding with {model_name}...")
    X = _embed(texts, model_name)

    # MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=list(range(len(OPP115_CATEGORIES))))
    Y = mlb.fit_transform(labels)

    # Train / val / test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size + val_size, random_state=random_state, shuffle=True
    )
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, test_size=1 - val_frac, random_state=random_state
    )

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Train classifier
    print("Training OneVsRestClassifier(LogisticRegression)...")
    clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            C=4.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_state,
        ),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)

    # Evaluate on val
    Y_val_pred = clf.predict(X_val)
    val_f1_macro = f1_score(Y_val, Y_val_pred, average="macro", zero_division=0)
    print(f"  Val macro F1: {val_f1_macro:.4f}")

    # Evaluate on test
    Y_test_pred = clf.predict(X_test)
    test_f1_macro = f1_score(Y_test, Y_test_pred, average="macro", zero_division=0)
    per_cat_f1 = f1_score(Y_test, Y_test_pred, average=None, zero_division=0)

    print(f"\n--- Test Results ---")
    print(f"Macro F1: {test_f1_macro:.4f}")
    print("\nPer-category F1:")
    for cat_name, f1 in zip(OPP115_CATEGORIES, per_cat_f1):
        print(f"  {cat_name:<42} {f1:.3f}")
    print()
    print(classification_report(
        Y_test, Y_test_pred,
        target_names=[c[:30] for c in OPP115_CATEGORIES],
        zero_division=0,
    ))

    # Save model + binarizer
    payload = {
        "clf": clf,
        "mlb": mlb,
        "model_name": model_name,
        "categories": OPP115_CATEGORIES,
        "slugs": OPP115_SLUGS,
        "val_f1_macro": float(val_f1_macro),
        "test_f1_macro": float(test_f1_macro),
        "per_cat_f1": {cat: float(f1) for cat, f1 in zip(OPP115_CATEGORIES, per_cat_f1)},
    }
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=5)
    print(f"Saved classifier to {output_path}")

    return {
        "val_f1_macro": float(val_f1_macro),
        "test_f1_macro": float(test_f1_macro),
        "per_category_f1": payload["per_cat_f1"],
    }


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

class OPP115Classifier:
    """Wrapper around the trained sklearn classifier + sentence encoder."""

    def __init__(self, payload: dict) -> None:
        self._clf = payload["clf"]
        self._mlb = payload["mlb"]
        self._categories = payload["categories"]
        self._slugs = payload["slugs"]
        self._model_name = payload["model_name"]
        self._encoder: Optional[object] = None  # lazy init

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def predict_categories(self, text: str) -> List[Tuple[str, float]]:
        """
        Return [(category_name, probability), ...] for non-zero classes,
        sorted descending by probability.
        """
        encoder = self._get_encoder()
        emb = encoder.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        # Probability estimates for each class via OneVsRest
        proba = self._clf.predict_proba(emb)[0]  # shape (n_classes,)
        results = [
            (cat, float(p))
            for cat, p in zip(self._categories, proba)
            if p > 0.0
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def predict_top_category(self, text: str) -> str:
        """Return the highest-probability category name."""
        preds = self.predict_categories(text)
        if not preds:
            return "Other"
        return preds[0][0]


def load_classifier(model_path: Path | str) -> OPP115Classifier:
    """Load a saved classifier from disk."""
    with open(model_path, "rb") as fh:
        payload = pickle.load(fh)
    return OPP115Classifier(payload)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy) — loaded on first call to predict_categories()
# ---------------------------------------------------------------------------

_SINGLETON: Optional[OPP115Classifier] = None


def _get_singleton(model_path: Path | None = None) -> OPP115Classifier:
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    if model_path is None:
        # Resolve from standard project path
        here = Path(__file__).resolve()
        for p in [here.parent, here.parent.parent, here.parent.parent.parent]:
            candidate = p / "data" / "processed" / "opp115_classifier.pkl"
            if candidate.exists():
                model_path = candidate
                break
            candidate = p.parent / "data" / "processed" / "opp115_classifier.pkl"
            if candidate.exists():
                model_path = candidate
                break
    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            "opp115_classifier.pkl not found. Run: "
            "python -m m2_policy_graph.classifier"
        )
    _SINGLETON = load_classifier(model_path)
    return _SINGLETON


def predict_categories(text: str) -> List[Tuple[str, float]]:
    """
    Module-level convenience function.
    Returns [(category_name, prob), ...] sorted by descending probability.
    Loads the classifier singleton on first call.
    """
    return _get_singleton().predict_categories(text)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Resolve project root
    here = Path(__file__).resolve()
    project_root = here.parent.parent.parent
    output = project_root / "data" / "processed" / "opp115_classifier.pkl"

    results = train_and_save(output)
    print("\nSummary:")
    print(f"  Val macro F1:  {results['val_f1_macro']:.4f}")
    print(f"  Test macro F1: {results['test_f1_macro']:.4f}")
