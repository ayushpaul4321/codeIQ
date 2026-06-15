"""
SprintGuard - TF-IDF + SVM Baseline Training Script
Research comparison for Q1: Does BERT+MLP outperform a classical approach?

Usage:
    python backend/app/models/train_baseline.py

Outputs:
    storage/models/tfidf_svm/tfidf_svm_pipeline.joblib
    storage/models/tfidf_svm/baseline_metrics.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import top_k_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

# ── paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR   = os.path.join(ROOT, "storage", "datasets", "splits")
OUTPUT_DIR   = os.path.join(ROOT, "storage", "models", "tfidf_svm")
PIPELINE_PATH = os.path.join(OUTPUT_DIR, "tfidf_svm_pipeline.joblib")
METRICS_PATH  = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
BERT_METRICS  = os.path.join(ROOT, "storage", "models", "bert_triage", "training_metrics.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def predict_top3(text: str, pipeline: Pipeline) -> list[dict]:
    """
    Return the top-3 developer predictions for a raw bug text string.

    Args:
        text:     Raw bug text (will be passed directly to the TF-IDF vectorizer).
        pipeline: A fitted sklearn Pipeline with a TfidfVectorizer + LinearSVC.

    Returns:
        List of up to 3 dicts sorted by descending probability:
        [{"dev": str, "probability": float}, ...]
    """
    svc = pipeline.named_steps["svc"]
    classes = svc.classes_

    decision_scores = pipeline.decision_function([text])  # shape (1, n_classes)
    probs = softmax(decision_scores, axis=1)[0]           # shape (n_classes,)

    top_k = min(3, len(classes))
    top_indices = np.argsort(probs)[::-1][:top_k]

    return [
        {"dev": str(classes[i]), "probability": float(round(probs[i], 4))}
        for i in top_indices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pipeline: Pipeline, texts: list[str], labels: np.ndarray,
                    all_classes: np.ndarray) -> dict[str, float]:
    """Compute Top-1, Top-3 accuracy and MRR using decision_function + softmax."""
    decision_scores = pipeline.decision_function(texts)   # (N, n_classes)
    probs = softmax(decision_scores, axis=1)               # (N, n_classes)

    num_classes = len(all_classes)
    top1 = top_k_accuracy_score(labels, probs, k=1, labels=range(num_classes))
    top3 = top_k_accuracy_score(labels, probs, k=min(3, num_classes),
                                 labels=range(num_classes))

    # MRR: rank of the true label in the sorted prediction list
    ranks_desc = np.argsort(-probs, axis=1)
    rr = []
    for i, true_label in enumerate(labels):
        rank_pos = np.where(ranks_desc[i] == true_label)[0]
        if len(rank_pos) > 0:
            rr.append(1.0 / (rank_pos[0] + 1))
        else:
            rr.append(0.0)
    mrr = float(np.mean(rr))

    return {
        "top1": round(top1, 4),
        "top3": round(top3, 4),
        "mrr":  round(mrr, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train() -> dict[str, Any]:
    print("=" * 60)
    print("  SprintGuard — TF-IDF + SVM Baseline Training")
    print("=" * 60)

    # ── Load splits ───────────────────────────────────────────────────────────
    print("\n[Data] Loading splits...")
    train_path = os.path.join(SPLITS_DIR, "train.csv")
    test_path  = os.path.join(SPLITS_DIR, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error("Split CSVs not found — run storage/datasets/preprocess.py first.")
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # dev_id is an integer label encoded by preprocess.py; reconstruct label space
    # using the union of train + test so the label indices are consistent
    all_dev_ids = np.sort(
        np.unique(np.concatenate([df_train["dev_id"].values, df_test["dev_id"].values]))
    )
    # Re-map to contiguous 0..N-1 (in case of gaps after filtering)
    dev_id_to_idx = {dev_id: idx for idx, dev_id in enumerate(all_dev_ids)}
    # dev_name lookup: assigned_to column
    dev_id_to_name: dict[int, str] = {}
    for _, row in pd.concat([df_train, df_test]).drop_duplicates("dev_id").iterrows():
        dev_id_to_name[int(row["dev_id"])] = str(row["assigned_to"])

    train_texts  = df_train["text"].fillna("").tolist()
    test_texts   = df_test["text"].fillna("").tolist()
    train_labels = np.array([dev_id_to_idx[d] for d in df_train["dev_id"].values])
    test_labels  = np.array([dev_id_to_idx[d] for d in df_test["dev_id"].values])
    num_classes  = len(all_dev_ids)

    print(f"  Train: {len(df_train):,}  Test: {len(df_test):,}")
    print(f"  Developers: {num_classes}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    # The SVC classes_ attribute will correspond to remapped integer indices.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("svc", LinearSVC(C=1.0, max_iter=5000)),
    ])

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[Train] Fitting TF-IDF + LinearSVC...")
    t0 = time.time()
    pipeline.fit(train_texts, train_labels)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[Eval] Computing metrics on test set...")
    all_classes = pipeline.named_steps["svc"].classes_
    metrics = compute_metrics(pipeline, test_texts, test_labels, all_classes)

    print(f"\n{'='*60}")
    print(f"  TF-IDF + SVM BASELINE TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy : {metrics['top1']:.3f}  ({metrics['top1']*100:.1f}%)")
    print(f"  Top-3 Accuracy : {metrics['top3']:.3f}  ({metrics['top3']*100:.1f}%)")
    print(f"  MRR            : {metrics['mrr']:.3f}")
    print(f"{'='*60}\n")

    # ── Save pipeline ─────────────────────────────────────────────────────────
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline saved → {PIPELINE_PATH}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved  → {METRICS_PATH}")

    # ── Compare against BERT+MLP ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON: BERT+MLP  vs  TF-IDF+SVM")
    print(f"{'='*60}")

    if os.path.exists(BERT_METRICS):
        with open(BERT_METRICS) as f:
            bert_data = json.load(f)
        bert = bert_data.get("test_metrics", {})
        bert_top3 = bert.get("top3", None)
        bert_mrr  = bert.get("mrr",  None)

        baseline_top3 = metrics["top3"]
        baseline_mrr  = metrics["mrr"]

        print(f"  {'Metric':<18} {'BERT+MLP':>10} {'TF-IDF+SVM':>12} {'Delta':>8}")
        print(f"  {'-'*50}")
        if bert_top3 is not None:
            delta_top3 = bert_top3 - baseline_top3
            print(f"  {'Top-3 Accuracy':<18} {bert_top3:>10.3f} {baseline_top3:>12.3f} {delta_top3:>+8.3f}")
        if bert_mrr is not None:
            delta_mrr = bert_mrr - baseline_mrr
            print(f"  {'MRR':<18} {bert_mrr:>10.3f} {baseline_mrr:>12.3f} {delta_mrr:>+8.3f}")
        print(f"{'='*60}\n")

        # Assert BERT beats baseline by ≥ 10 percentage points on Top-3
        if bert_top3 is not None:
            MIN_ADVANTAGE = 0.10
            advantage = bert_top3 - baseline_top3
            if advantage >= MIN_ADVANTAGE:
                print(
                    f"PASS: BERT Top-3 ({bert_top3:.3f}) exceeds baseline ({baseline_top3:.3f}) "
                    f"by {advantage*100:.1f}pp ≥ 10pp required."
                )
            else:
                shortfall = MIN_ADVANTAGE - advantage
                logging.error(
                    json.dumps({
                        "check": "bert_vs_baseline_top3",
                        "bert_top3": bert_top3,
                        "baseline_top3": baseline_top3,
                        "advantage_pp": round(advantage * 100, 2),
                        "required_pp": 10.0,
                        "shortfall_pp": round(shortfall * 100, 2),
                    })
                )
                print(
                    f"FAIL: BERT Top-3 ({bert_top3:.3f}) exceeds baseline ({baseline_top3:.3f}) "
                    f"by only {advantage*100:.1f}pp — need ≥ 10pp. "
                    f"Shortfall: {shortfall*100:.1f}pp."
                )
                # Not a hard exit — the training artifact is still valid; caller can decide
    else:
        logging.warning(
            f"BERT metrics file not found at {BERT_METRICS}. "
            "Run train_triage.py first for a comparison."
        )
        print(f"  (BERT metrics not found — skipping comparison)\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
