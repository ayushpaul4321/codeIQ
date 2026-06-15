"""
SprintGuard - Triage Model Training Script
Week 2 core task: train MLP on BERT embeddings for dev assignment.

Usage:
    python backend/app/models/train_triage.py

Outputs:
    storage/models/bert_triage/mlp_classifier.pt
    storage/models/bert_triage/training_metrics.json
"""

from __future__ import annotations
import os, sys, json, time, logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── path setup so we can import project modules ──────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from backend.app.models.bug_embedding import embed_bugs, EMBED_DIM
from backend.app.models.dev_assignment import DevAssignmentMLP

# ── paths ─────────────────────────────────────────────────────────────────────
SPLITS_DIR  = os.path.join(ROOT, "storage", "datasets", "splits")
MODELS_DIR  = os.path.join(ROOT, "storage", "models",   "bert_triage")
EMBED_CACHE = os.path.join(ROOT, "storage", "models",   "bert_triage", "embeddings_cache.npz")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── hyper-parameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 256
EPOCHS       = 40
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BugDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding (with cache so we don't re-run BERT every time)
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(df_train, df_val, df_test) -> tuple:
    """Generate or load cached embeddings."""
    if os.path.exists(EMBED_CACHE):
        print("[Embed] Loading cached embeddings...")
        cache = np.load(EMBED_CACHE)
        return cache["train"], cache["val"], cache["test"]

    print(f"[Embed] Generating embeddings on {DEVICE}...")
    t0 = time.time()
    e_train = embed_bugs(df_train["text"].tolist())
    e_val   = embed_bugs(df_val["text"].tolist())
    e_test  = embed_bugs(df_test["text"].tolist())
    print(f"[Embed] Done in {time.time()-t0:.1f}s")

    np.savez_compressed(EMBED_CACHE, train=e_train, val=e_val, test=e_test)
    print(f"[Embed] Cached → {EMBED_CACHE}")
    return e_train, e_val, e_test


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: DevAssignmentMLP, loader: DataLoader, num_classes: int) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels)

    top1 = top_k_accuracy_score(labels, logits, k=1, labels=range(num_classes))
    top3 = top_k_accuracy_score(labels, logits, k=3, labels=range(num_classes))

    # MRR
    ranks = np.argsort(-logits, axis=1)
    rr = []
    for i, label in enumerate(labels):
        rank = np.where(ranks[i] == label)[0][0] + 1
        rr.append(1.0 / rank)
    mrr = float(np.mean(rr))

    return {"top1": round(top1, 4), "top3": round(top3, 4), "mrr": round(mrr, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  SprintGuard — Triage Model Training")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # ── Safety net: delete stale embedding cache ──────────────────────────────
    CACHE_PATH = os.path.join(ROOT, "storage", "models", "bert_triage", "embeddings_cache.npz")
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
        print(f"[Cache] Deleted stale embedding cache → {CACHE_PATH}")

    # ── Check if split CSVs are stale or absent; re-run preprocessing if so ───
    CLEANED_CSV  = os.path.join(ROOT, "storage", "datasets", "cleaned", "bugs_clean.csv")
    split_paths  = [
        os.path.join(SPLITS_DIR, "train.csv"),
        os.path.join(SPLITS_DIR, "val.csv"),
        os.path.join(SPLITS_DIR, "test.csv"),
    ]
    splits_stale = any(not os.path.exists(p) for p in split_paths)
    if not splits_stale and os.path.exists(CLEANED_CSV):
        clean_mtime = os.path.getmtime(CLEANED_CSV)
        splits_stale = any(os.path.getmtime(p) < clean_mtime for p in split_paths)

    if splits_stale:
        print("\n[Preprocess] Split CSVs are stale or absent — regenerating...")
        # Import and run all preprocessing stages
        import importlib.util
        preprocess_path = os.path.join(ROOT, "storage", "datasets", "preprocess.py")
        spec = importlib.util.spec_from_file_location("preprocess", preprocess_path)
        preprocess = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess)
        preprocess.stage_clean()
        preprocess.stage_filter()
        preprocess.stage_features()
        preprocess.stage_split()
        print("[Preprocess] Splits regenerated successfully.\n")
    else:
        print("\n[Preprocess] Split CSVs are up-to-date — skipping preprocessing.")

    # ── Load splits ───────────────────────────────────────────────────────────
    print("\n[Data] Loading splits...")
    df_train = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    df_val   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    df_test  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

    # Re-encode dev_id to be contiguous 0..N-1
    le = LabelEncoder()
    all_devs = pd.concat([df_train, df_val, df_test])["dev_id"].values
    le.fit(all_devs)
    df_train["label"] = le.transform(df_train["dev_id"])
    df_val["label"]   = le.transform(df_val["dev_id"])
    df_test["label"]  = le.transform(df_test["dev_id"])
    num_devs = len(le.classes_)

    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")
    print(f"  Developers: {num_devs}")

    # ── Embeddings ────────────────────────────────────────────────────────────
    e_train, e_val, e_test = get_embeddings(df_train, df_val, df_test)

    train_ds = BugDataset(e_train, df_train["label"].values)
    val_ds   = BugDataset(e_val,   df_val["label"].values)
    test_ds  = BugDataset(e_test,  df_test["label"].values)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DevAssignmentMLP(input_dim=EMBED_DIM, num_devs=num_devs).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[Train] {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}\n")
    best_val_top3 = 0.0
    best_epoch    = 0
    history       = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, num_devs)

        row = {
            "epoch":    epoch,
            "loss":     round(avg_loss, 4),
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_mrr":  val_metrics["mrr"],
            "elapsed":  round(time.time() - t0, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  "
            f"val_top1={val_metrics['top1']:.3f}  "
            f"val_top3={val_metrics['top3']:.3f}  "
            f"mrr={val_metrics['mrr']:.3f}  "
            f"({time.time()-t0:.1f}s)"
        )

        # Save best checkpoint
        if val_metrics["top3"] > best_val_top3:
            best_val_top3 = val_metrics["top3"]
            best_epoch    = epoch
            torch.save({
                "epoch":     epoch,
                "model_state": model.state_dict(),
                "num_devs":  num_devs,
                "input_dim": EMBED_DIM,
                "val_metrics": val_metrics,
            }, os.path.join(MODELS_DIR, "mlp_classifier.pt"))

    print(f"\n[Train] Best val Top-3: {best_val_top3:.3f} at epoch {best_epoch}")

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    print("\n[Test] Loading best checkpoint...")
    ckpt = torch.load(os.path.join(MODELS_DIR, "mlp_classifier.pt"), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, num_devs)

    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy : {test_metrics['top1']:.3f}  ({test_metrics['top1']*100:.1f}%)")
    print(f"  Top-3 Accuracy : {test_metrics['top3']:.3f}  ({test_metrics['top3']*100:.1f}%)")
    print(f"  MRR            : {test_metrics['mrr']:.3f}")
    print(f"{'='*60}\n")

    # ── PASS / FAIL validation ─────────────────────────────────────────────────
    top3_pass = test_metrics["top3"] >= 0.70
    mrr_pass  = test_metrics["mrr"]  >= 0.55
    if top3_pass and mrr_pass:
        print("PASS: Top-3={:.3f} (≥0.70), MRR={:.3f} (≥0.55)".format(test_metrics["top3"], test_metrics["mrr"]))
    else:
        if not top3_pass:
            gap = 0.70 - test_metrics["top3"]
            logging.error(json.dumps({"metric": "top3", "achieved": test_metrics["top3"], "target": 0.70, "gap": round(gap, 4)}))
        if not mrr_pass:
            gap = 0.55 - test_metrics["mrr"]
            logging.error(json.dumps({"metric": "mrr", "achieved": test_metrics["mrr"], "target": 0.55, "gap": round(gap, 4)}))
        print("FAIL: Top-3={:.3f}, MRR={:.3f} — see logs for gap details".format(test_metrics["top3"], test_metrics["mrr"]))

    # ── Save full metrics ─────────────────────────────────────────────────────
    results = {
        "test_metrics": test_metrics,
        "best_val_top3": best_val_top3,
        "best_epoch": best_epoch,
        "num_devs": num_devs,
        "epochs": EPOCHS,
        "history": history,
    }
    metrics_path = os.path.join(MODELS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    print("Model saved  → storage/models/bert_triage/mlp_classifier.pt")


if __name__ == "__main__":
    train()
