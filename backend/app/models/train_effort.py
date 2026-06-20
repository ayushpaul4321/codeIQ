"""
SprintGuard - LSTM Effort Estimator Training Script

Standalone script — NOT imported by FastAPI.

Usage:
    python -m backend.app.models.train_effort
    # or from the repo root:
    python backend/app/models/train_effort.py

Outputs:
    storage/models/effort_estimator/lstm_estimator.pt
    storage/models/effort_estimator/effort_metrics.json
    storage/datasets/processed/effort_dataset.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

CLEANED_CSV = os.path.join(REPO_ROOT, "storage", "datasets", "cleaned", "bugs_clean.csv")
EFFORT_CSV = os.path.join(REPO_ROOT, "storage", "datasets", "processed", "effort_dataset.csv")
MODEL_DIR = os.path.join(REPO_ROOT, "storage", "models", "effort_estimator")
MODEL_OUT = os.path.join(MODEL_DIR, "lstm_estimator.pt")
METRICS_OUT = os.path.join(MODEL_DIR, "effort_metrics.json")


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10          # early stopping patience
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1.  Build effort dataset
# ---------------------------------------------------------------------------

def build_effort_dataset() -> pd.DataFrame:
    """Load cleaned CSV, compute fix_hours, filter outliers, return DataFrame."""
    logger.info("Loading cleaned bugs from %s", CLEANED_CSV)
    df = pd.read_csv(CLEANED_CSV)

    df["creation_time"] = pd.to_datetime(df["creation_time"], utc=True)
    df["last_change_time"] = pd.to_datetime(df["last_change_time"], utc=True)
    df["fix_hours"] = (
        (df["last_change_time"] - df["creation_time"]).dt.total_seconds() / 3600
    )

    before = len(df)
    df = df[(df["fix_hours"] >= 0.5) & (df["fix_hours"] <= 80)]
    logger.info(
        "After filtering outliers: %d rows (dropped %d)", len(df), before - len(df)
    )

    if len(df) < 300:
        logger.warning(
            "Dataset too small after filtering: only %d rows (need >= 300). Aborting.",
            len(df),
        )
        sys.exit(1)

    df = df[["id", "text", "fix_hours"]].reset_index(drop=True)

    # Save effort dataset
    os.makedirs(os.path.dirname(EFFORT_CSV), exist_ok=True)
    df.to_csv(EFFORT_CSV, index=False)
    logger.info("Effort dataset saved to %s (%d rows)", EFFORT_CSV, len(df))

    return df


# ---------------------------------------------------------------------------
# 2.  Embed texts
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Embed bug texts directly using embed_bugs().
    Re-embedding the effort dataset texts is simpler and more reliable
    than trying to align with a potentially stale embeddings cache.
    """
    # Adjust sys.path so we can import from the models package regardless of
    # where the script is invoked from.
    models_dir = os.path.dirname(os.path.abspath(__file__))
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    from bug_embedding import embed_bugs  # type: ignore[import]

    logger.info("Embedding %d bug texts (this may take a few minutes)…", len(texts))
    embeddings = embed_bugs(texts, batch_size=128, normalize=True)
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# 3.  Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    from effort_estimator import EffortEstimatorLSTM  # type: ignore[import]
    return EffortEstimatorLSTM(input_dim=384, hidden=128, num_layers=2, dropout=0.2)


# ---------------------------------------------------------------------------
# 4.  Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """
    Compute MMRE and PRED(25) from log-hour predictions.

    Args:
        y_true_log: ground-truth log-hours
        y_pred_log: predicted log-hours

    Returns:
        {"mmre": float, "pred25": float}
    """
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    # Avoid division by zero
    rel_errors = np.abs(y_pred - y_true) / (y_true + 1e-8)
    mmre = float(np.mean(rel_errors))
    pred25 = float(np.mean(rel_errors <= 0.25))
    return {"mmre": mmre, "pred25": pred25}


# ---------------------------------------------------------------------------
# 5.  Training loop
# ---------------------------------------------------------------------------

def train_model(
    X: np.ndarray, y: np.ndarray
) -> tuple[nn.Module, dict]:
    """
    Full train/val/test split → train LSTM with early stopping → evaluate.

    Returns:
        (trained_model, metrics_dict)
    """
    # Log-transform targets
    y_log = np.log(y.astype(np.float32) + 1e-8)

    # 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_log, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    logger.info(
        "Splits — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )

    # Build tensors — shape (N, 1, 384) for LSTM seq_len=1
    def to_tensor(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)

    X_tr = to_tensor(X_train)
    X_vl = to_tensor(X_val)
    X_ts = to_tensor(X_test)
    y_tr = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    y_vl = torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)
    y_ts = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    model = build_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl.to(device))
            val_loss = criterion(val_preds, y_vl.to(device)).item()

        avg_train = float(np.mean(train_losses))
        logger.info(
            "Epoch %3d/%d — train_loss=%.4f  val_loss=%.4f",
            epoch, EPOCHS, avg_train, val_loss,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    # --- Evaluate on test set ---
    with torch.no_grad():
        test_preds = model(X_ts.to(device)).cpu().numpy().flatten()

    y_test_np = y_ts.numpy().flatten()
    metrics = compute_metrics(y_test_np, test_preds)
    logger.info(
        "Test metrics — MMRE=%.4f  PRED(25)=%.4f", metrics["mmre"], metrics["pred25"]
    )

    # PASS / FAIL check
    pass_flag = metrics["mmre"] <= 0.40 and metrics["pred25"] >= 0.50
    if pass_flag:
        logger.info("✅ PASS: MMRE ≤ 0.40 and PRED(25) ≥ 0.50")
    else:
        logger.warning(
            "⚠️  WARN: Quality targets not met — MMRE=%.4f (target ≤ 0.40), "
            "PRED(25)=%.4f (target ≥ 0.50). Model saved anyway.",
            metrics["mmre"], metrics["pred25"],
        )

    return model, metrics


# ---------------------------------------------------------------------------
# 6.  Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(model: nn.Module, metrics: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), MODEL_OUT)
    logger.info("Model saved to %s", MODEL_OUT)

    # Save metrics
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_OUT)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure models directory is on path for local imports
    models_dir = os.path.dirname(os.path.abspath(__file__))
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)

    logger.info("=== SprintGuard LSTM Effort Estimator — Training ===")

    # 1. Build dataset
    df = build_effort_dataset()

    # 2. Embed texts
    X = get_embeddings(df["text"].tolist())
    y = df["fix_hours"].values.astype(np.float32)

    # Sanity check shapes
    assert X.shape == (len(df), 384), f"Unexpected embedding shape: {X.shape}"

    # 3. Train
    model, metrics = train_model(X, y)

    # 4. Save
    save_artefacts(model, metrics)

    logger.info("=== Training complete ===")
