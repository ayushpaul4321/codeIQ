"""
SprintGuard - LSTM Effort Estimator

Architecture:
    Input  : (batch, seq_len=1, 384) — bug embedding
    Output : scalar prediction in log-hours

Service:
    EffortEstimatorService  — loads model, runs 30-sample Monte Carlo Dropout
                              to produce a point estimate + 5th/95th CI in hours.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class EffortEstimatorLSTM(nn.Module):
    """
    2-layer LSTM for bug effort estimation.

    Input:  (batch, seq_len=1, input_dim=384)
    Output: (batch, 1) — predicted log-hours
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len=1, input_dim=384)

        Returns:
            Tensor of shape (batch, 1) — log-hours predictions.
        """
        # output: (batch, seq_len, hidden), h_n: (num_layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)
        # Take last layer's hidden state: (batch, hidden)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)  # (batch, 1)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class EffortEstimatorService:
    """
    Wraps EffortEstimatorLSTM for inference with Monte Carlo Dropout.

    Usage:
        svc = EffortEstimatorService(model_path="storage/models/effort_estimator/lstm_estimator.pt")
        svc.load()
        result = svc.predict(embedding)
        # {"hours": 4.2, "confidence_interval": [2.1, 8.5]}
    """

    _MC_SAMPLES = 30

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model: EffortEstimatorLSTM | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load lstm_estimator.pt from disk.

        Raises:
            FileNotFoundError: if the model file is absent.
            RuntimeError:      if the checkpoint format is unrecognised.
        """
        checkpoint = torch.load(
            self._model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        # Accept bare state_dict or wrapped {"model_state_dict": ...}
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise RuntimeError(
                f"Unrecognised checkpoint format in {self._model_path!r}. "
                "Expected a state_dict or dict with 'model_state_dict' key."
            )

        model = EffortEstimatorLSTM()
        model.load_state_dict(state_dict)
        model.eval()

        self._model = model
        self._loaded = True
        logger.info("EffortEstimatorService loaded: %s", self._model_path)

    def predict(self, embedding: np.ndarray) -> dict:
        """
        Predict effort in hours using 30-sample Monte Carlo Dropout.

        Args:
            embedding: 1-D numpy array of shape (384,).

        Returns:
            dict with keys:
                "hours"               — float, mean predicted hours
                "confidence_interval" — [float, float], 5th and 95th percentile hours

        Raises:
            RuntimeError: if called before load().
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("EffortEstimatorService.predict() called before load()")

        # Shape: (1, 1, 384) — batch=1, seq_len=1, input_dim=384
        tensor = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        log_hour_samples: list[float] = []

        for _ in range(self._MC_SAMPLES):
            # Enable dropout for Monte Carlo sampling
            self._model.train()
            with torch.no_grad():
                out = self._model(tensor)  # (1, 1)
            log_hour_samples.append(out.item())

        # Restore eval mode after sampling
        self._model.eval()

        samples_np = np.array(log_hour_samples)

        # Convert log-hours → hours
        hours_samples = np.exp(samples_np)

        mean_hours = float(np.mean(hours_samples))
        ci_low = float(np.percentile(hours_samples, 5))
        ci_high = float(np.percentile(hours_samples, 95))

        return {
            "hours": mean_hours,
            "confidence_interval": [ci_low, ci_high],
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True only after a successful call to :meth:`load`."""
        return self._loaded
