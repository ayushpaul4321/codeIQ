"""
SprintGuard - Triage Service

Wraps the DevAssignmentMLP PyTorch model and dev_label_map.json to provide
top-k developer predictions for a given bug embedding.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TriageService:
    """
    Loads and wraps DevAssignmentMLP for inference.

    Usage:
        svc = TriageService(model_path="...", label_map_path="...")
        svc.load()
        predictions = svc.predict(embedding, top_k=3)
    """

    def __init__(self, model_path: str, label_map_path: str) -> None:
        self._model_path = model_path
        self._label_map_path = label_map_path
        self._model: object | None = None
        # index -> dev name (inverted from the JSON which is dev name -> index)
        self._index_to_dev: dict[int, str] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Deserialize mlp_classifier.pt and dev_label_map.json from disk.

        The label map JSON uses the format:
            { "developer@email.com": int_index, ... }
        We invert it to  { int_index: "developer@email.com" }  for fast
        lookup by predicted class index.

        Raises:
            FileNotFoundError: if either asset file is missing.
            RuntimeError: if the checkpoint cannot be loaded.
        """
        from app.models.dev_assignment import DevAssignmentMLP

        # --- Load label map (dev_name -> index, then invert) ---
        with open(self._label_map_path, "r", encoding="utf-8") as f:
            name_to_idx: dict[str, int] = json.load(f)

        self._index_to_dev = {idx: name for name, idx in name_to_idx.items()}
        num_devs = len(self._index_to_dev)
        logger.info("Label map loaded: %d developers", num_devs)

        # --- Load PyTorch model checkpoint ---
        checkpoint = torch.load(
            self._model_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        # checkpoint may be a state_dict or {"model_state_dict": ..., "num_devs": ...}
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # prefer stored num_devs over label-map count if present
            if "num_devs" in checkpoint:
                num_devs = checkpoint["num_devs"]
        elif isinstance(checkpoint, dict) and all(
            isinstance(v, torch.Tensor) for v in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise RuntimeError(
                f"Unrecognised checkpoint format in {self._model_path!r}. "
                "Expected a state_dict or dict with 'model_state_dict' key."
            )

        model = DevAssignmentMLP(input_dim=384, num_devs=num_devs)
        model.load_state_dict(state_dict)
        model.eval()

        self._model = model
        self._loaded = True
        logger.info(
            "TriageService loaded: model=%s, num_devs=%d",
            self._model_path,
            num_devs,
        )

    def predict(self, embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Return top-k developer predictions sorted descending by probability.

        Args:
            embedding: 1-D numpy array of shape (384,) — L2-normalised embedding.
            top_k:     Number of predictions to return.

        Returns:
            List of dicts, each with keys ``dev`` (str) and ``probability`` (float),
            sorted descending by probability.

        Raises:
            RuntimeError: if called before :meth:`load`.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("TriageService.predict() called before load()")

        # Ensure 2-D tensor: (1, 384)
        tensor = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0)

        indices_batch, probs_batch = self._model.predict_top_k(tensor, k=top_k)

        # predict_top_k returns lists-of-lists for batch dim
        indices = indices_batch[0]
        probs = probs_batch[0]

        results = []
        for idx, prob in zip(indices, probs):
            dev_name = self._index_to_dev.get(idx, f"dev_{idx}")
            results.append({"dev": dev_name, "probability": float(prob)})

        # Already sorted descending by predict_top_k, but be explicit
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True only after a successful call to :meth:`load`."""
        return self._loaded
