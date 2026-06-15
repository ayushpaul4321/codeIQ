"""
SprintGuard - Bug Embedding Module
Wraps sentence-transformers to produce 384-dim bug embeddings.
Model: all-MiniLM-L6-v2  (6x faster than full BERT, near-identical accuracy)
"""

from __future__ import annotations
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM  = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load model once and cache it in memory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BugEmbedding] Loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    return model


def embed_bugs(texts: list[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
    """
    Embed a list of bug description strings.

    Args:
        texts:      List of cleaned bug text strings.
        batch_size: How many to encode per GPU batch.
        normalize:  L2-normalize embeddings (recommended for cosine similarity).

    Returns:
        numpy array of shape (N, EMBED_DIM)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 500,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings


def embed_single(text: str, normalize: bool = True) -> np.ndarray:
    """Embed a single bug description. Returns shape (EMBED_DIM,)."""
    return embed_bugs([text], normalize=normalize)[0]


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(v1, v2))


class BugEmbeddingService:
    """
    Service wrapper around the embedding utilities.
    Provides a single source of truth for the enriched text format used at
    both training time (preprocess.py stage_features) and inference time (API).
    """

    @staticmethod
    def build_enriched_text(
        title: str,
        description: str,
        product: str = "unknown",
        component: str = "unknown",
    ) -> str:
        """
        Build the canonical enriched text string for embedding.

        Format: "product: {product} component: {component} summary: {title} {description}"

        This format must match the training-time template in preprocess.py stage_features()
        exactly so that training and inference embeddings are produced from identical inputs.

        Args:
            title:       Bug title / summary.
            description: Bug description body.
            product:     Bug product field (defaults to "unknown" at inference time).
            component:   Bug component field (defaults to "unknown" at inference time).

        Returns:
            Enriched text string ready to be passed to embed_single() or embed_bugs().
        """
        return f"product: {product} component: {component} summary: {title} {description}"

    def embed(self, title: str, description: str, product: str = "unknown",
              component: str = "unknown", normalize: bool = True) -> np.ndarray:
        """Convenience method: build enriched text and embed it in one call."""
        text = self.build_enriched_text(title, description, product, component)
        return embed_single(text, normalize=normalize)
