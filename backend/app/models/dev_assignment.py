"""
SprintGuard - Developer Assignment MLP Classifier
Input  : 384-dim bug embedding
Output : probability over all known developers
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DevAssignmentMLP(nn.Module):
    """
    3-hidden-layer MLP for developer assignment.

    Architecture:
        Input (384) → Linear(512) → BN → ReLU → Dropout(0.3)
                    → Linear(256) → BN → ReLU → Dropout(0.3)
                    → Linear(128) → BN → ReLU → Dropout(0.2)
                    → Linear(num_devs)
    """

    def __init__(self, input_dim: int = 384, num_devs: int = 50, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.67),   # 0.2

            nn.Linear(128, num_devs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits. Apply softmax for probabilities."""
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability distribution over devs."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict_top_k(self, x: torch.Tensor, k: int = 3) -> tuple[list[int], list[float]]:
        """
        Returns top-k developer indices and their probabilities.

        Returns:
            (indices, probabilities) both sorted descending by probability
        """
        probs = self.predict_proba(x)          # (batch, num_devs)
        top_probs, top_idx = torch.topk(probs, k, dim=-1)
        return top_idx.cpu().tolist(), top_probs.cpu().tolist()


    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probability distribution over devs."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict_top_k(self, x: torch.Tensor, k: int = 3) -> tuple[list[int], list[float]]:
        """
        Returns top-k developer indices and their probabilities.

        Returns:
            (indices, probabilities) both sorted descending by probability
        """
        probs = self.predict_proba(x)          # (batch, num_devs)
        top_probs, top_idx = torch.topk(probs, k, dim=-1)
        return top_idx.cpu().tolist(), top_probs.cpu().tolist()
