"""GBD-CTU training losses.

This module contains custom loss functions for graph node classification. Input
is logits and labels; output is a scalar differentiable loss tensor.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    functional = None
    nn = object


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weighting."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        if torch is None or functional is None:
            raise ImportError("torch is required to instantiate FocalLoss.")
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for logits and integer targets."""

        ce_loss = functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_tensor = torch.where(targets > 0, torch.full_like(pt, self.alpha), torch.full_like(pt, 1 - self.alpha))
            focal_weight = focal_weight * alpha_tensor
        return (focal_weight * ce_loss).mean()
