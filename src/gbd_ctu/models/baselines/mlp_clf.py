"""GBD-CTU MLP deep tabular baseline.

A multilayer perceptron trained on the same 22-dimensional flow feature
vector used by the GNN models, with no access to graph structure.  Its
role is to isolate the contribution of the graph topology: if the MLP
matches the GNN, graph structure is not adding value.

Architecture (default hidden=[128, 64, 32])
-------------------------------------------
    Linear(in → 128) → BatchNorm1d → ReLU → Dropout
    Linear(128 → 64) → BatchNorm1d → ReLU → Dropout
    Linear(64  → 32) → BatchNorm1d → ReLU
    Linear(32  → 2)                              # raw logits

Total default parameters: ~12 K (comparable param budget to the GNNs).
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False


class MLPClassifier(nn.Module):
    """Feed-forward MLP for node-level binary classification.

    Parameters
    ----------
    in_channels:
        Number of input features (22 for the 22-dim ``FLOW_FEATURE_COLUMNS``
        vector, or the value of ``graph.num_node_features`` at runtime).
    hidden:
        Ordered list of hidden layer widths.  Default ``[128, 64, 32]``
        matches the architecture spec in the issue.
    dropout:
        Dropout probability applied after every hidden layer except the last.
        Set to 0.0 to disable dropout entirely.

    Attributes
    ----------
    net : ``nn.Sequential``
        The full layer stack including the output head.
    n_params : int
        Total trainable parameter count (logged at construction time).
    """

    def __init__(
        self,
        in_channels: int = 22,
        hidden: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch is required to instantiate MLPClassifier.")
        super().__init__()

        if hidden is None:
            hidden = [128, 64, 32]

        layers: list[nn.Module] = []
        prev = in_channels
        for i, width in enumerate(hidden):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            if i < len(hidden) - 1:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 2))

        self.net = nn.Sequential(*layers)
        self.n_params: int = sum(p.numel() for p in self.parameters())
        _logger.info(
            "MLPClassifier: in=%d hidden=%s out=2 | trainable params=%d",
            in_channels, hidden, self.n_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(N, 2)``.

        Parameters
        ----------
        x:
            Float tensor of shape ``(N, in_channels)``.

        Returns
        -------
        Logit tensor of shape ``(N, 2)``.
        """
        return self.net(x)
