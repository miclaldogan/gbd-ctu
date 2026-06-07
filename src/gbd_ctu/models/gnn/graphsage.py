"""GBD-CTU GraphSAGE classifier.

This module defines a GraphSAGE-based node classifier. Input is a
torch_geometric.data.Data object; output is a tensor of node logits with shape
[num_nodes, num_classes].

Architecture (default 3-layer, hidden_channels=128):
    SAGEConv(in  →  128) → BatchNorm → ReLU → Dropout(0.3)
    SAGEConv(128 →   64) → BatchNorm → ReLU → Dropout(0.3)
    SAGEConv( 64 →   32) → BatchNorm → ReLU
    Linear(32 → out_channels)

Each hidden layer halves the channel width, giving a pyramid encoder.
The number of layers, initial hidden width, and dropout are all configurable
via ``configs/gnn.yaml``.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.nn import SAGEConv
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    nn = type("nn", (), {"Module": object})()  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]


class GraphSAGENodeClassifier(nn.Module):
    """Multi-layer GraphSAGE node classifier with per-layer BatchNorm.

    Hidden channel width halves with every layer (pyramid encoder), so for
    ``hidden_channels=128`` and ``num_layers=3`` the widths are
    128 → 64 → 32.

    Parameters
    ----------
    in_channels:
        Dimensionality of the input node features.
    hidden_channels:
        Width of the first hidden layer.  Subsequent layers use
        ``hidden_channels >> i`` (right-shift = integer halving).
    out_channels:
        Number of output classes (default 2 for binary botnet detection).
    num_layers:
        Total number of SAGEConv layers (must be ≥ 1).
    dropout:
        Dropout probability applied after every layer except the last conv.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        if torch is None or functional is None or SAGEConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate GraphSAGENodeClassifier.")
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        # Build channel widths: [in, h, h//2, h//4, ...]
        channel_dims = [in_channels] + [max(1, hidden_channels >> i) for i in range(num_layers)]

        self.convolutions = nn.ModuleList(
            SAGEConv(channel_dims[i], channel_dims[i + 1]) for i in range(num_layers)
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(channel_dims[i + 1]) for i in range(num_layers)
        )
        self.dropout_prob = dropout
        self.classifier = nn.Linear(channel_dims[-1], out_channels)

        self.num_parameters: int = sum(p.numel() for p in self.parameters())
        _logger.info(
            "GraphSAGENodeClassifier | layers=%d  hidden_channels=%d  "
            "dropout=%.2f  total_params=%d",
            num_layers,
            hidden_channels,
            dropout,
            self.num_parameters,
        )

    def forward(self, data) -> "torch.Tensor":
        """Compute per-node logits.

        Parameters
        ----------
        data:
            A ``torch_geometric.data.Data`` object with attributes ``x``
            (node features) and ``edge_index``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, out_channels]``.
        """
        x, edge_index = data.x, data.edge_index
        last = len(self.convolutions) - 1
        for i, (conv, bn) in enumerate(zip(self.convolutions, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = functional.relu(x)
            if i < last:
                x = functional.dropout(x, p=self.dropout_prob, training=self.training)
        return self.classifier(x)
