"""GBD-CTU GAT classifier.

This module defines a Graph Attention Network variant for node classification on
IP communication graphs. Input is a PyG Data object; output is node-level logits.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.nn import GATConv
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    functional = None
    nn = object
    GATConv = None


class GATNodeClassifier(nn.Module):
    """A two-layer GAT node classifier for CTU-13 IP graphs."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        if torch is None or functional is None or GATConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate GAT.")
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=False)
        self.gat2 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=False)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data) -> torch.Tensor:
        """Compute node logits for a PyG graph batch."""

        x = functional.elu(self.gat1(data.x, data.edge_index))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = functional.elu(self.gat2(x, data.edge_index))
        x = functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
