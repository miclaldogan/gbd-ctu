"""GBD-CTU GraphSAGE classifier.

This module defines a GraphSAGE-based node classifier. Input is a
torch_geometric.data.Data object; output is a tensor of node logits with shape
[num_nodes, num_classes].
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.nn import SAGEConv
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    functional = None
    nn = object
    SAGEConv = None


class GraphSAGENodeClassifier(nn.Module):
    """A compact multi-layer GraphSAGE classifier for IP-node prediction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        if torch is None or functional is None or SAGEConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate GraphSAGE.")
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2 for GraphSAGENodeClassifier.")
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.layers.append(SAGEConv(hidden_channels, hidden_channels))
        self.normalization = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data) -> torch.Tensor:
        """Compute node logits for a PyG graph batch."""

        x = data.x
        edge_index = data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = functional.relu(x)
            x = self.normalization(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
