"""GBD-CTU GraphSAGE + GAT hybrid classifier.

This module combines GraphSAGE aggregation with GAT attention. Input is a PyG
Data object; output is node-level logits for botnet versus benign prediction.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    functional = None
    nn = object
    GATConv = None
    SAGEConv = None


class GraphSageGATHybridClassifier(nn.Module):
    """Hybrid GraphSAGE and GAT classifier used as the main GNN model."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        if torch is None or functional is None or GATConv is None or SAGEConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate the hybrid GNN.")
        super().__init__()
        self.sage = SAGEConv(in_channels, hidden_channels)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=dropout)
        self.norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, data) -> torch.Tensor:
        """Compute node logits for a PyG graph batch."""

        x = self.sage(data.x, data.edge_index)
        x = functional.relu(x)
        x = self.norm(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        x = self.gat(x, data.edge_index)
        x = functional.elu(x)
        x = functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
