"""GBD-CTU GAT classifier.

Architecture:
    GATConv(in  →  hidden_channels, heads=heads, concat=True)  →  ELU  →  Dropout
    GATConv(hidden_channels*heads → embed_channels, heads=1, concat=False)  →  ELU
    Linear(embed_channels → out_channels)

With defaults hidden_channels=64, heads=4, embed_channels=32 the shapes are:
    in → 256 → 32 → 2
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.nn import GATConv
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    GATConv = None  # type: ignore[assignment]


class GATNodeClassifier(nn.Module):
    """Two-layer GAT node classifier for CTU-13 IP graphs.

    Parameters
    ----------
    in_channels:
        Dimensionality of input node features.
    hidden_channels:
        Per-head output width of the first GAT layer (total output =
        ``hidden_channels * heads`` when ``concat=True``).
    embed_channels:
        Output width of the second GAT layer (single-head, no concat).
    out_channels:
        Number of output classes.
    heads:
        Number of attention heads in the first layer.
    dropout:
        Dropout probability applied between the two GAT layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        embed_channels: int = 32,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        if torch is None or functional is None or GATConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate GATNodeClassifier.")
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, embed_channels, heads=1, concat=False, dropout=dropout)
        self.dropout_prob = dropout
        self.classifier = nn.Linear(embed_channels, out_channels)

        self.num_parameters: int = sum(p.numel() for p in self.parameters())
        _logger.info(
            "GATNodeClassifier | hidden=%d  heads=%d  embed=%d  dropout=%.2f  total_params=%d",
            hidden_channels, heads, embed_channels, dropout, self.num_parameters,
        )

    def forward(self, data) -> "torch.Tensor":
        """Compute per-node logits.

        Parameters
        ----------
        data:
            PyG ``Data`` object with ``x`` and ``edge_index``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, out_channels]``.
        """
        x = functional.elu(self.gat1(data.x, data.edge_index))
        x = functional.dropout(x, p=self.dropout_prob, training=self.training)
        x = functional.elu(self.gat2(x, data.edge_index))
        return self.classifier(x)

