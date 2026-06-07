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
    nn = type("nn", (), {"Module": object})()  # type: ignore[assignment]
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

    def forward(self, data, return_attention: bool = False):
        """Compute per-node logits.

        Parameters
        ----------
        data:
            PyG ``Data`` object with ``x`` and ``edge_index``.
        return_attention:
            When ``True`` return ``(logits, attention_dict)`` where
            ``attention_dict`` maps ``"layer_0"`` / ``"layer_1"`` to tensors
            of shape ``[num_edges, num_heads]``.

        Returns
        -------
        torch.Tensor or tuple
            Logits of shape ``[num_nodes, out_channels]``, or
            ``(logits, {"layer_0": alpha_0, "layer_1": alpha_1})`` when
            ``return_attention=True``.
        """
        if return_attention:
            E = data.edge_index.shape[1]
            out1, (_, alpha1) = self.gat1(data.x, data.edge_index, return_attention_weights=True)
            out1 = functional.elu(out1)
            out1 = functional.dropout(out1, p=self.dropout_prob, training=self.training)
            out2, (_, alpha2) = self.gat2(out1, data.edge_index, return_attention_weights=True)
            out2 = functional.elu(out2)
            logits = self.classifier(out2)
            return logits, {"layer_0": alpha1[:E], "layer_1": alpha2[:E]}

        x = functional.elu(self.gat1(data.x, data.edge_index))
        x = functional.dropout(x, p=self.dropout_prob, training=self.training)
        x = functional.elu(self.gat2(x, data.edge_index))
        return self.classifier(x)

