"""GBD-CTU GraphSAGE + GAT hybrid classifier.

Architecture — parallel branches sharing the same input graph:
    SAGE branch:  SAGEConv(in → embed_channels) → ReLU
    GAT  branch:  GATConv(in → embed_channels, heads=1, concat=False) → ELU
    Concat: [sage_embed || gat_embed]  →  Linear(2*embed_channels → out_channels)

With default embed_channels=32 the concatenated representation is 64-dim.
Both branches are trained jointly.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]


class GraphSageGATHybridClassifier(nn.Module):
    """Parallel GraphSAGE + GAT hybrid classifier for CTU-13 botnet detection.

    Parameters
    ----------
    in_channels:
        Dimensionality of input node features.
    embed_channels:
        Output width of each branch (SAGE and GAT each produce this width).
        The concatenated representation is ``2 * embed_channels``.
    out_channels:
        Number of output classes (default 2).
    heads:
        Attention heads used in the GAT branch.  Kept for API compatibility;
        the GAT branch always uses ``heads=1, concat=False`` so its output
        is always ``embed_channels``.
    dropout:
        Dropout probability applied after each branch embedding.
    hidden_channels:
        Deprecated alias for ``embed_channels`` (kept for back-compat).
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        out_channels: int = 2,
        heads: int = 1,
        dropout: float = 0.3,
        # back-compat: older call sites pass hidden_channels
        hidden_channels: int | None = None,
    ) -> None:
        if torch is None or functional is None or GATConv is None or SAGEConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate the hybrid GNN.")
        super().__init__()
        if hidden_channels is not None:
            embed_channels = hidden_channels  # honour legacy kwarg

        self.sage_branch = SAGEConv(in_channels, embed_channels)
        self.gat_branch = GATConv(in_channels, embed_channels, heads=1, concat=False, dropout=dropout)
        self.dropout_prob = dropout
        self.classifier = nn.Linear(2 * embed_channels, out_channels)

        self.num_parameters: int = sum(p.numel() for p in self.parameters())
        _logger.info(
            "GraphSageGATHybridClassifier | embed=%d  dropout=%.2f  total_params=%d",
            embed_channels, dropout, self.num_parameters,
        )

    def forward(self, data) -> "torch.Tensor":
        """Compute per-node logits via parallel SAGE and GAT branches.

        Parameters
        ----------
        data:
            PyG ``Data`` object with ``x`` and ``edge_index``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_nodes, out_channels]``.
        """
        x, edge_index = data.x, data.edge_index

        sage_embed = functional.relu(self.sage_branch(x, edge_index))
        sage_embed = functional.dropout(sage_embed, p=self.dropout_prob, training=self.training)

        gat_embed = functional.elu(self.gat_branch(x, edge_index))
        gat_embed = functional.dropout(gat_embed, p=self.dropout_prob, training=self.training)

        combined = torch.cat([sage_embed, gat_embed], dim=-1)
        return self.classifier(combined)

