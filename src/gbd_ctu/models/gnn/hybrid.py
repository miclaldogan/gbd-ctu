"""GBD-CTU GraphSAGE + GAT hybrid classifier.

Architecture — parallel branches sharing the same input graph:

    SAGE branch:  SAGEConv(in → sage_hidden=128) → BN(128) → ReLU → Dropout
                  SAGEConv(128 → embed_channels=32) → BN(32) → ReLU

    GAT  branch:  GATConv(in → gat_hidden=64, heads=4, concat=True) → ELU → Dropout
                                                                     # out: 256
                  GATConv(256 → embed_channels=32, heads=1, concat=False) → ELU

    Fusion:       concat([sage_32, gat_32]) = 64 → Linear(64 → out_channels)

With default embed_channels=32 the concatenated representation is 64-dim.
Both branches receive the same data.x and data.edge_index; gradients flow
through both.
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
    nn = type("nn", (), {"Module": object})()  # type: ignore[assignment]
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]


class _SageBranch(nn.Module):
    """Two-layer GraphSAGE sub-network used inside the hybrid classifier.

    Architecture: SAGEConv(in → hidden) → BN → ReLU → Dropout
                  SAGEConv(hidden → embed) → BN → ReLU
    """

    def __init__(self, in_channels: int, hidden: int, embed: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = SAGEConv(hidden, embed)
        self.bn2 = nn.BatchNorm1d(embed)
        self.dropout_prob = dropout

    def forward(self, x, edge_index):  # type: ignore[override]
        x = functional.relu(self.bn1(self.conv1(x, edge_index)))
        x = functional.dropout(x, p=self.dropout_prob, training=self.training)
        return functional.relu(self.bn2(self.conv2(x, edge_index)))


class _GATBranch(nn.Module):
    """Two-layer GAT sub-network used inside the hybrid classifier.

    Architecture: GATConv(in → hidden, heads=heads, concat=True) → ELU → Dropout
                  GATConv(hidden*heads → embed, heads=1, concat=False) → ELU
    """

    def __init__(
        self, in_channels: int, hidden: int, embed: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, embed, heads=1, concat=False, dropout=dropout)
        self.dropout_prob = dropout

    def forward(self, x, edge_index, return_attention: bool = False):  # type: ignore[override]
        if return_attention:
            E = edge_index.shape[1]
            out1, (_, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
            out1 = functional.elu(out1)
            out1 = functional.dropout(out1, p=self.dropout_prob, training=self.training)
            out2, (_, alpha2) = self.conv2(out1, edge_index, return_attention_weights=True)
            out2 = functional.elu(out2)
            return out2, {"layer_0": alpha1[:E], "layer_1": alpha2[:E]}
        x = functional.elu(self.conv1(x, edge_index))
        x = functional.dropout(x, p=self.dropout_prob, training=self.training)
        return functional.elu(self.conv2(x, edge_index))


class GraphSageGATHybridClassifier(nn.Module):
    """Parallel GraphSAGE + GAT hybrid classifier for CTU-13 botnet detection.

    Parameters
    ----------
    in_channels:
        Dimensionality of input node features.
    embed_channels:
        Output width produced by each branch (both SAGE and GAT branches
        produce this many dimensions).  The concatenated vector fed to the
        classifier is ``2 * embed_channels``.
    out_channels:
        Number of output classes (default 2).
    sage_hidden:
        Hidden width of the SAGE branch first layer (default 128).
    gat_hidden:
        Per-head output width of the GAT branch first layer (default 64).
        With ``gat_heads=4`` the total first-layer output is 256.
    gat_heads:
        Number of attention heads in the GAT branch first layer (default 4).
    dropout:
        Dropout probability applied after each branch's first layer.
    hidden_channels:
        Deprecated alias for ``embed_channels`` (kept for back-compat).
    """

    def __init__(
        self,
        in_channels: int,
        embed_channels: int = 32,
        out_channels: int = 2,
        sage_hidden: int = 128,
        gat_hidden: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.3,
        # back-compat: older call sites pass hidden_channels
        hidden_channels: int | None = None,
    ) -> None:
        if torch is None or functional is None or GATConv is None or SAGEConv is None:
            raise ImportError("torch and torch-geometric are required to instantiate the hybrid GNN.")
        super().__init__()
        if hidden_channels is not None:
            embed_channels = hidden_channels  # honour legacy kwarg

        self.sage_branch = _SageBranch(in_channels, sage_hidden, embed_channels, dropout)
        self.gat_branch = _GATBranch(in_channels, gat_hidden, embed_channels, gat_heads, dropout)
        self.dropout_prob = dropout
        self.classifier = nn.Linear(2 * embed_channels, out_channels)

        self.num_parameters: int = sum(p.numel() for p in self.parameters())
        _logger.info(
            "GraphSageGATHybridClassifier | sage_hidden=%d  gat_hidden=%d  "
            "gat_heads=%d  embed=%d  dropout=%.2f  total_params=%d",
            sage_hidden, gat_hidden, gat_heads, embed_channels, dropout, self.num_parameters,
        )

    def forward(self, data, return_attention: bool = False):
        """Compute per-node logits via parallel SAGE and GAT branches.

        Parameters
        ----------
        data:
            PyG ``Data`` object with ``x`` and ``edge_index``.
        return_attention:
            When ``True`` return ``(logits, attention_dict)`` where
            ``attention_dict`` maps ``"layer_0"`` / ``"layer_1"`` to tensors
            of shape ``[num_edges, num_heads]`` from the GAT branch.

        Returns
        -------
        torch.Tensor or tuple
            Logits of shape ``[num_nodes, out_channels]``, or
            ``(logits, {"layer_0": alpha_0, "layer_1": alpha_1})`` when
            ``return_attention=True``.
        """
        x, edge_index = data.x, data.edge_index
        sage_embed = self.sage_branch(x, edge_index)
        if return_attention:
            gat_embed, attn = self.gat_branch(x, edge_index, return_attention=True)
            logits = self.classifier(torch.cat([sage_embed, gat_embed], dim=-1))
            return logits, attn
        gat_embed = self.gat_branch(x, edge_index)
        return self.classifier(torch.cat([sage_embed, gat_embed], dim=-1))

