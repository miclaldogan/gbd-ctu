"""GBD-CTU GraphSAGE + GAT hybrid classifier.

Architecture — parallel branches sharing the same input graph:

    SAGE branch:  3 layers, each with parallel mean+max SAGEConv aggregation.
                  Residual connections at every layer prevent over-smoothing
                  and preserve discriminative raw-feature signal.

                  Layer 1: [SAGEConv_mean(in → h/2) ‖ SAGEConv_max(in → h/2)]
                           = hidden → BN → ReLU + Linear(in → hidden) [residual]
                  Layer 2: [SAGEConv_mean(h → h/2) ‖ SAGEConv_max(h → h/2)]
                           = hidden → BN → ReLU + identity [residual]
                  Layer 3: [SAGEConv_mean(h → e/2) ‖ SAGEConv_max(h → e/2)]
                           = embed → BN → ReLU + Linear(h → embed) [residual]

    GAT  branch:  GATConv(in → gat_hidden=64, heads=4, concat=True) → ELU → Dropout
                  + Linear(in → hidden*heads) [residual]
                  GATConv(hidden*heads → embed_channels=32, heads=1) → ELU

    Skip:         Linear(in → embed) — raw features always reach the classifier.

    Fusion:       concat([sage_embed, gat_embed, skip]) = 3*embed → Linear → out

The mean aggregator captures average neighborhood behavior; max preserves
extreme botnet values (high byte counts, unusual ports) that mean dilutes.
3-hop neighborhoods allow the GNN to reach C&C peers of a bot's direct
connections, revealing the botnet cluster structure.
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
    """Three-layer GraphSAGE with parallel mean+max aggregation and residuals.

    Each layer runs two SAGEConv ops (mean and max) in parallel and
    concatenates their outputs.  Mean captures average neighborhood behavior;
    max preserves extreme botnet values (high bytes, unusual ports) that mean
    dilutes.  Residual connections at every layer prevent over-smoothing so the
    3rd hop doesn't wash out the discriminative raw-feature signal.
    """

    def __init__(self, in_channels: int, hidden: int, embed: int, dropout: float) -> None:
        super().__init__()
        half_h = hidden // 2
        half_e = embed // 2
        # Layer 1: in → hidden (mean + max, each produces half_h / rest)
        self.conv1_mean = SAGEConv(in_channels, half_h, aggr="mean")
        self.conv1_max  = SAGEConv(in_channels, hidden - half_h, aggr="max")
        self.bn1  = nn.BatchNorm1d(hidden)
        self.res1 = nn.Linear(in_channels, hidden, bias=False)
        # Layer 2: hidden → hidden (identity residual — same dim)
        self.conv2_mean = SAGEConv(hidden, half_h, aggr="mean")
        self.conv2_max  = SAGEConv(hidden, hidden - half_h, aggr="max")
        self.bn2  = nn.BatchNorm1d(hidden)
        # Layer 3: hidden → embed
        self.conv3_mean = SAGEConv(hidden, half_e, aggr="mean")
        self.conv3_max  = SAGEConv(hidden, embed - half_e, aggr="max")
        self.bn3  = nn.BatchNorm1d(embed)
        self.res3 = nn.Linear(hidden, embed, bias=False)
        self.dropout_prob = dropout

    def forward(self, x, edge_index):  # type: ignore[override]
        # Layer 1 — BN applied to (conv_out + residual) to prevent un-normalised
        # skip path from dominating the sum and causing activation explosion.
        h = torch.cat([self.conv1_mean(x, edge_index),
                       self.conv1_max(x, edge_index)], dim=-1)
        h = functional.relu(self.bn1(h + self.res1(x)))
        h = functional.dropout(h, p=self.dropout_prob, training=self.training)
        # Layer 2 — identity residual (same dim), BN on sum
        h2 = torch.cat([self.conv2_mean(h, edge_index),
                        self.conv2_max(h, edge_index)], dim=-1)
        h2 = functional.relu(self.bn2(h2 + h))
        h2 = functional.dropout(h2, p=self.dropout_prob, training=self.training)
        # Layer 3 — project residual then BN on sum
        out = torch.cat([self.conv3_mean(h2, edge_index),
                         self.conv3_max(h2, edge_index)], dim=-1)
        return functional.relu(self.bn3(out + self.res3(h2)))


class _GATBranch(nn.Module):
    """Two-layer GAT sub-network with residual connection on the first layer.

    Architecture: GATConv(in → hidden, heads=heads, concat=True) → ELU → Dropout
                  + Linear(in → hidden*heads) [residual]
                  GATConv(hidden*heads → embed, heads=1, concat=False) → ELU
    """

    def __init__(
        self, in_channels: int, hidden: int, embed: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, embed, heads=1, concat=False, dropout=dropout)
        self.res1  = nn.Linear(in_channels, hidden * heads, bias=False)
        self.dropout_prob = dropout

    def forward(self, x, edge_index, return_attention: bool = False):  # type: ignore[override]
        if return_attention:
            E = edge_index.shape[1]
            out1, (_, alpha1) = self.conv1(x, edge_index, return_attention_weights=True)
            out1 = functional.elu(out1 + self.res1(x))
            out1 = functional.dropout(out1, p=self.dropout_prob, training=self.training)
            out2, (_, alpha2) = self.conv2(out1, edge_index, return_attention_weights=True)
            out2 = functional.elu(out2)
            return out2, {"layer_0": alpha1[:E], "layer_1": alpha2[:E]}
        h = functional.elu(self.conv1(x, edge_index) + self.res1(x))
        h = functional.dropout(h, p=self.dropout_prob, training=self.training)
        return functional.elu(self.conv2(h, edge_index))


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
        sage_hidden: int = 64,
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

        # sage_hidden is halved from 128→64: the 3-layer × 2-aggregator (mean+max)
        # SAGE branch has 6 forward passes per graph and 100K nodes exhaust GPU VRAM
        # at sage_hidden=128. With sage_hidden=64 each parallel conv produces 32-dim
        # embeddings (total=64=same as before) while adding depth and mean aggregation.
        # Input BN normalises raw node features before any branch sees them.
        # RobustScaler preserves outliers (good for tree models) but they
        # produce loss spikes through the skip linear at init; input BN fixes this.
        self.input_bn = nn.BatchNorm1d(in_channels)
        self.sage_branch = _SageBranch(in_channels, sage_hidden, embed_channels, dropout)
        self.gat_branch = _GATBranch(in_channels, gat_hidden, embed_channels, gat_heads, dropout)
        self.dropout_prob = dropout
        self.skip = nn.Linear(in_channels, embed_channels)
        self.classifier = nn.Linear(3 * embed_channels, out_channels)

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
        x_raw, edge_index = data.x, data.edge_index
        x = self.input_bn(x_raw)
        skip = functional.relu(self.skip(x))
        sage_embed = self.sage_branch(x, edge_index)
        if return_attention:
            gat_embed, attn = self.gat_branch(x, edge_index, return_attention=True)
            logits = self.classifier(torch.cat([sage_embed, gat_embed, skip], dim=-1))
            return logits, attn
        gat_embed = self.gat_branch(x, edge_index)
        return self.classifier(torch.cat([sage_embed, gat_embed, skip], dim=-1))

    def get_embeddings(self, data) -> "torch.Tensor":
        """Return the pre-classifier embedding for each node (shape [N, 3*embed]).

        Used by the GNN+XGBoost ensemble to build an enriched feature matrix.
        """
        x_raw, edge_index = data.x, data.edge_index
        with torch.no_grad():
            x = self.input_bn(x_raw)
            skip = functional.relu(self.skip(x))
            sage_embed = self.sage_branch(x, edge_index)
            gat_embed = self.gat_branch(x, edge_index)
        return torch.cat([sage_embed, gat_embed, skip], dim=-1)

