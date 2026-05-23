"""GBD-CTU GNN forward tests.

These tests validate that the hybrid GNN produces correctly shaped logits for a
small synthetic PyG graph.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier


def test_hybrid_forward_returns_node_logits() -> None:
    """The hybrid model should emit one logit vector per node."""

    data = pyg_data.Data(
        x=torch.randn(4, 14),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
    )
    model = GraphSageGATHybridClassifier(in_channels=14, hidden_channels=16, out_channels=2, heads=2)
    logits = model(data)
    assert logits.shape == (4, 2)
