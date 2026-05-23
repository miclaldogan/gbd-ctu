"""GBD-CTU GNN forward tests.

These tests validate that the hybrid GNN produces correctly shaped logits for a
small synthetic PyG graph.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from gbd_ctu.models.gnn.gat import GATNodeClassifier
from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier
from gbd_ctu.models.gnn import build_gnn_from_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_graph(num_nodes: int = 5, in_channels: int = 6):
    """Return a minimal synthetic PyG Data object."""
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]],
        dtype=torch.long,
    )
    return pyg_data.Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# GraphSAGENodeClassifier
# ---------------------------------------------------------------------------

def test_graphsage_forward_output_shape() -> None:
    """Forward pass must return logits with shape [num_nodes, out_channels]."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    logits = model(data)
    assert logits.shape == (5, 2), f"Expected (5, 2), got {logits.shape}"


def test_graphsage_output_shape_with_6_node_features() -> None:
    """Output shape must equal [N, 2] on a graph built from 6-dim node features
    (matching the CTU-13 IPGraphBuilder schema)."""
    data = _small_graph(num_nodes=10, in_channels=6)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    logits = model(data)
    assert logits.shape == (10, 2)


def test_graphsage_output_is_finite() -> None:
    """Forward pass must not produce NaN or Inf logits."""
    import math
    data = _small_graph(num_nodes=8, in_channels=6)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=64, out_channels=2, num_layers=2)
    model.eval()
    with torch.no_grad():
        logits = model(data)
    assert torch.isfinite(logits).all(), "GraphSAGE produced non-finite logits"


def test_graphsage_param_count_logged_on_init() -> None:
    """num_parameters attribute must be set on init and be a positive integer."""
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    assert hasattr(model, "num_parameters"), "num_parameters attribute missing"
    assert isinstance(model.num_parameters, int)
    assert model.num_parameters > 0


def test_graphsage_param_count_matches_actual() -> None:
    """Stored num_parameters must equal the actual sum of parameter elements."""
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    actual = sum(p.numel() for p in model.parameters())
    assert model.num_parameters == actual


def test_graphsage_configurable_hidden_channels() -> None:
    """hidden_channels parameter must control first-layer width."""
    model_large = GraphSAGENodeClassifier(in_channels=6, hidden_channels=256, out_channels=2)
    model_small = GraphSAGENodeClassifier(in_channels=6, hidden_channels=32, out_channels=2)
    assert model_large.num_parameters > model_small.num_parameters


def test_graphsage_configurable_num_layers() -> None:
    """num_layers must control the number of SAGEConv layers."""
    m2 = GraphSAGENodeClassifier(in_channels=6, hidden_channels=64, out_channels=2, num_layers=2)
    m4 = GraphSAGENodeClassifier(in_channels=6, hidden_channels=64, out_channels=2, num_layers=4)
    assert len(m2.convolutions) == 2
    assert len(m4.convolutions) == 4


def test_graphsage_configurable_dropout_has_no_effect_in_eval() -> None:
    """Dropout must be disabled in eval mode so two forward passes give identical output."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=64, out_channels=2, dropout=0.9)
    model.eval()
    with torch.no_grad():
        out1 = model(data)
        out2 = model(data)
    assert torch.allclose(out1, out2), "Eval-mode outputs differ — dropout not disabled"


def test_graphsage_train_eval_mode_differ() -> None:
    """Training mode with high dropout must differ from eval mode (statistically)."""
    torch.manual_seed(0)
    data = _small_graph(num_nodes=20, in_channels=6)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, dropout=0.9)
    model.train()
    out_train = model(data)
    model.eval()
    with torch.no_grad():
        out_eval = model(data)
    # With 90 % dropout it is astronomically unlikely all values are equal
    assert not torch.allclose(out_train, out_eval, atol=1e-4)


def test_graphsage_single_layer() -> None:
    """A single-layer model (num_layers=1) must still produce valid output."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=64, out_channels=2, num_layers=1)
    logits = model(data)
    assert logits.shape == (5, 2)


def test_graphsage_per_layer_batch_norms() -> None:
    """There must be one BatchNorm1d per conv layer."""
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    assert len(model.batch_norms) == len(model.convolutions) == 3


def test_graphsage_channel_halving() -> None:
    """Conv output widths must halve layer-by-layer (128, 64, 32 for hidden=128)."""
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    expected_out_dims = [128, 64, 32]
    for i, bn in enumerate(model.batch_norms):
        assert bn.num_features == expected_out_dims[i], (
            f"Layer {i} BatchNorm width: expected {expected_out_dims[i]}, got {bn.num_features}"
        )


# ---------------------------------------------------------------------------
# Hybrid (updated parallel architecture)
# ---------------------------------------------------------------------------

def test_hybrid_forward_output_shape() -> None:
    """Parallel SAGE+GAT hybrid must emit [N, 2] logits."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32, out_channels=2)
    logits = model(data)
    assert logits.shape == (5, 2)


def test_hybrid_backward_compat_hidden_channels_kwarg() -> None:
    """Callers using the legacy hidden_channels kwarg must still work."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSageGATHybridClassifier(in_channels=6, hidden_channels=16, out_channels=2)
    logits = model(data)
    assert logits.shape == (5, 2)


def test_hybrid_output_is_finite() -> None:
    """Hybrid forward pass must not produce NaN or Inf."""
    import numpy as np
    data = _small_graph(num_nodes=8, in_channels=6)
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=16, out_channels=2)
    model.eval()
    with torch.no_grad():
        logits = model(data)
    assert torch.isfinite(logits).all()


def test_hybrid_both_branches_have_parameters() -> None:
    """sage_branch and gat_branch must each have trainable parameters."""
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32, out_channels=2)
    sage_params = sum(p.numel() for p in model.sage_branch.parameters())
    gat_params = sum(p.numel() for p in model.gat_branch.parameters())
    assert sage_params > 0
    assert gat_params > 0


def test_hybrid_classifier_input_is_concatenated() -> None:
    """The final Linear layer input must be 2*embed_channels."""
    embed = 32
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=embed, out_channels=2)
    assert model.classifier.in_features == 2 * embed


def test_hybrid_param_count_stored_on_init() -> None:
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32, out_channels=2)
    assert hasattr(model, "num_parameters")
    assert model.num_parameters == sum(p.numel() for p in model.parameters())


def test_hybrid_dropout_disabled_in_eval() -> None:
    """Eval mode must produce deterministic outputs."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32, dropout=0.9)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(data), model(data))


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------

def test_gat_forward_output_shape() -> None:
    """GAT forward pass must return [N, 2] logits."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, embed_channels=32, out_channels=2, heads=4)
    logits = model(data)
    assert logits.shape == (5, 2)


def test_gat_output_shape_with_6_node_features() -> None:
    """GAT must handle 6-dim node features (CTU-13 schema)."""
    data = _small_graph(num_nodes=10, in_channels=6)
    model = GATNodeClassifier(in_channels=6)
    logits = model(data)
    assert logits.shape == (10, 2)


def test_gat_output_is_finite() -> None:
    """GAT forward pass must not produce NaN or Inf."""
    data = _small_graph(num_nodes=8, in_channels=6)
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, embed_channels=32)
    model.eval()
    with torch.no_grad():
        logits = model(data)
    assert torch.isfinite(logits).all()


def test_gat_first_layer_concat_true() -> None:
    """gat1 must use concat=True so its output width = hidden_channels * heads."""
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, heads=4)
    assert model.gat1.concat is True


def test_gat_second_layer_single_head_no_concat() -> None:
    """gat2 must use heads=1 and concat=False."""
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, embed_channels=32, heads=4)
    assert model.gat2.heads == 1
    assert model.gat2.concat is False


def test_gat_classifier_input_matches_embed_channels() -> None:
    """The Linear classifier input must equal embed_channels."""
    embed = 32
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, embed_channels=embed, heads=4)
    assert model.classifier.in_features == embed


def test_gat_param_count_stored_on_init() -> None:
    model = GATNodeClassifier(in_channels=6)
    assert hasattr(model, "num_parameters")
    assert model.num_parameters == sum(p.numel() for p in model.parameters())


def test_gat_dropout_disabled_in_eval() -> None:
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GATNodeClassifier(in_channels=6, dropout=0.9)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(data), model(data))


def test_gat_configurable_heads() -> None:
    """More heads → more parameters in gat1."""
    m2 = GATNodeClassifier(in_channels=6, hidden_channels=32, heads=2)
    m8 = GATNodeClassifier(in_channels=6, hidden_channels=32, heads=8)
    assert m8.num_parameters > m2.num_parameters


# ---------------------------------------------------------------------------
# Factory: build_gnn_from_config
# ---------------------------------------------------------------------------

def test_factory_builds_graphsage() -> None:
    config = {"model_type": "graphsage", "graphsage": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3}}
    model = build_gnn_from_config(in_channels=6, config=config)
    assert isinstance(model, GraphSAGENodeClassifier)
    data = _small_graph(num_nodes=5, in_channels=6)
    assert model(data).shape == (5, 2)


def test_factory_builds_gat() -> None:
    config = {"model_type": "gat", "gat": {"hidden_channels": 32, "embed_channels": 16, "heads": 2}}
    model = build_gnn_from_config(in_channels=6, config=config)
    assert isinstance(model, GATNodeClassifier)
    data = _small_graph(num_nodes=5, in_channels=6)
    assert model(data).shape == (5, 2)


def test_factory_builds_hybrid() -> None:
    config = {"model_type": "hybrid", "hybrid": {"embed_channels": 16, "dropout": 0.2}}
    model = build_gnn_from_config(in_channels=6, config=config)
    assert isinstance(model, GraphSageGATHybridClassifier)
    data = _small_graph(num_nodes=5, in_channels=6)
    assert model(data).shape == (5, 2)


def test_factory_defaults_to_hybrid_when_no_model_type() -> None:
    config = {}
    model = build_gnn_from_config(in_channels=6, config=config)
    assert isinstance(model, GraphSageGATHybridClassifier)


def test_factory_raises_on_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_gnn_from_config(in_channels=6, config={"model_type": "transformer"})

