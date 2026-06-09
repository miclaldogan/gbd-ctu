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
from gbd_ctu.models.gnn import build_gnn_from_config, build_gnn


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
    """The final Linear layer input must be 3*embed_channels (sage+gat+skip)."""
    embed = 32
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=embed, out_channels=2)
    assert model.classifier.in_features == 3 * embed


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


# ---------------------------------------------------------------------------
# Issue #10 — architecture spec compliance
# ---------------------------------------------------------------------------

def test_graphsage_param_count_in_expected_range() -> None:
    """GraphSAGENodeClassifier(6, 128, 2) param count must be in a reasonable range
    for the pyramid encoder spec (in→128→64→32→2).  Computed value is ~22 K."""
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=128, out_channels=2, num_layers=3)
    assert 15_000 <= model.num_parameters <= 50_000, (
        f"Unexpected param count {model.num_parameters}; check pyramid architecture."
    )


def test_gat_first_layer_output_dim_256() -> None:
    """GATConv(in→64, heads=4, concat=True) must produce 256-dim node embeddings."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, embed_channels=32, heads=4)
    captured: list = []

    def _hook(module, inp, out):
        captured.append(out)

    handle = model.gat1.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        model(data)
    handle.remove()

    assert len(captured) == 1
    assert captured[0].shape == (5, 256), (
        f"Expected gat1 output (5, 256), got {captured[0].shape}"
    )


def test_hybrid_both_branches_activate_via_hook() -> None:
    """Both sage_branch and gat_branch must execute during a forward pass."""
    data = _small_graph(num_nodes=5, in_channels=6)
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32, out_channels=2)
    sage_outputs: list = []
    gat_outputs: list = []

    h1 = model.sage_branch.register_forward_hook(lambda m, i, o: sage_outputs.append(o))
    h2 = model.gat_branch.register_forward_hook(lambda m, i, o: gat_outputs.append(o))
    model.eval()
    with torch.no_grad():
        logits = model(data)
    h1.remove()
    h2.remove()

    assert len(sage_outputs) == 1, "sage_branch did not execute"
    assert len(gat_outputs) == 1, "gat_branch did not execute"
    assert sage_outputs[0].shape == (5, 32)
    assert gat_outputs[0].shape == (5, 32)
    assert logits.shape == (5, 2)


def test_hybrid_sage_branch_has_two_conv_layers() -> None:
    """SAGE branch must have three parallel mean+max layer pairs (conv1_mean..conv3_max)."""
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32)
    assert hasattr(model.sage_branch, "conv1_mean"), "sage_branch missing conv1_mean"
    assert hasattr(model.sage_branch, "conv1_max"),  "sage_branch missing conv1_max"
    assert hasattr(model.sage_branch, "conv3_mean"), "sage_branch missing conv3_mean"
    # BN dimensions: layer1=sage_hidden(64), layer2=sage_hidden(64), layer3=embed(32)
    assert model.sage_branch.bn1.num_features == 64
    assert model.sage_branch.bn2.num_features == 64
    assert model.sage_branch.bn3.num_features == 32


def test_hybrid_gat_branch_first_layer_uses_four_heads() -> None:
    """GAT branch conv1 must use heads=4 and concat=True (→ 256-dim output)."""
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32)
    assert model.gat_branch.conv1.heads == 4
    assert model.gat_branch.conv1.concat is True


def test_hybrid_gat_branch_second_layer_single_head() -> None:
    """GAT branch conv2 must use heads=1 and concat=False (→ embed_channels output)."""
    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32)
    assert model.gat_branch.conv2.heads == 1
    assert model.gat_branch.conv2.concat is False


# ---------------------------------------------------------------------------
# Issue #11 — build_gnn factory + gradient-flow acceptance criteria
# ---------------------------------------------------------------------------

def test_build_gnn_factory_builds_hybrid() -> None:
    """build_gnn must instantiate a hybrid model when model_type=hybrid."""
    config = {"in_channels": 6, "model_type": "hybrid", "hybrid": {"embed_channels": 16}}
    model = build_gnn(config)
    assert isinstance(model, GraphSageGATHybridClassifier)
    data = _small_graph(num_nodes=5, in_channels=6)
    assert model(data).shape == (5, 2)


def test_build_gnn_factory_builds_graphsage() -> None:
    """build_gnn must instantiate GraphSAGENodeClassifier when model_type=graphsage."""
    config = {"in_channels": 6, "model_type": "graphsage"}
    model = build_gnn(config)
    assert isinstance(model, GraphSAGENodeClassifier)
    data = _small_graph(num_nodes=5, in_channels=6)
    assert model(data).shape == (5, 2)


def test_build_gnn_factory_builds_gat() -> None:
    """build_gnn must instantiate GATNodeClassifier when model_type=gat."""
    config = {"in_channels": 6, "model_type": "gat"}
    model = build_gnn(config)
    assert isinstance(model, GATNodeClassifier)
    data = _small_graph(num_nodes=5, in_channels=6)
    assert model(data).shape == (5, 2)


def test_build_gnn_raises_when_in_channels_missing() -> None:
    """build_gnn must raise ValueError when in_channels is absent."""
    with pytest.raises(ValueError, match="in_channels"):
        build_gnn({"model_type": "hybrid"})


def test_build_gnn_exported_in_package() -> None:
    """build_gnn must be importable directly from gbd_ctu.models.gnn."""
    from gbd_ctu.models.gnn import build_gnn as _bg
    assert callable(_bg)


def test_hybrid_gradients_flow_through_both_branches() -> None:
    """Both SAGE and GAT branches must contribute non-zero gradients to data.x."""
    torch.manual_seed(42)
    data = _small_graph(num_nodes=10, in_channels=6)
    data.x = data.x.requires_grad_(True)

    model = GraphSageGATHybridClassifier(in_channels=6, embed_channels=32, out_channels=2)
    model.train()
    logits = model(data)
    # Use sum of all logits as scalar loss proxy
    logits.sum().backward()

    assert data.x.grad is not None, "No gradient on data.x after backward"
    assert data.x.grad.shape == data.x.shape
    assert not torch.all(data.x.grad == 0), "Gradient on data.x is all-zero"


def test_graphsage_gradients_flow_to_input() -> None:
    """GraphSAGE must propagate gradients back to data.x."""
    torch.manual_seed(0)
    data = _small_graph(num_nodes=8, in_channels=6)
    data.x = data.x.requires_grad_(True)
    model = GraphSAGENodeClassifier(in_channels=6, hidden_channels=64, out_channels=2, num_layers=2)
    model.train()
    model(data).sum().backward()
    assert data.x.grad is not None
    assert not torch.all(data.x.grad == 0)


def test_gat_gradients_flow_to_input() -> None:
    """GAT must propagate gradients back to data.x."""
    torch.manual_seed(0)
    data = _small_graph(num_nodes=8, in_channels=6)
    data.x = data.x.requires_grad_(True)
    model = GATNodeClassifier(in_channels=6, hidden_channels=64, embed_channels=32, heads=4)
    model.train()
    model(data).sum().backward()
    assert data.x.grad is not None
    assert not torch.all(data.x.grad == 0)

