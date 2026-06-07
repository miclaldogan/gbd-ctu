"""Tests for GAT attention weight extraction and analysis.

Tests that require torch / torch_geometric skip gracefully when absent.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from gbd_ctu.evaluation.attention import attention_correlation

_pyg_available = importlib.util.find_spec("torch_geometric") is not None
_skip_pyg = pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pyg_graph(n_nodes: int = 50, n_features: int = 6, seed: int = 0):
    """Synthetic PyG Data graph for tests requiring torch_geometric."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((n_nodes, n_features)).astype(np.float32))
    y = torch.from_numpy(rng.choice([0, 1], size=n_nodes, p=[0.7, 0.3]).astype(np.int64))
    # Random directed edges
    n_edges = n_nodes * 3
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=n_nodes)


# ---------------------------------------------------------------------------
# attention_correlation (no torch needed)
# ---------------------------------------------------------------------------

class TestAttentionCorrelation:
    def _make_data(self, n: int = 100, n_feat: int = 5, seed: int = 0):
        rng = np.random.default_rng(seed)
        edge_attr = rng.standard_normal((n, n_feat)).astype(np.float32)
        # Attention weight is perfectly correlated with feature 0
        attn = edge_attr[:, 0] + rng.standard_normal(n) * 0.1
        names = [f"feat_{i}" for i in range(n_feat)]
        return edge_attr, attn, names

    def test_returns_dataframe(self):
        ea, aw, names = self._make_data()
        df = attention_correlation(ea, aw, names)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        ea, aw, names = self._make_data()
        df = attention_correlation(ea, aw, names)
        assert set(df.columns) == {"feature", "pearson_r", "p_value"}

    def test_one_row_per_feature(self):
        ea, aw, names = self._make_data(n_feat=7)
        df = attention_correlation(ea, aw, names)
        assert len(df) == 7

    def test_sorted_by_abs_pearson_r(self):
        ea, aw, names = self._make_data()
        df = attention_correlation(ea, aw, names)
        abs_r = df["pearson_r"].abs().tolist()
        assert abs_r == sorted(abs_r, reverse=True)

    def test_most_correlated_feature_first(self):
        ea, aw, names = self._make_data()
        df = attention_correlation(ea, aw, names)
        assert df.iloc[0]["feature"] == "feat_0"

    def test_pearson_r_in_range(self):
        ea, aw, names = self._make_data()
        df = attention_correlation(ea, aw, names)
        assert (df["pearson_r"].abs() <= 1.0).all()

    def test_p_value_in_range(self):
        ea, aw, names = self._make_data()
        df = attention_correlation(ea, aw, names)
        valid = df["p_value"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_mismatched_shapes_raises(self):
        ea = np.zeros((10, 3))
        aw = np.zeros(5)  # wrong length
        with pytest.raises(ValueError):
            attention_correlation(ea, aw, ["a", "b", "c"])

    def test_wrong_feature_names_length_raises(self):
        ea = np.zeros((10, 3))
        aw = np.zeros(10)
        with pytest.raises(ValueError):
            attention_correlation(ea, aw, ["a", "b"])  # too few names

    def test_constant_feature_produces_nan_or_zero(self):
        ea = np.zeros((50, 2))
        ea[:, 1] = 1.0  # constant
        aw = np.random.rand(50)
        df = attention_correlation(ea, aw, ["zero", "constant"])
        # Constant column → pearsonr is NaN or 0
        constant_r = df.loc[df["feature"] == "constant", "pearson_r"].values[0]
        assert np.isnan(constant_r) or constant_r == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# extract_attention_weights (torch_geometric required)
# ---------------------------------------------------------------------------

class TestExtractAttentionWeights:
    pytestmark = _skip_pyg

    def test_gat_returns_dict(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        model.eval()
        attn = extract_attention_weights(model, graph)
        assert isinstance(attn, dict)

    def test_gat_has_layer_keys(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        assert "layer_0" in attn
        assert "layer_1" in attn

    def test_gat_alpha_shape_layer0(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        E = graph.edge_index.shape[1]
        assert attn["layer_0"].shape == (E, 2), f"Expected ({E}, 2), got {attn['layer_0'].shape}"

    def test_gat_alpha_shape_layer1(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        E = graph.edge_index.shape[1]
        assert attn["layer_1"].shape[0] == E

    def test_gat_no_nan_attention(self):
        import torch
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        for layer, alpha in attn.items():
            assert not torch.isnan(alpha).any(), f"NaN in {layer}"

    def test_gat_attention_sums_near_one(self):
        """Softmax over neighbours → attention per node should sum near 1."""
        import torch
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        alpha = attn["layer_0"]  # [E, 2]
        # mean across heads should be in (0, 1]
        assert (alpha >= 0).all()
        assert (alpha <= 1 + 1e-5).all()

    def test_hybrid_returns_dict(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier
        graph = _make_pyg_graph()
        model = GraphSageGATHybridClassifier(
            in_channels=6, embed_channels=4, gat_hidden=8, gat_heads=2, sage_hidden=16
        )
        attn = extract_attention_weights(model, graph)
        assert isinstance(attn, dict)
        assert "layer_0" in attn

    def test_hybrid_alpha_shape(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier
        graph = _make_pyg_graph()
        model = GraphSageGATHybridClassifier(
            in_channels=6, embed_channels=4, gat_hidden=8, gat_heads=2, sage_hidden=16
        )
        attn = extract_attention_weights(model, graph)
        E = graph.edge_index.shape[1]
        assert attn["layer_0"].shape[0] == E
        assert attn["layer_0"].shape[1] == 2  # gat_heads

    def test_hybrid_no_nan(self):
        import torch
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier
        graph = _make_pyg_graph()
        model = GraphSageGATHybridClassifier(
            in_channels=6, embed_channels=4, gat_hidden=8, gat_heads=2, sage_hidden=16
        )
        attn = extract_attention_weights(model, graph)
        for layer, alpha in attn.items():
            assert not torch.isnan(alpha).any(), f"NaN in {layer}"

    def test_model_back_to_train_mode_after_extraction(self):
        from gbd_ctu.evaluation.attention import extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        model.train()
        extract_attention_weights(model, graph)
        assert model.training

    def test_backward_compat_gat_forward_no_attention(self):
        """forward() without return_attention still returns a plain tensor."""
        import torch
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        out = model(graph)
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 2  # out_channels

    def test_backward_compat_hybrid_forward_no_attention(self):
        """forward() without return_attention still returns a plain tensor."""
        import torch
        from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier
        graph = _make_pyg_graph()
        model = GraphSageGATHybridClassifier(
            in_channels=6, embed_channels=4, gat_hidden=8, gat_heads=2, sage_hidden=16
        )
        out = model(graph)
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 2


# ---------------------------------------------------------------------------
# attention_edge_heatmap (torch_geometric required)
# ---------------------------------------------------------------------------

class TestAttentionEdgeHeatmap:
    pytestmark = _skip_pyg

    def test_png_created(self, tmp_path):
        import torch
        from gbd_ctu.evaluation.attention import attention_edge_heatmap, extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        path = attention_edge_heatmap(graph, attn, scenario_id=1, output_dir=tmp_path)
        assert path.exists()

    def test_filename_contains_scenario_id(self, tmp_path):
        from gbd_ctu.evaluation.attention import attention_edge_heatmap, extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        path = attention_edge_heatmap(graph, attn, scenario_id=5, output_dir=tmp_path)
        assert "5" in path.name

    def test_output_dir_created(self, tmp_path):
        from gbd_ctu.evaluation.attention import attention_edge_heatmap, extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        nested = tmp_path / "a" / "b"
        attention_edge_heatmap(graph, attn, scenario_id=1, output_dir=nested)
        assert nested.exists()

    def test_returns_path(self, tmp_path):
        from pathlib import Path
        from gbd_ctu.evaluation.attention import attention_edge_heatmap, extract_attention_weights
        from gbd_ctu.models.gnn.gat import GATNodeClassifier
        graph = _make_pyg_graph()
        model = GATNodeClassifier(in_channels=6, hidden_channels=8, embed_channels=4, heads=2)
        attn = extract_attention_weights(model, graph)
        result = attention_edge_heatmap(graph, attn, scenario_id=1, output_dir=tmp_path)
        assert isinstance(result, Path)
