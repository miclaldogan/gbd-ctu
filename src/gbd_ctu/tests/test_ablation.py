"""Tests for the ablation study module.

Tests that require torch / torch_geometric skip gracefully when absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# AblationConfig dataclass
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_instantiation_with_name_only(self):
        from gbd_ctu.evaluation.ablation import AblationConfig
        cfg = AblationConfig(name="test")
        assert cfg.name == "test"
        assert cfg.model_kwargs == {}
        assert cfg.graph_kwargs == {}
        assert cfg.trainer_kwargs == {}

    def test_from_yaml_dict_full_model(self):
        from gbd_ctu.evaluation.ablation import AblationConfig
        entry = {
            "name": "full_model",
            "model": {"type": "hybrid", "hidden": 128},
            "graph": {"use_edge_attr": True},
            "loss": {"type": "focal", "gamma": 2.0, "alpha": 0.75},
        }
        cfg = AblationConfig.from_yaml_dict(entry)
        assert cfg.name == "full_model"
        assert cfg.model_kwargs == {"type": "hybrid", "hidden": 128}
        assert cfg.graph_kwargs == {"use_edge_attr": True}
        assert cfg.trainer_kwargs["loss"]["type"] == "focal"

    def test_from_yaml_dict_missing_optional_keys(self):
        from gbd_ctu.evaluation.ablation import AblationConfig
        entry = {"name": "no_edge_features", "graph": {"use_edge_attr": False}}
        cfg = AblationConfig.from_yaml_dict(entry)
        assert cfg.graph_kwargs["use_edge_attr"] is False
        assert cfg.model_kwargs == {}


# ---------------------------------------------------------------------------
# configs_from_yaml
# ---------------------------------------------------------------------------

class TestConfigsFromYaml:
    def test_bundled_yaml_loads(self):
        pytest.importorskip("yaml")
        from gbd_ctu.evaluation.ablation import configs_from_yaml
        configs = configs_from_yaml()
        assert len(configs) == 10

    def test_all_configs_have_names(self):
        pytest.importorskip("yaml")
        from gbd_ctu.evaluation.ablation import configs_from_yaml
        configs = configs_from_yaml()
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), "Duplicate ablation names"

    def test_expected_ablation_names_present(self):
        pytest.importorskip("yaml")
        from gbd_ctu.evaluation.ablation import configs_from_yaml
        configs = configs_from_yaml()
        names = {c.name for c in configs}
        for expected in ("full_model", "no_edge_features", "cross_entropy_loss",
                         "2layer_sage", "undirected_graph", "self_loops"):
            assert expected in names, f"Missing ablation: {expected}"

    def test_custom_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml
        from gbd_ctu.evaluation.ablation import configs_from_yaml
        custom = {"ablations": [{"name": "my_ablation", "model": {"type": "gat"}}]}
        p = tmp_path / "custom.yaml"
        p.write_text(yaml.dump(custom), encoding="utf-8")
        configs = configs_from_yaml(p)
        assert len(configs) == 1
        assert configs[0].name == "my_ablation"


# ---------------------------------------------------------------------------
# _apply_graph_transforms (pure Python, no torch_geometric needed)
# ---------------------------------------------------------------------------

class TestApplyGraphTransforms:
    def _fake_graph(self, n_features=6):
        """Return a minimal mock graph object."""
        torch = pytest.importorskip("torch")

        class FakeGraph:
            def __init__(self):
                self.x = torch.randn(20, n_features)
                self.edge_index = torch.zeros(2, 10, dtype=torch.long)
                self.edge_attr = torch.randn(10, 22)
                self.num_node_features = n_features
                self.num_nodes = 20

        return FakeGraph()

    def test_use_edge_attr_false_clears_edge_attr(self):
        from gbd_ctu.evaluation.ablation import _apply_graph_transforms
        g = self._fake_graph()
        result = _apply_graph_transforms([g], {"use_edge_attr": False})
        assert result[0].edge_attr is None

    def test_no_transform_leaves_x_unchanged(self):
        from gbd_ctu.evaluation.ablation import _apply_graph_transforms
        g = self._fake_graph()
        result = _apply_graph_transforms([g], {})
        assert result[0].x.shape == g.x.shape

    def test_node_feature_subset_reduces_columns(self):
        from gbd_ctu.evaluation.ablation import _apply_graph_transforms
        g = self._fake_graph(n_features=6)
        result = _apply_graph_transforms(
            [g],
            {"node_features": ["flow_count", "total_bytes_sent", "botnet_flow_ratio"]},
        )
        assert result[0].x.shape[1] == 3

    def test_all_node_features_unchanged(self):
        from gbd_ctu.evaluation.ablation import _apply_graph_transforms
        g = self._fake_graph(n_features=6)
        result = _apply_graph_transforms([g], {"node_features": "all"})
        assert result[0].x.shape[1] == 6


# ---------------------------------------------------------------------------
# _resolve_node_feature_indices
# ---------------------------------------------------------------------------

def test_resolve_all_returns_none():
    from gbd_ctu.evaluation.ablation import _resolve_node_feature_indices
    assert _resolve_node_feature_indices("all") is None


def test_resolve_subset_returns_indices():
    from gbd_ctu.evaluation.ablation import _resolve_node_feature_indices
    indices = _resolve_node_feature_indices(
        ["flow_count", "total_bytes_sent", "botnet_flow_ratio"]
    )
    assert indices == [0, 2, 5]


# ---------------------------------------------------------------------------
# run_ablation_suite (mocked graph loading, no torch_geometric needed for
# the summary-shape tests; skipped for training tests)
# ---------------------------------------------------------------------------

def _make_fake_graphs(n=2):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    rng = np.random.default_rng(0)
    graphs = []
    for i in range(n):
        n_nodes = 50
        x = torch.from_numpy(rng.standard_normal((n_nodes, 6)).astype(np.float32))
        y = torch.from_numpy(rng.choice([0, 1], size=n_nodes, p=[0.7, 0.3]).astype(np.int64))
        ei = torch.zeros(2, 20, dtype=torch.long)
        tm = torch.zeros(n_nodes, dtype=torch.bool)
        vm = torch.zeros(n_nodes, dtype=torch.bool)
        tsm = torch.zeros(n_nodes, dtype=torch.bool)
        tm[:30] = True; vm[30:40] = True; tsm[40:] = True
        g = Data(x=x, edge_index=ei, y=y,
                 train_mask=tm, val_mask=vm, test_mask=tsm)
        g.scenario_id = i + 1
        g.scenario = f"s{i+1}"
        graphs.append(g)
    return graphs


class TestRunAblationSuite:
    def test_returns_dataframe(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.ablation import AblationConfig, run_ablation_suite

        graphs = _make_fake_graphs()
        cfg = AblationConfig(
            name="test_ablation",
            model_kwargs={"type": "hybrid"},
            trainer_kwargs={"epochs": 1, "loss": {"type": "focal"}},
        )
        with patch("gbd_ctu.evaluation.ablation.load_graphs", return_value=graphs):
            df = run_ablation_suite([cfg], graph_dir=tmp_path, output_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_config(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.ablation import AblationConfig, run_ablation_suite

        graphs = _make_fake_graphs()
        cfgs = [
            AblationConfig(name="a", trainer_kwargs={"epochs": 1}),
            AblationConfig(name="b", trainer_kwargs={"epochs": 1}),
        ]
        with patch("gbd_ctu.evaluation.ablation.load_graphs", return_value=graphs):
            df = run_ablation_suite(cfgs, graph_dir=tmp_path, output_dir=tmp_path)
        assert len(df) == 2

    def test_name_column_present(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.ablation import AblationConfig, run_ablation_suite

        graphs = _make_fake_graphs()
        cfg = AblationConfig(name="my_cfg", trainer_kwargs={"epochs": 1})
        with patch("gbd_ctu.evaluation.ablation.load_graphs", return_value=graphs):
            df = run_ablation_suite([cfg], graph_dir=tmp_path, output_dir=tmp_path)
        assert "name" in df.columns
        assert df["name"].iloc[0] == "my_cfg"

    def test_metric_columns_present(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.ablation import AblationConfig, run_ablation_suite

        graphs = _make_fake_graphs()
        cfg = AblationConfig(name="x", trainer_kwargs={"epochs": 1})
        with patch("gbd_ctu.evaluation.ablation.load_graphs", return_value=graphs):
            df = run_ablation_suite([cfg], graph_dir=tmp_path, output_dir=tmp_path)
        for col in ("mean_auc", "std_auc", "mean_f1"):
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_written(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.ablation import AblationConfig, run_ablation_suite

        graphs = _make_fake_graphs()
        cfg = AblationConfig(name="y", trainer_kwargs={"epochs": 1})
        with patch("gbd_ctu.evaluation.ablation.load_graphs", return_value=graphs):
            run_ablation_suite([cfg], graph_dir=tmp_path, output_dir=tmp_path)
        assert (tmp_path / "ablation_results.csv").exists()
        assert (tmp_path / "ablation_results.json").exists()


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def _env_with_src():
    import os
    env = os.environ.copy()
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_run_ablation_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.run_ablation", "--help"],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode == 0
    assert "--graph-dir" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--latex" in result.stdout


def test_run_ablation_dry_run(tmp_path):
    pytest.importorskip("yaml")
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.run_ablation",
         "--graph-dir", str(tmp_path), "--dry-run"],
        capture_output=True, text=True, env=_env_with_src(),
    )
    assert result.returncode == 0
    assert "full_model" in result.stdout
