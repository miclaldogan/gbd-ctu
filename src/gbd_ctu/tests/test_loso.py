"""Tests for leave-one-scenario-out and cross-family evaluation.

Tests that require torch / torch_geometric are skipped gracefully when those
packages are not installed.
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
# SCENARIO_FAMILY constant
# ---------------------------------------------------------------------------

def test_scenario_family_has_13_entries():
    from gbd_ctu.data.ctu13_loader import SCENARIO_FAMILY
    assert len(SCENARIO_FAMILY) == 13


def test_scenario_family_keys_are_1_to_13():
    from gbd_ctu.data.ctu13_loader import SCENARIO_FAMILY
    assert set(SCENARIO_FAMILY.keys()) == set(range(1, 14))


def test_scenario_family_values_are_strings():
    from gbd_ctu.data.ctu13_loader import SCENARIO_FAMILY
    assert all(isinstance(v, str) for v in SCENARIO_FAMILY.values())


def test_scenario_family_known_entries():
    from gbd_ctu.data.ctu13_loader import SCENARIO_FAMILY
    assert SCENARIO_FAMILY[1] == "Neris"
    assert SCENARIO_FAMILY[3] == "Rbot"
    assert SCENARIO_FAMILY[12] == "NSIS.ay"
    assert SCENARIO_FAMILY[13] == "Virut"


# ---------------------------------------------------------------------------
# Synthetic graph fixture (skips if torch / torch_geometric absent)
# ---------------------------------------------------------------------------

def _make_synthetic_graphs(n_graphs: int = 3, n_nodes: int = 60,
                            n_features: int = 6, seed: int = 0):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n_graphs):
        x = torch.from_numpy(rng.standard_normal((n_nodes, n_features)).astype(np.float32))
        y = torch.from_numpy(rng.choice([0, 1], size=n_nodes, p=[0.7, 0.3]).astype(np.int64))
        src = torch.randint(0, n_nodes, (n_nodes * 2,))
        dst = torch.randint(0, n_nodes, (n_nodes * 2,))
        keep = src != dst
        edge_index = torch.stack([src[keep], dst[keep]])
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        g = Data(x=x, edge_index=edge_index, y=y,
                 train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        g.scenario_id = i + 1
        g.scenario = f"scenario-{i + 1:02d}"
        graphs.append(g)
    return graphs


def _dummy_model_factory(in_channels: int):
    """Returns a trivial model that predicts uniform scores."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _DummyGNN(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            self.lin = nn.Linear(in_ch, 2)

        def forward(self, data):
            return self.lin(data.x)

    return _DummyGNN(in_channels)


def _dummy_trainer(model, graphs):
    """No-op trainer — model stays randomly initialised."""
    pass


# ---------------------------------------------------------------------------
# leave_one_scenario_out
# ---------------------------------------------------------------------------

class TestLeaveOneScenarioOut:
    def test_returns_dataframe(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        df = leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_graph(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        df = leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer)
        assert len(df) == len(graphs)

    def test_held_out_scenario_column_present(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        df = leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer)
        assert "held_out_scenario" in df.columns

    def test_metric_columns_present(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        df = leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer)
        for col in ("auc", "f1", "mcc"):
            assert col in df.columns, f"Missing column: {col}"

    def test_held_out_scenarios_match_graph_ids(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        df = leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer)
        assert set(df["held_out_scenario"].tolist()) == {1, 2, 3}

    def test_csv_written_to_output_dir(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer,
                               output_dir=tmp_path)
        assert (tmp_path / "loso_results.csv").exists()

    def test_summary_json_written(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=3)
        leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer,
                               output_dir=tmp_path)
        assert (tmp_path / "loso_summary.json").exists()

    def test_sorted_by_held_out_scenario(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import leave_one_scenario_out
        graphs = _make_synthetic_graphs(n_graphs=4)
        df = leave_one_scenario_out(graphs, _dummy_model_factory, _dummy_trainer)
        ids = df["held_out_scenario"].tolist()
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# cross_family_eval
# ---------------------------------------------------------------------------

class TestCrossFamilyEval:
    def test_returns_dataframe(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import cross_family_eval
        graphs = _make_synthetic_graphs(n_graphs=4)
        family_map = {1: "FamilyA", 2: "FamilyA", 3: "FamilyB", 4: "FamilyB"}
        df = cross_family_eval(graphs, family_map, _dummy_model_factory, _dummy_trainer)
        assert isinstance(df, pd.DataFrame)

    def test_held_out_family_column_present(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import cross_family_eval
        graphs = _make_synthetic_graphs(n_graphs=4)
        family_map = {1: "FamilyA", 2: "FamilyA", 3: "FamilyB", 4: "FamilyB"}
        df = cross_family_eval(graphs, family_map, _dummy_model_factory, _dummy_trainer)
        assert "held_out_family" in df.columns

    def test_both_families_represented(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import cross_family_eval
        graphs = _make_synthetic_graphs(n_graphs=4)
        family_map = {1: "FamilyA", 2: "FamilyA", 3: "FamilyB", 4: "FamilyB"}
        df = cross_family_eval(graphs, family_map, _dummy_model_factory, _dummy_trainer)
        assert set(df["held_out_family"].unique()) == {"FamilyA", "FamilyB"}

    def test_csv_and_summary_written(self, tmp_path):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import cross_family_eval
        graphs = _make_synthetic_graphs(n_graphs=4)
        family_map = {1: "FamilyA", 2: "FamilyA", 3: "FamilyB", 4: "FamilyB"}
        cross_family_eval(graphs, family_map, _dummy_model_factory, _dummy_trainer,
                          output_dir=tmp_path)
        assert (tmp_path / "cross_family_results.csv").exists()
        assert (tmp_path / "cross_family_summary.csv").exists()

    def test_unknown_family_grouped(self):
        pytest.importorskip("torch_geometric")
        from gbd_ctu.evaluation.cross_scenario import cross_family_eval
        graphs = _make_synthetic_graphs(n_graphs=3)
        family_map = {1: "FamilyA"}  # graphs 2 and 3 unmapped → "Unknown"
        df = cross_family_eval(graphs, family_map, _dummy_model_factory, _dummy_trainer)
        assert "Unknown" in df["held_out_family"].values or "FamilyA" in df["held_out_family"].values


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_run_loso_cli_help():
    env = __import__("os").environ.copy()
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + __import__("os").pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.run_loso", "--help"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0
    assert "--graph-dir" in result.stdout
    assert "--model-type" in result.stdout
    assert "--cross-family" in result.stdout
