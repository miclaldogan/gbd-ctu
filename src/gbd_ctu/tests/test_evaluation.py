"""Tests for GBD-CTU evaluation metrics and scenario evaluator.

Covers:
  - Standalone metric functions with synthetic binary data.
  - confusion_matrix_plot returns a matplotlib Figure.
  - evaluate_all_scenarios with synthetic PyG graphs.
  - evaluate_gnn_checkpoint round-trip (save checkpoint → evaluate).
  - scripts/evaluate.py CLI help flag.
  - Results CSV written correctly.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gbd_ctu.evaluation.metrics import (
    auc_roc,
    classification_metrics,
    confusion_matrix_plot,
    f1_botnet,
    false_positive_rate,
    precision_recall,
)
from gbd_ctu.evaluation.scenario_eval import evaluate_all_scenarios, evaluate_gnn_checkpoint
from gbd_ctu.training.trainer_gnn import _make_synthetic_graph, train_gnn

_pyg_available = importlib.util.find_spec("torch_geometric") is not None
_skip_pyg = pytest.mark.skipif(not _pyg_available, reason="torch_geometric not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_data(n: int = 100, seed: int = 0):
    """Return (y_true, y_score) with both classes present."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)
    return y_true.astype(int), y_score.astype(float)


def _pred_from_score(y_score: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_score >= threshold).astype(int)


# ---------------------------------------------------------------------------
# auc_roc
# ---------------------------------------------------------------------------

class TestAucRoc:
    def test_returns_float(self):
        y_true, y_score = _rng_data()
        result = auc_roc(y_true, y_score)
        assert isinstance(result, float)

    def test_value_in_range(self):
        y_true, y_score = _rng_data()
        result = auc_roc(y_true, y_score)
        assert 0.0 <= result <= 1.0

    def test_perfect_predictor(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert auc_roc(y_true, y_score) == pytest.approx(1.0)

    def test_single_class_returns_nan(self):
        y_true = np.zeros(10, dtype=int)
        y_score = np.random.rand(10)
        assert np.isnan(auc_roc(y_true, y_score))


# ---------------------------------------------------------------------------
# f1_botnet
# ---------------------------------------------------------------------------

class TestF1Botnet:
    def test_returns_float(self):
        y_true, y_score = _rng_data()
        result = f1_botnet(y_true, _pred_from_score(y_score))
        assert isinstance(result, float)

    def test_perfect_f1(self):
        y_true = np.array([0, 1, 0, 1])
        assert f1_botnet(y_true, y_true) == pytest.approx(1.0)

    def test_all_wrong_f1(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert f1_botnet(y_true, y_pred) == pytest.approx(0.0)

    def test_no_botnet_zero_division_safe(self):
        y_true = np.zeros(10, dtype=int)
        y_pred = np.zeros(10, dtype=int)
        assert f1_botnet(y_true, y_pred) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# precision_recall
# ---------------------------------------------------------------------------

class TestPrecisionRecall:
    def test_returns_tuple_of_floats(self):
        y_true, y_score = _rng_data()
        result = precision_recall(y_true, _pred_from_score(y_score))
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_both_in_range(self):
        y_true, y_score = _rng_data()
        prec, rec = precision_recall(y_true, _pred_from_score(y_score))
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        prec, rec = precision_recall(y_true, y_true)
        assert prec == pytest.approx(1.0)
        assert rec == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# false_positive_rate
# ---------------------------------------------------------------------------

class TestFalsePositiveRate:
    def test_returns_float(self):
        y_true, y_score = _rng_data()
        result = false_positive_rate(y_true, _pred_from_score(y_score))
        assert isinstance(result, float)

    def test_in_range(self):
        y_true, y_score = _rng_data()
        fpr = false_positive_rate(y_true, _pred_from_score(y_score))
        assert 0.0 <= fpr <= 1.0

    def test_zero_fpr_on_no_fp(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])  # all correct
        assert false_positive_rate(y_true, y_pred) == pytest.approx(0.0)

    def test_max_fpr_all_benign_flagged(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # all flagged as botnet
        assert false_positive_rate(y_true, y_pred) == pytest.approx(1.0)

    def test_no_benign_class_safe(self):
        # All botnet; FPR denominator is 0 → should not raise
        y_true = np.ones(10, dtype=int)
        y_pred = np.ones(10, dtype=int)
        result = false_positive_rate(y_true, y_pred)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# confusion_matrix_plot
# ---------------------------------------------------------------------------

class TestConfusionMatrixPlot:
    def test_returns_figure(self):
        import matplotlib.figure
        y_true, y_score = _rng_data()
        fig = confusion_matrix_plot(y_true, _pred_from_score(y_score))
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_title(self):
        import matplotlib.figure
        y_true = np.array([0, 1, 0, 1])
        fig = confusion_matrix_plot(y_true, y_true, title="My Plot")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_axes(self):
        y_true = np.array([0, 1, 0, 1])
        fig = confusion_matrix_plot(y_true, y_true)
        assert len(fig.axes) >= 1


# ---------------------------------------------------------------------------
# evaluate_all_scenarios
# ---------------------------------------------------------------------------

def _dummy_model(n_features: int = 30):
    """Return an untrained GraphSAGE model for testing."""
    from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
    return GraphSAGENodeClassifier(in_channels=n_features, hidden_channels=16,
                                   out_channels=2, num_layers=2, dropout=0.0)


class TestEvaluateAllScenarios:
    pytestmark = _skip_pyg

    def _graphs(self, n: int = 3, n_nodes: int = 40):
        return [_make_synthetic_graph(n_nodes=n_nodes, seed=i) for i in range(n)]

    def test_returns_dataframe(self):
        model = _dummy_model()
        graphs = self._graphs()
        result = evaluate_all_scenarios(model, graphs)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        model = _dummy_model()
        graphs = self._graphs()
        result = evaluate_all_scenarios(model, graphs)
        expected = {"scenario_id", "scenario", "auc", "f1", "precision", "recall",
                    "fpr", "n_botnet_nodes", "n_total_nodes"}
        assert expected.issubset(set(result.columns))

    def test_one_row_per_graph(self):
        model = _dummy_model()
        graphs = self._graphs(n=3)
        result = evaluate_all_scenarios(model, graphs, split="test")
        assert len(result) == 3

    def test_n_total_nodes_matches_mask(self):
        model = _dummy_model()
        graphs = self._graphs(n=2, n_nodes=50)
        result = evaluate_all_scenarios(model, graphs, split="test")
        for i, graph in enumerate(graphs):
            expected_total = int(graph.test_mask.sum())
            assert result.iloc[i]["n_total_nodes"] == expected_total

    def test_csv_written_when_output_dir_provided(self, tmp_path):
        model = _dummy_model()
        graphs = self._graphs(n=2)
        result = evaluate_all_scenarios(model, graphs, model_name="sage",
                                         output_dir=tmp_path)
        csv_path = tmp_path / "results_sage.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(result)

    def test_sorted_by_scenario_id(self):
        model = _dummy_model()
        graphs = self._graphs(n=5)
        result = evaluate_all_scenarios(model, graphs)
        ids = result["scenario_id"].tolist()
        assert ids == sorted(ids)

    def test_empty_graph_list_returns_empty_frame(self):
        model = _dummy_model()
        result = evaluate_all_scenarios(model, [])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# evaluate_gnn_checkpoint
# ---------------------------------------------------------------------------

class TestEvaluateGnnCheckpoint:
    pytestmark = _skip_pyg

    def test_round_trip(self, tmp_path):
        """Train dry-run → save checkpoint → evaluate_gnn_checkpoint → CSV exists."""
        ckpt = tmp_path / "graphsage_1.pt"
        train_gnn(family="graphsage", checkpoint_path=ckpt, dry_run=True, seed=0)

        # Patch checkpoint with synthetic graphs compatible with test
        state = torch.load(ckpt, weights_only=False)
        # Align graph_dir with the checkpoint's actual in_channels
        in_channels = state["model_kwargs"]["in_channels"]
        torch.save(state, ckpt)

        graph_dir = tmp_path / "graphs"
        graph_dir.mkdir()
        g = _make_synthetic_graph(n_features=in_channels, seed=42)
        torch.save(g, graph_dir / "scenario_synthetic.pt")

        output_csv = tmp_path / "reports" / "results.csv"
        report = evaluate_gnn_checkpoint(
            graph_dir=graph_dir,
            checkpoint_path=ckpt,
            output_path=output_csv,
        )
        assert isinstance(report, pd.DataFrame)
        assert output_csv.exists()


# ---------------------------------------------------------------------------
# CLI: scripts/evaluate.py
# ---------------------------------------------------------------------------

class TestEvaluateCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.evaluate", "--help"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent)},
        )
        assert result.returncode == 0
        assert "--checkpoint" in result.stdout
        assert "--model" in result.stdout

    def test_missing_checkpoint_errors(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.evaluate"],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent)},
        )
        assert result.returncode != 0
