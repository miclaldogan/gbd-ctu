"""GBD-CTU baseline classifier tests.

Tests cover the XGBoostBaseline and RandomForestBaseline wrappers:
- fit / predict / predict_proba on small synthetic arrays
- feature_importance() DataFrame schema and content
- scale_pos_weight auto-computation in XGBoostBaseline
- class_weight='balanced' in RandomForestBaseline
- guard against calling predict/predict_proba before fit
- train_baselines CLI --scenario flag (smoke test)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbd_ctu.models.baselines.random_forest_clf import RandomForestBaseline

# XGBoost is optional — skip only the XGBoost test class when not installed
try:
    import xgboost as _xgb  # noqa: F401
    _xgboost_available = True
except ImportError:
    _xgboost_available = False

from gbd_ctu.models.baselines.xgboost_clf import XGBoostBaseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_samples: int = 100, n_features: int = 22, seed: int = 0):
    """Return a small imbalanced synthetic (X, y) pair."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # 80 % negatives / 20 % positives — mimics botnet imbalance
    y = rng.choice([0, 1], size=n_samples, p=[0.8, 0.2]).astype(np.int64)
    return x, y


FEATURE_NAMES = [f"feat_{i}" for i in range(22)]


# ---------------------------------------------------------------------------
# RandomForestBaseline
# ---------------------------------------------------------------------------

class TestRandomForestBaseline:
    def _fitted(self):
        clf = RandomForestBaseline(n_estimators=10, random_state=0, n_jobs=1)
        x, y = _make_data()
        clf.fit(x, y)
        return clf, x, y

    def test_fit_returns_self(self):
        clf = RandomForestBaseline(n_estimators=10, n_jobs=1)
        x, y = _make_data()
        assert clf.fit(x, y) is clf

    def test_predict_shape(self):
        clf, x, _ = self._fitted()
        preds = clf.predict(x)
        assert preds.shape == (len(x),)

    def test_predict_values_binary(self):
        clf, x, _ = self._fitted()
        assert set(clf.predict(x).tolist()).issubset({0, 1})

    def test_predict_proba_shape(self):
        clf, x, _ = self._fitted()
        proba = clf.predict_proba(x)
        assert proba.shape == (len(x), 2)

    def test_predict_proba_sums_to_1(self):
        clf, x, _ = self._fitted()
        proba = clf.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(x)), atol=1e-5)

    def test_feature_importance_columns(self):
        clf = RandomForestBaseline(n_estimators=10, n_jobs=1)
        x, y = _make_data()
        clf.fit(x, y, feature_names=FEATURE_NAMES)
        df = clf.feature_importance()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["feature", "importance"]

    def test_feature_importance_row_count(self):
        clf = RandomForestBaseline(n_estimators=10, n_jobs=1)
        x, y = _make_data()
        clf.fit(x, y, feature_names=FEATURE_NAMES)
        df = clf.feature_importance()
        assert len(df) == x.shape[1]

    def test_feature_importance_uses_supplied_names(self):
        clf = RandomForestBaseline(n_estimators=10, n_jobs=1)
        x, y = _make_data()
        clf.fit(x, y, feature_names=FEATURE_NAMES)
        assert set(clf.feature_importance()["feature"]) == set(FEATURE_NAMES)

    def test_feature_importance_fallback_names(self):
        clf = RandomForestBaseline(n_estimators=10, n_jobs=1)
        x, y = _make_data()
        clf.fit(x, y)  # no feature_names
        df = clf.feature_importance()
        assert df["feature"].iloc[0].startswith("f")

    def test_feature_importance_sorted_descending(self):
        clf = RandomForestBaseline(n_estimators=10, n_jobs=1)
        x, y = _make_data()
        clf.fit(x, y)
        importances = clf.feature_importance()["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_class_weight_is_balanced(self):
        clf = RandomForestBaseline()
        assert clf.model.class_weight == "balanced"

    def test_get_params_returns_dict(self):
        clf, _, _ = self._fitted()
        params = clf.get_params()
        assert isinstance(params, dict)
        assert "n_estimators" in params


# ---------------------------------------------------------------------------
# XGBoostBaseline
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _xgboost_available, reason="xgboost not installed")
class TestXGBoostBaseline:
    def _fitted(self, feature_names=None):
        clf = XGBoostBaseline(n_estimators=10, random_state=0)
        x, y = _make_data()
        clf.fit(x, y, feature_names=feature_names)
        return clf, x, y

    def test_fit_returns_self(self):
        clf = XGBoostBaseline(n_estimators=10)
        x, y = _make_data()
        assert clf.fit(x, y) is clf

    def test_predict_shape(self):
        clf, x, _ = self._fitted()
        preds = clf.predict(x)
        assert preds.shape == (len(x),)

    def test_predict_values_binary(self):
        clf, x, _ = self._fitted()
        assert set(clf.predict(x).tolist()).issubset({0, 1})

    def test_predict_proba_shape(self):
        clf, x, _ = self._fitted()
        proba = clf.predict_proba(x)
        assert proba.shape == (len(x), 2)

    def test_predict_proba_sums_to_1(self):
        clf, x, _ = self._fitted()
        proba = clf.predict_proba(x)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(x)), atol=1e-5)

    def test_feature_importance_columns(self):
        clf, _, _ = self._fitted(feature_names=FEATURE_NAMES)
        df = clf.feature_importance()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["feature", "importance"]

    def test_feature_importance_row_count(self):
        clf, x, _ = self._fitted(feature_names=FEATURE_NAMES)
        assert len(clf.feature_importance()) == x.shape[1]

    def test_feature_importance_uses_supplied_names(self):
        clf, _, _ = self._fitted(feature_names=FEATURE_NAMES)
        assert set(clf.feature_importance()["feature"]) == set(FEATURE_NAMES)

    def test_feature_importance_sorted_descending(self):
        clf, _, _ = self._fitted()
        importances = clf.feature_importance()["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_scale_pos_weight_computed_from_labels(self):
        """scale_pos_weight must equal n_neg/n_pos derived from training labels."""
        x, y = _make_data()
        n_neg = int(np.sum(y == 0))
        n_pos = int(np.sum(y == 1))
        expected = n_neg / n_pos
        clf = XGBoostBaseline(n_estimators=5)
        clf.fit(x, y)
        actual = clf.model.get_params()["scale_pos_weight"]
        assert abs(actual - expected) < 1e-6

    def test_scale_pos_weight_with_perfectly_balanced_labels(self):
        """When perfectly balanced, scale_pos_weight must be 1.0."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((100, 5)).astype(np.float32)
        y = np.array([0, 1] * 50, dtype=np.int64)
        clf = XGBoostBaseline(n_estimators=5)
        clf.fit(x, y)
        assert clf.model.get_params()["scale_pos_weight"] == pytest.approx(1.0)

    def test_predict_before_fit_raises(self):
        clf = XGBoostBaseline(n_estimators=5)
        x, _ = _make_data()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(x)

    def test_predict_proba_before_fit_raises(self):
        clf = XGBoostBaseline(n_estimators=5)
        x, _ = _make_data()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(x)

    def test_feature_importance_before_fit_raises(self):
        clf = XGBoostBaseline(n_estimators=5)
        with pytest.raises(RuntimeError, match="fit"):
            clf.feature_importance()

    def test_get_params_before_fit_returns_init_kwargs(self):
        clf = XGBoostBaseline(n_estimators=77)
        params = clf.get_params()
        assert params["n_estimators"] == 77

    def test_get_params_after_fit_returns_dict(self):
        clf, _, _ = self._fitted()
        params = clf.get_params()
        assert isinstance(params, dict)
        assert "n_estimators" in params


# ---------------------------------------------------------------------------
# train_baselines integration (mocked graph loading)
# ---------------------------------------------------------------------------

def _make_graph(n_nodes: int = 80, n_features: int = 6, seed: int = 0):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((n_nodes, n_features)).astype(np.float32))
    y = torch.from_numpy(rng.choice([0, 1], size=n_nodes, p=[0.8, 0.2]).astype(np.int64))
    n_train, n_val = int(n_nodes * 0.6), int(n_nodes * 0.2)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    graph = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    graph.scenario = 1
    return graph


class TestTrainBaselines:
    def test_returns_dataframe_with_fold_column(self, tmp_path):
        graph = _make_graph()
        with patch("gbd_ctu.training.trainer_baseline.load_graphs", return_value=[graph]):
            from gbd_ctu.training.trainer_baseline import train_baselines
            report = train_baselines(graph_dir=tmp_path / "graphs", output_dir=tmp_path / "out")
        assert isinstance(report, pd.DataFrame)
        assert "fold" in report.columns

    def test_fold_values_are_zero_to_four(self, tmp_path):
        graph = _make_graph()
        with patch("gbd_ctu.training.trainer_baseline.load_graphs", return_value=[graph]):
            from gbd_ctu.training.trainer_baseline import train_baselines
            report = train_baselines(graph_dir=tmp_path / "graphs", output_dir=tmp_path / "out")
        assert set(report["fold"].unique()).issubset({0, 1, 2, 3, 4})

    def test_joblib_file_saved(self, tmp_path):
        graph = _make_graph()
        out = tmp_path / "out"
        with patch("gbd_ctu.training.trainer_baseline.load_graphs", return_value=[graph]):
            from gbd_ctu.training.trainer_baseline import train_baselines
            train_baselines(graph_dir=tmp_path / "graphs", output_dir=out)
        assert (out / "random_forest.joblib").exists()

    def test_summary_csv_saved(self, tmp_path):
        graph = _make_graph()
        out = tmp_path / "out"
        with patch("gbd_ctu.training.trainer_baseline.load_graphs", return_value=[graph]):
            from gbd_ctu.training.trainer_baseline import train_baselines
            train_baselines(graph_dir=tmp_path / "graphs", output_dir=out)
        assert (out / "baseline_cv_summary.csv").exists()

    def test_model_column_present(self, tmp_path):
        graph = _make_graph()
        with patch("gbd_ctu.training.trainer_baseline.load_graphs", return_value=[graph]):
            from gbd_ctu.training.trainer_baseline import train_baselines
            report = train_baselines(graph_dir=tmp_path / "graphs", output_dir=tmp_path / "out")
        assert "model" in report.columns
        assert "random_forest" in report["model"].values


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_train_baselines_cli_help():
    """--help must exit 0 and mention --scenario."""
    import os
    env = os.environ.copy()
    # Ensure the src layout is importable in the subprocess
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "gbd_ctu.scripts.train_baselines", "--help"],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0
    assert "--scenario" in result.stdout
