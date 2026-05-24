"""GBD-CTU metric tests.

These tests validate metric computation for binary classification outputs used in
CTU-13 scenario evaluation — covering AUPRC, MCC, FPR@TPR95, plots, and the
compare.py propagation of new metric columns.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gbd_ctu.evaluation.metrics import (
    _fpr_at_tpr,
    auc_roc,
    classification_metrics,
    f1_botnet,
    false_positive_rate,
    metrics_frame,
    precision_recall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _imbalanced_arrays(n_total: int = 1000, botnet_rate: float = 0.03, seed: int = 0):
    """Return (y_true, y_score) simulating a highly imbalanced dataset."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n_total) < botnet_rate).astype(int)
    y_score = np.where(y_true == 1, rng.uniform(0.6, 1.0, n_total), rng.uniform(0.0, 0.5, n_total))
    return y_true, y_score


def _perfect_arrays(n: int = 100, botnet_rate: float = 0.1):
    """Return perfectly-separated labels and scores."""
    y_true = np.array([1] * int(n * botnet_rate) + [0] * int(n * (1 - botnet_rate)))
    y_score = np.where(y_true == 1, 1.0, 0.0).astype(float)
    return y_true, y_score


def _all_same_class(n: int = 50):
    """Return arrays where only one class is present."""
    return np.zeros(n, dtype=int), np.random.rand(n)


# ---------------------------------------------------------------------------
# Original test (kept)
# ---------------------------------------------------------------------------

def test_classification_metrics_include_fpr_and_auc() -> None:
    """Metric output should include the requested evaluation fields."""
    report = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.4])
    assert report["auc"] > 0.9
    assert report["f1"] > 0.7
    assert 0.0 <= report["fpr"] <= 1.0


# ---------------------------------------------------------------------------
# _fpr_at_tpr (private helper)
# ---------------------------------------------------------------------------

def test_fpr_at_tpr_perfect_classifier() -> None:
    """A perfect classifier achieves 0 FPR at TPR=0.95."""
    y_true, y_score = _perfect_arrays()
    result = _fpr_at_tpr(y_true, y_score, tpr_target=0.95)
    assert result == pytest.approx(0.0, abs=1e-6)


def test_fpr_at_tpr_returns_nan_single_class() -> None:
    """_fpr_at_tpr must return nan when only one class is present."""
    y_true, y_score = _all_same_class()
    assert math.isnan(_fpr_at_tpr(y_true, y_score))


def test_fpr_at_tpr_value_in_range() -> None:
    """FPR@TPR95 must be in [0, 1] for a valid binary dataset."""
    y_true, y_score = _imbalanced_arrays()
    result = _fpr_at_tpr(y_true, y_score, tpr_target=0.95)
    assert 0.0 <= result <= 1.0


def test_fpr_at_tpr_higher_target_higher_fpr() -> None:
    """Higher TPR target should require a lower threshold and thus higher FPR."""
    y_true, y_score = _imbalanced_arrays()
    fpr_80 = _fpr_at_tpr(y_true, y_score, tpr_target=0.80)
    fpr_95 = _fpr_at_tpr(y_true, y_score, tpr_target=0.95)
    assert math.isnan(fpr_95) or fpr_95 >= fpr_80


def test_fpr_at_tpr_unreachable_target_returns_nan() -> None:
    """Returns nan if tpr_target > 1.0 (unreachable)."""
    y_true, y_score = _imbalanced_arrays()
    result = _fpr_at_tpr(y_true, y_score, tpr_target=2.0)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# classification_metrics — new keys
# ---------------------------------------------------------------------------

def test_classification_metrics_returns_auprc_key() -> None:
    y_true, y_score = _imbalanced_arrays()
    assert "auprc" in classification_metrics(y_true, y_score)


def test_classification_metrics_returns_mcc_key() -> None:
    y_true, y_score = _imbalanced_arrays()
    assert "mcc" in classification_metrics(y_true, y_score)


def test_classification_metrics_returns_fpr_at_tpr95_key() -> None:
    y_true, y_score = _imbalanced_arrays()
    assert "fpr_at_tpr95" in classification_metrics(y_true, y_score)


def test_classification_metrics_auprc_in_unit_interval() -> None:
    y_true, y_score = _imbalanced_arrays()
    m = classification_metrics(y_true, y_score)
    assert 0.0 <= m["auprc"] <= 1.0


def test_classification_metrics_auprc_perfect() -> None:
    """Perfect classifier must achieve AUPRC = 1.0."""
    y_true, y_score = _perfect_arrays()
    m = classification_metrics(y_true, y_score)
    assert m["auprc"] == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_auprc_nan_single_class() -> None:
    y_true, y_score = _all_same_class()
    m = classification_metrics(y_true, y_score)
    assert math.isnan(m["auprc"])


def test_classification_metrics_mcc_perfect() -> None:
    """Perfect predictions must give MCC = 1.0."""
    y_true, y_score = _perfect_arrays()
    m = classification_metrics(y_true, y_score)
    assert m["mcc"] == pytest.approx(1.0, abs=1e-6)


def test_classification_metrics_mcc_in_range() -> None:
    y_true, y_score = _imbalanced_arrays()
    m = classification_metrics(y_true, y_score)
    assert -1.0 <= m["mcc"] <= 1.0


def test_classification_metrics_fpr_at_tpr95_in_range() -> None:
    y_true, y_score = _imbalanced_arrays()
    val = classification_metrics(y_true, y_score)["fpr_at_tpr95"]
    assert math.isnan(val) or 0.0 <= val <= 1.0


def test_classification_metrics_fpr_at_tpr95_nan_single_class() -> None:
    y_true, y_score = _all_same_class()
    assert math.isnan(classification_metrics(y_true, y_score)["fpr_at_tpr95"])


def test_classification_metrics_existing_keys_still_present() -> None:
    """Existing keys must still be returned alongside the new ones."""
    y_true, y_score = _imbalanced_arrays()
    m = classification_metrics(y_true, y_score)
    for key in ("auc", "f1", "precision", "recall", "fpr", "support", "botnet_rate"):
        assert key in m, f"Legacy key '{key}' missing"


# ---------------------------------------------------------------------------
# metrics_frame — new columns included
# ---------------------------------------------------------------------------

def test_metrics_frame_includes_auprc_column() -> None:
    y_true, y_score = _imbalanced_arrays()
    m = classification_metrics(y_true, y_score)
    m.update({"model": "hybrid", "scenario": "S1", "split": "test"})
    assert "auprc" in metrics_frame([m]).columns


def test_metrics_frame_includes_mcc_column() -> None:
    y_true, y_score = _imbalanced_arrays()
    m = classification_metrics(y_true, y_score)
    m.update({"model": "hybrid", "scenario": "S1", "split": "test"})
    assert "mcc" in metrics_frame([m]).columns


def test_metrics_frame_includes_fpr_at_tpr95_column() -> None:
    y_true, y_score = _imbalanced_arrays()
    m = classification_metrics(y_true, y_score)
    m.update({"model": "hybrid", "scenario": "S1", "split": "test"})
    assert "fpr_at_tpr95" in metrics_frame([m]).columns


# ---------------------------------------------------------------------------
# evaluation/plots.py
# ---------------------------------------------------------------------------

def test_confusion_matrix_plot_saves_file(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from gbd_ctu.evaluation.plots import confusion_matrix_plot
    y_true, y_score = _imbalanced_arrays(n_total=200)
    y_pred = (y_score >= 0.5).astype(int)
    out = confusion_matrix_plot(y_true, y_pred, scenario_id=1, output_dir=tmp_path)
    assert out.exists() and out.suffix == ".png"


def test_confusion_matrix_plot_filename_includes_scenario(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from gbd_ctu.evaluation.plots import confusion_matrix_plot
    out = confusion_matrix_plot(
        np.array([0, 1, 0, 1]), np.array([0, 1, 0, 0]),
        scenario_id=5, output_dir=tmp_path,
    )
    assert "5" in out.name


def test_roc_pr_curves_saves_files(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from gbd_ctu.evaluation.plots import roc_pr_curves
    y_true, y_score = _imbalanced_arrays(n_total=300)
    records = [
        {"model": "Hybrid", "y_true": y_true, "y_score": y_score, "scenario_id": 1},
        {"model": "XGBoost", "y_true": y_true, "y_score": y_score[::-1], "scenario_id": 2},
    ]
    out_dir = roc_pr_curves(records, output_dir=tmp_path)
    assert (out_dir / "roc_curves.png").exists()
    assert (out_dir / "pr_curves.png").exists()


def test_roc_pr_curves_skips_single_class(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from gbd_ctu.evaluation.plots import roc_pr_curves
    records = [{"model": "Hybrid", "y_true": np.zeros(10, dtype=int), "y_score": np.random.rand(10)}]
    roc_pr_curves(records, output_dir=tmp_path)  # must not raise


def test_scenario_comparison_heatmap_saves_file(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import pandas as pd
    from gbd_ctu.evaluation.plots import scenario_comparison_heatmap
    df = pd.DataFrame({
        "scenario_id": [1, 1, 2, 2],
        "model": ["Hybrid", "XGBoost", "Hybrid", "XGBoost"],
        "auc": [0.95, 0.88, 0.91, 0.82],
    })
    out = scenario_comparison_heatmap(df, metric="auc", output_dir=tmp_path)
    assert out.exists() and "auc" in out.name


def test_scenario_comparison_heatmap_raises_unknown_metric(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    import pandas as pd
    from gbd_ctu.evaluation.plots import scenario_comparison_heatmap
    df = pd.DataFrame({"scenario_id": [1], "model": ["Hybrid"], "auc": [0.9]})
    with pytest.raises(ValueError, match="unknown_metric"):
        scenario_comparison_heatmap(df, metric="unknown_metric", output_dir=tmp_path)


def test_palette_contains_expected_models() -> None:
    from gbd_ctu.evaluation.plots import PALETTE
    for key in ("GraphSAGE", "GAT", "Hybrid", "XGBoost", "RandomForest"):
        assert key in PALETTE, f"PALETTE missing key '{key}'"


# ---------------------------------------------------------------------------
# compare.py — new metric columns propagated
# ---------------------------------------------------------------------------

def test_compare_build_comparison_table_includes_auprc() -> None:
    import pandas as pd
    from gbd_ctu.evaluation.compare import build_comparison_table
    df = pd.DataFrame({
        "model": ["hybrid", "xgboost"],
        "scenario_id": [1, 1],
        "scenario": ["S1", "S1"],
        "auc": [0.95, 0.88],
        "auprc": [0.72, 0.65],
        "mcc": [0.60, 0.50],
        "f1": [0.80, 0.70],
        "fpr": [0.05, 0.08],
        "fpr_at_tpr95": [0.10, 0.15],
    })
    tables = build_comparison_table(df)
    assert "auprc" in tables
    assert "mcc" in tables
    assert "fpr_at_tpr95" in tables


def test_compare_fpr_at_tpr95_best_model_uses_min() -> None:
    """For fpr_at_tpr95, Best Model must be the one with the LOWEST value."""
    import pandas as pd
    from gbd_ctu.evaluation.compare import build_comparison_table
    df = pd.DataFrame({
        "model": ["hybrid", "xgboost"],
        "scenario_id": [1, 1],
        "scenario": ["S1", "S1"],
        "fpr_at_tpr95": [0.05, 0.15],
        "auc": [0.95, 0.88],
        "f1": [0.80, 0.75],
        "fpr": [0.03, 0.06],
    })
    tables = build_comparison_table(df)
    best = tables["fpr_at_tpr95"]["Best Model"].iloc[0]
    assert best == "Hybrid", f"Expected Hybrid (lower fpr_at_tpr95), got {best}"
