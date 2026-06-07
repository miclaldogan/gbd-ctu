"""Tests for GBD-CTU statistical significance utilities.

All tests run without torch / torch_geometric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbd_ctu.evaluation.stats import (
    bootstrap_ci,
    full_significance_table,
    wilcoxon_comparison,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MODELS = ["hybrid", "graphsage", "gat", "xgboost", "random_forest"]
_N_SCENARIOS = 13


def _make_results_df(
    seed: int = 0,
    hybrid_boost: float = 0.0,
) -> pd.DataFrame:
    """Synthetic long-format results with 5 models × 13 scenarios.

    *hybrid_boost* adds a fixed offset to every hybrid AUC so that
    the Wilcoxon test detects a clear difference when > 0.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for model in _MODELS:
        for sid in range(1, _N_SCENARIOS + 1):
            base_auc = float(rng.uniform(0.65, 0.90))
            if model == "hybrid":
                base_auc = min(base_auc + hybrid_boost, 1.0)
            rows.append({
                "model": model,
                "scenario_id": sid,
                "scenario": f"s{sid:02d}",
                "auc": base_auc,
                "auprc": float(rng.uniform(0.55, 0.88)),
                "mcc": float(rng.uniform(0.30, 0.80)),
                "f1": float(rng.uniform(0.55, 0.90)),
            })
    return pd.DataFrame(rows)


def _make_clear_df() -> pd.DataFrame:
    """DataFrame where hybrid clearly outperforms graphsage on AUC."""
    rows = []
    rng = np.random.default_rng(99)
    for sid in range(1, 14):
        rows.append({"model": "hybrid", "scenario_id": sid, "auc": 0.95, "auprc": 0.90, "mcc": 0.80})
        rows.append({"model": "graphsage", "scenario_id": sid, "auc": 0.60, "auprc": 0.55, "mcc": 0.40})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# wilcoxon_comparison
# ---------------------------------------------------------------------------

class TestWilcoxonComparison:
    def test_returns_dict(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc")
        assert isinstance(result, dict)

    def test_expected_keys(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc")
        required = {
            "model_a", "model_b", "metric", "statistic",
            "p_value", "p_bonferroni", "effect_size_d",
            "significant_at_0.05", "n_scenarios", "note",
        }
        assert required.issubset(result.keys())

    def test_model_names_preserved(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "xgboost", "auc")
        assert result["model_a"] == "hybrid"
        assert result["model_b"] == "xgboost"

    def test_metric_preserved(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "mcc")
        assert result["metric"] == "mcc"

    def test_n_scenarios_correct(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc")
        assert result["n_scenarios"] == _N_SCENARIOS

    def test_p_value_in_range(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc")
        p = result["p_value"]
        if not np.isnan(p):
            assert 0.0 <= p <= 1.0

    def test_p_bonferroni_geq_p_raw(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc", n_comparisons=15)
        if not np.isnan(result["p_value"]) and not np.isnan(result["p_bonferroni"]):
            assert result["p_bonferroni"] >= result["p_value"]

    def test_p_bonferroni_capped_at_1(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc", n_comparisons=1000)
        assert result["p_bonferroni"] <= 1.0

    def test_effect_size_is_float(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc")
        assert isinstance(result["effect_size_d"], float)

    def test_significant_is_bool(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc")
        assert isinstance(result["significant_at_0.05"], bool)

    def test_clear_difference_detected(self):
        """With a large performance gap, p_value should be < 0.05."""
        df = _make_clear_df()
        result = wilcoxon_comparison(df, "hybrid", "graphsage", "auc", n_comparisons=1)
        assert result["p_value"] < 0.05

    def test_all_zeros_diff_graceful(self):
        """When all differences are zero, test should not raise."""
        df = _make_results_df()
        # Make both models identical
        df2 = df[df["model"].isin(["hybrid", "graphsage"])].copy()
        df2.loc[df2["model"] == "graphsage", "auc"] = (
            df2.loc[df2["model"] == "hybrid", "auc"].values
        )
        result = wilcoxon_comparison(df2, "hybrid", "graphsage", "auc")
        assert isinstance(result, dict)

    def test_too_few_scenarios_graceful(self):
        """< 5 scenarios → returns NaN, does not raise."""
        df = _make_results_df()
        df_small = df[df["scenario_id"] <= 3].copy()
        result = wilcoxon_comparison(df_small, "hybrid", "graphsage", "auc")
        assert isinstance(result, dict)
        assert result["n_scenarios"] <= 3

    def test_case_insensitive_model_names(self):
        df = _make_results_df()
        result = wilcoxon_comparison(df, "Hybrid", "GraphSAGE", "auc")
        assert result["n_scenarios"] == _N_SCENARIOS


# ---------------------------------------------------------------------------
# full_significance_table
# ---------------------------------------------------------------------------

class TestFullSignificanceTable:
    def test_returns_dataframe(self):
        df = _make_results_df()
        table = full_significance_table(df, reference_model="hybrid")
        assert isinstance(table, pd.DataFrame)

    def test_expected_columns(self):
        df = _make_results_df()
        table = full_significance_table(df)
        expected = {
            "model_a", "model_b", "metric",
            "statistic", "p_raw", "p_bonferroni",
            "effect_size", "significant", "n_scenarios",
            "significant_marker",
        }
        assert expected.issubset(set(table.columns))

    def test_row_count(self):
        """4 other models × 3 metrics = 12 rows."""
        df = _make_results_df()
        table = full_significance_table(df, reference_model="hybrid",
                                        metrics=["auc", "auprc", "mcc"])
        assert len(table) == 4 * 3

    def test_reference_model_is_model_a(self):
        df = _make_results_df()
        table = full_significance_table(df, reference_model="hybrid")
        assert (table["model_a"] == "hybrid").all()

    def test_significant_marker_star_when_significant(self):
        df = _make_clear_df()
        table = full_significance_table(df, reference_model="hybrid",
                                        metrics=["auc"], n_comparisons=1)
        # hybrid vs graphsage on AUC should be significant
        row = table[(table["model_b"] == "graphsage") & (table["metric"] == "auc")]
        assert row["significant_marker"].values[0] == "*"

    def test_significant_marker_empty_when_not_significant(self):
        """When models differ only by noise, marker should be empty for some rows."""
        df = _make_results_df(hybrid_boost=0.0)  # no boost → no clear difference
        table = full_significance_table(df, reference_model="hybrid",
                                        metrics=["auc"], n_comparisons=15)
        markers = set(table["significant_marker"].unique())
        assert markers.issubset({"*", ""})

    def test_custom_metrics(self):
        df = _make_results_df()
        table = full_significance_table(df, metrics=["f1"])
        assert (table["metric"] == "f1").all()

    def test_missing_metric_skipped(self):
        df = _make_results_df().drop(columns=["mcc"])
        table = full_significance_table(df, metrics=["auc", "mcc"])
        assert "mcc" not in table["metric"].values

    def test_exportable_to_latex(self):
        df = _make_results_df()
        table = full_significance_table(df)
        tex = table.to_latex(index=False)
        assert "\\begin{tabular}" in tex

    def test_exportable_to_csv(self, tmp_path):
        df = _make_results_df()
        table = full_significance_table(df)
        csv_path = tmp_path / "sig.csv"
        table.to_csv(csv_path, index=False)
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == len(table)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_tuple(self):
        df = _make_results_df()
        ci = bootstrap_ci(df, "hybrid", "auc")
        assert isinstance(ci, tuple) and len(ci) == 2

    def test_lower_leq_upper(self):
        df = _make_results_df()
        lo, hi = bootstrap_ci(df, "hybrid", "auc")
        assert lo <= hi

    def test_ci_contains_sample_mean(self):
        """95 % CI should contain the sample mean (almost always)."""
        df = _make_results_df()
        scores = df[df["model"] == "hybrid"]["auc"].values
        sample_mean = float(scores.mean())
        lo, hi = bootstrap_ci(df, "hybrid", "auc", n_resamples=999)
        assert lo <= sample_mean <= hi

    def test_bounds_in_reasonable_range(self):
        df = _make_results_df()
        lo, hi = bootstrap_ci(df, "hybrid", "auc")
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_narrower_with_more_data(self):
        """More scenarios → tighter CI."""
        rng = np.random.default_rng(7)
        rows_small, rows_large = [], []
        for sid in range(1, 6):
            rows_small.append({"model": "hybrid", "scenario_id": sid,
                                "auc": float(rng.uniform(0.7, 0.9))})
        for sid in range(1, 51):
            rows_large.append({"model": "hybrid", "scenario_id": sid,
                                "auc": float(rng.uniform(0.7, 0.9))})
        df_small = pd.DataFrame(rows_small)
        df_large = pd.DataFrame(rows_large)
        lo_s, hi_s = bootstrap_ci(df_small, "hybrid", "auc", n_resamples=999)
        lo_l, hi_l = bootstrap_ci(df_large, "hybrid", "auc", n_resamples=999)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_reproducible_with_same_seed(self):
        df = _make_results_df()
        ci1 = bootstrap_ci(df, "hybrid", "auc", n_resamples=999, random_state=0)
        ci2 = bootstrap_ci(df, "hybrid", "auc", n_resamples=999, random_state=0)
        assert ci1 == ci2

    def test_too_few_samples_raises(self):
        df = pd.DataFrame([{"model": "hybrid", "scenario_id": 1, "auc": 0.8}])
        with pytest.raises(ValueError):
            bootstrap_ci(df, "hybrid", "auc")

    def test_case_insensitive(self):
        df = _make_results_df()
        ci = bootstrap_ci(df, "Hybrid", "auc")
        assert isinstance(ci, tuple)

    def test_mcc_metric(self):
        df = _make_results_df()
        lo, hi = bootstrap_ci(df, "graphsage", "mcc")
        assert lo <= hi
