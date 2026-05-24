"""Tests for GBD-CTU comparison report and figures.

Covers:
  - build_comparison_table: wide pivot, Best Model column, metric tables.
  - wilcoxon_hybrid_vs_baseline: returns expected keys, handles edge cases.
  - generate_figures: PNG/PDF files created for synthetic data.
  - run_comparison: end-to-end with temp CSVs.
  - compare_reports: legacy backward-compat wrapper.
  - scripts/compare_models.py CLI: --help, --results-dir with synthetic data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbd_ctu.evaluation.compare import (
    build_comparison_table,
    compare_reports,
    generate_figures,
    run_comparison,
    wilcoxon_hybrid_vs_baseline,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_MODELS = ["graphsage", "gat", "hybrid", "xgboost", "random_forest"]
_SCENARIOS = list(range(1, 6))  # scenarios 1-5


def _make_results_df(seed: int = 0) -> pd.DataFrame:
    """Synthetic long-format results DataFrame with 5 models × 5 scenarios."""
    rng = np.random.default_rng(seed)
    rows = []
    for model in _MODELS:
        for sid in _SCENARIOS:
            rows.append({
                "model": model,
                "scenario_id": sid,
                "scenario": f"scenario-{sid:02d}",
                "auc": float(rng.uniform(0.6, 0.99)),
                "f1": float(rng.uniform(0.5, 0.95)),
                "fpr": float(rng.uniform(0.01, 0.2)),
                "precision": float(rng.uniform(0.5, 0.95)),
                "recall": float(rng.uniform(0.5, 0.95)),
                "n_botnet_nodes": int(rng.integers(10, 100)),
                "n_total_nodes": int(rng.integers(200, 1000)),
            })
    return pd.DataFrame(rows)


def _write_results_csvs(tmp_path: Path, df: pd.DataFrame) -> None:
    """Split df by model and write results_{model}.csv files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()
        sub.to_csv(tmp_path / f"results_{model}.csv", index=False)


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_returns_dict_of_dataframes(self):
        df = _make_results_df()
        tables = build_comparison_table(df)
        assert isinstance(tables, dict)
        for v in tables.values():
            assert isinstance(v, pd.DataFrame)

    def test_keys_are_known_metrics(self):
        df = _make_results_df()
        tables = build_comparison_table(df)
        assert "auc" in tables
        assert "f1" in tables
        assert "fpr" in tables

    def test_auc_table_has_one_row_per_scenario(self):
        df = _make_results_df()
        tables = build_comparison_table(df)
        assert len(tables["auc"]) == len(_SCENARIOS)

    def test_auc_table_has_best_model_column(self):
        df = _make_results_df()
        tables = build_comparison_table(df)
        assert "Best Model" in tables["auc"].columns

    def test_best_model_is_never_empty(self):
        df = _make_results_df()
        tables = build_comparison_table(df)
        assert tables["auc"]["Best Model"].notna().all()

    def test_sorted_by_scenario_id(self):
        df = _make_results_df()
        tables = build_comparison_table(df)
        ids = tables["auc"]["scenario_id"].tolist()
        assert ids == sorted(ids)

    def test_fpr_best_model_is_lowest(self):
        """For FPR, best model should be the one with minimum (not maximum) value."""
        df = _make_results_df(seed=7)
        tables = build_comparison_table(df)
        fpr_table = tables["fpr"]
        model_cols = [c for c in fpr_table.columns
                      if c not in ("scenario_id", "scenario", "Best Model")]
        for _, row in fpr_table.iterrows():
            best = row["Best Model"]
            assert row[best] == row[model_cols].min()

    def test_missing_metric_column_skipped(self):
        df = _make_results_df().drop(columns=["fpr"])
        tables = build_comparison_table(df)
        assert "fpr" not in tables


# ---------------------------------------------------------------------------
# wilcoxon_hybrid_vs_baseline
# ---------------------------------------------------------------------------

class TestWilcoxonHybridVsBaseline:
    def test_returns_expected_keys(self):
        df = _make_results_df()
        result = wilcoxon_hybrid_vs_baseline(df)
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "n_scenarios" in result
        assert "note" in result

    def test_n_scenarios_correct(self):
        df = _make_results_df()
        result = wilcoxon_hybrid_vs_baseline(df)
        assert result["n_scenarios"] == len(_SCENARIOS)

    def test_p_value_in_range(self):
        df = _make_results_df()
        result = wilcoxon_hybrid_vs_baseline(df)
        p = result["p_value"]
        if not (p != p):  # not NaN
            assert 0.0 <= p <= 1.0

    def test_significant_is_bool(self):
        df = _make_results_df()
        result = wilcoxon_hybrid_vs_baseline(df)
        if result["significant"] is not None:
            assert isinstance(result["significant"], bool)

    def test_no_hybrid_data_returns_nan(self):
        df = _make_results_df()
        df_no_hybrid = df[df["model"] != "hybrid"].copy()
        result = wilcoxon_hybrid_vs_baseline(df_no_hybrid)
        assert result["n_scenarios"] == 0

    def test_single_scenario_graceful(self):
        """With only 1 paired scenario, Wilcoxon can't be computed."""
        df = _make_results_df()
        df_1 = df[df["scenario_id"] == 1].copy()
        result = wilcoxon_hybrid_vs_baseline(df_1)
        # Either nan (can't compute) or valid — must not raise
        assert "p_value" in result


# ---------------------------------------------------------------------------
# generate_figures
# ---------------------------------------------------------------------------

class TestGenerateFigures:
    def test_png_and_pdf_created(self, tmp_path):
        df = _make_results_df()
        figures_dir = tmp_path / "figures"
        saved = generate_figures(df, figures_dir)
        assert len(saved) > 0
        for p in saved:
            assert Path(p).exists()

    def test_auc_bar_chart_saved(self, tmp_path):
        df = _make_results_df()
        figures_dir = tmp_path / "figures"
        generate_figures(df, figures_dir)
        assert (figures_dir / "auc_by_scenario.png").exists()
        assert (figures_dir / "auc_by_scenario.pdf").exists()

    def test_f1_bar_chart_saved(self, tmp_path):
        df = _make_results_df()
        figures_dir = tmp_path / "figures"
        generate_figures(df, figures_dir)
        assert (figures_dir / "f1_by_scenario.png").exists()

    def test_roc_placeholder_saved(self, tmp_path):
        df = _make_results_df()
        figures_dir = tmp_path / "figures"
        generate_figures(df, figures_dir)
        assert (figures_dir / "roc_curves.png").exists()
        assert (figures_dir / "roc_curves.pdf").exists()

    def test_figures_dir_created_if_missing(self, tmp_path):
        df = _make_results_df()
        figures_dir = tmp_path / "a" / "b" / "figures"
        assert not figures_dir.exists()
        generate_figures(df, figures_dir)
        assert figures_dir.exists()


# ---------------------------------------------------------------------------
# run_comparison
# ---------------------------------------------------------------------------

class TestRunComparison:
    def test_returns_expected_keys(self, tmp_path):
        df = _make_results_df()
        _write_results_csvs(tmp_path / "results", df)
        result = run_comparison(tmp_path / "results", tmp_path / "out")
        assert "tables" in result
        assert "wilcoxon" in result
        assert "figures" in result
        assert "csv_paths" in result
        assert "combined_csv" in result
        assert "markdown_report" in result

    def test_csv_files_written(self, tmp_path):
        df = _make_results_df()
        _write_results_csvs(tmp_path / "results", df)
        result = run_comparison(tmp_path / "results", tmp_path / "out")
        for metric, csv_path in result["csv_paths"].items():
            assert Path(csv_path).exists(), f"Missing CSV for {metric}"

    def test_combined_csv_written(self, tmp_path):
        df = _make_results_df()
        _write_results_csvs(tmp_path / "results", df)
        result = run_comparison(tmp_path / "results", tmp_path / "out")
        assert Path(result["combined_csv"]).exists()

    def test_markdown_report_written(self, tmp_path):
        df = _make_results_df()
        _write_results_csvs(tmp_path / "results", df)
        result = run_comparison(tmp_path / "results", tmp_path / "out")
        md_path = Path(result["markdown_report"])
        assert md_path.exists()
        content = md_path.read_text()
        assert "AUC" in content

    def test_figures_generated(self, tmp_path):
        df = _make_results_df()
        _write_results_csvs(tmp_path / "results", df)
        result = run_comparison(tmp_path / "results", tmp_path / "out")
        assert len(result["figures"]) > 0
        for fp in result["figures"]:
            assert Path(fp).exists()

    def test_no_results_csvs_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            run_comparison(empty_dir, tmp_path / "out")


# ---------------------------------------------------------------------------
# compare_reports (legacy wrapper)
# ---------------------------------------------------------------------------

class TestCompareReports:
    def test_merges_two_csvs(self, tmp_path):
        df = _make_results_df()
        gnn = df[df["model"].isin(["graphsage", "gat", "hybrid"])].copy()
        baseline = df[df["model"].isin(["xgboost", "random_forest"])].copy()
        gnn_path = tmp_path / "gnn.csv"
        baseline_path = tmp_path / "baseline.csv"
        gnn.to_csv(gnn_path, index=False)
        baseline.to_csv(baseline_path, index=False)

        output_path = tmp_path / "combined.csv"
        result = compare_reports(gnn_path, baseline_path, output_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert output_path.exists()

    def test_json_companion_written(self, tmp_path):
        df = _make_results_df()
        gnn = df[df["model"] == "graphsage"].copy()
        baseline = df[df["model"] == "xgboost"].copy()
        gnn.to_csv(tmp_path / "gnn.csv", index=False)
        baseline.to_csv(tmp_path / "baseline.csv", index=False)
        compare_reports(tmp_path / "gnn.csv", tmp_path / "baseline.csv",
                        tmp_path / "out.csv")
        assert (tmp_path / "out.json").exists()


# ---------------------------------------------------------------------------
# CLI: scripts/compare_models.py
# ---------------------------------------------------------------------------

class TestCompareModelsCLI:
    def _env(self):
        import os
        src = str(Path(__file__).parent.parent.parent)
        return {**os.environ, "PYTHONPATH": src}

    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.compare_models", "--help"],
            capture_output=True, text=True, env=self._env(),
        )
        assert result.returncode == 0
        assert "--results-dir" in result.stdout

    def test_missing_results_dir_exits_nonzero(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.compare_models",
             "--results-dir", str(empty)],
            capture_output=True, text=True, env=self._env(),
        )
        assert result.returncode != 0

    def test_runs_with_synthetic_csvs(self, tmp_path):
        df = _make_results_df()
        results_dir = tmp_path / "results"
        _write_results_csvs(results_dir, df)
        output_dir = tmp_path / "out"
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.compare_models",
             "--results-dir", str(results_dir),
             "--output-dir", str(output_dir)],
            capture_output=True, text=True, env=self._env(),
        )
        assert result.returncode == 0
        assert "AUC" in result.stdout
