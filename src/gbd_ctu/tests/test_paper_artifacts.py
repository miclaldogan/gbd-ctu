"""Tests for paper artifact export utilities.

Groups
------
- ``TestBoldBest``: pure-Python table formatting; no matplotlib required.
- ``TestLatexTables``: LaTeX table export; no matplotlib required.
- ``TestArchitectureDot``: .dot file generation; no external tools required.
- ``TestFigures``: matplotlib-based figure generation; skipped if absent.
- ``TestCLI``: smoke tests for the export_paper CLI entry-point.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_mpl_available = importlib.util.find_spec("matplotlib") is not None
_sklearn_available = importlib.util.find_spec("sklearn") is not None

_skip_mpl = pytest.mark.skipif(not _mpl_available, reason="matplotlib not installed")
_skip_mpl_sklearn = pytest.mark.skipif(
    not (_mpl_available and _sklearn_available),
    reason="matplotlib or sklearn not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics_df(n_models: int = 3, n_scenarios: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = ["hybrid", "graphsage", "gat"][:n_models]
    rows = []
    for sid in range(1, n_scenarios + 1):
        for model in models:
            rows.append({
                "model": model,
                "scenario": f"scenario_{sid:02d}",
                "scenario_id": sid,
                "auc": float(rng.uniform(0.6, 1.0)),
                "f1": float(rng.uniform(0.5, 1.0)),
                "mcc": float(rng.uniform(0.4, 1.0)),
                "auprc": float(rng.uniform(0.5, 1.0)),
                "fpr": float(rng.uniform(0.0, 0.3)),
            })
    return pd.DataFrame(rows)


def _make_ablation_df(n_variants: int = 4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = ["baseline", "no_edge_attr", "gat_only", "hybrid"][:n_variants]
    rows = []
    for name in names:
        rows.append({
            "name": name,
            "model_type": "hybrid",
            "auc": float(rng.uniform(0.7, 1.0)),
            "f1": float(rng.uniform(0.6, 1.0)),
            "mcc": float(rng.uniform(0.5, 1.0)),
            "auprc": float(rng.uniform(0.6, 1.0)),
        })
    return pd.DataFrame(rows)


def _make_sig_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"model_a": "hybrid", "model_b": "graphsage", "metric": "auc",
         "statistic": 12.0, "p_value": 0.03, "p_bonferroni": 0.045,
         "effect_size_d": 0.8, "n_scenarios": 13},
        {"model_a": "hybrid", "model_b": "gat", "metric": "auc",
         "statistic": 8.0, "p_value": 0.008, "p_bonferroni": 0.012,
         "effect_size_d": 1.1, "n_scenarios": 13},
    ])


def _make_records(n: int = 20, seed: int = 2) -> list[dict]:
    rng = np.random.default_rng(seed)
    records = []
    for sid in range(1, 4):
        for model in ("hybrid", "graphsage"):
            y_true = rng.integers(0, 2, n).tolist()
            y_score = rng.uniform(0, 1, n).tolist()
            records.append({
                "model": model,
                "scenario_id": sid,
                "y_true": y_true,
                "y_score": y_score,
            })
    return records


# ---------------------------------------------------------------------------
# TestBoldBest
# ---------------------------------------------------------------------------


class TestBoldBest:
    def test_max_column_bolded(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [0.8, 0.9, 0.7]})
        out = bold_best(df, ["auc"])
        assert r"\textbf" in out.loc[1, "auc"]
        assert r"\textbf" not in out.loc[0, "auc"]
        assert r"\textbf" not in out.loc[2, "auc"]

    def test_min_column_bolded_for_lower_better(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"fpr": [0.2, 0.05, 0.15]})
        out = bold_best(df, ["fpr"], lower_better=["fpr"])
        assert r"\textbf" in out.loc[1, "fpr"]
        assert r"\textbf" not in out.loc[0, "fpr"]

    def test_nan_formatted_as_dash(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [0.8, float("nan"), 0.7]})
        out = bold_best(df, ["auc"])
        assert out.loc[1, "auc"] == "—"

    def test_non_metric_columns_unchanged(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"model": ["a", "b"], "auc": [0.8, 0.9]})
        out = bold_best(df, ["auc"])
        assert list(out["model"]) == ["a", "b"]

    def test_missing_column_skipped(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [0.8, 0.9]})
        out = bold_best(df, ["auc", "nonexistent"])
        assert "nonexistent" not in out.columns

    def test_all_nan_column_not_crash(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [float("nan"), float("nan")]})
        out = bold_best(df, ["auc"])
        assert len(out) == 2

    def test_output_values_are_strings(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [0.7, 0.9, 0.8]})
        out = bold_best(df, ["auc"])
        assert all(isinstance(v, str) for v in out["auc"])

    def test_formatted_to_four_decimal_places(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [0.8, 0.9]})
        out = bold_best(df, ["auc"])
        plain = out.loc[0, "auc"]
        assert "." in plain
        decimal_part = plain.split(".")[-1]
        assert len(decimal_part) == 4

    def test_lower_better_default_includes_fpr(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best, LOWER_BETTER
        assert "fpr" in LOWER_BETTER

    def test_multiple_metric_cols(self):
        from gbd_ctu.evaluation.paper_artifacts import bold_best
        df = pd.DataFrame({"auc": [0.8, 0.9], "f1": [0.75, 0.70]})
        out = bold_best(df, ["auc", "f1"])
        assert r"\textbf" in out.loc[1, "auc"]
        assert r"\textbf" in out.loc[0, "f1"]


# ---------------------------------------------------------------------------
# TestLatexTables
# ---------------------------------------------------------------------------


class TestExportResultsTable:
    def test_writes_file(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        df = _make_metrics_df()
        tex = export_results_table(df, tmp_path / "r.tex")
        assert (tmp_path / "r.tex").exists()

    def test_returns_string(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        df = _make_metrics_df()
        tex = export_results_table(df, tmp_path / "r.tex")
        assert isinstance(tex, str)

    def test_has_toprule(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        tex = export_results_table(_make_metrics_df(), tmp_path / "r.tex")
        assert r"\toprule" in tex

    def test_has_midrule(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        tex = export_results_table(_make_metrics_df(), tmp_path / "r.tex")
        assert r"\midrule" in tex

    def test_has_bottomrule(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        tex = export_results_table(_make_metrics_df(), tmp_path / "r.tex")
        assert r"\bottomrule" in tex

    def test_no_hline(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        tex = export_results_table(_make_metrics_df(), tmp_path / "r.tex")
        assert r"\hline" not in tex

    def test_textbf_appears(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        tex = export_results_table(_make_metrics_df(), tmp_path / "r.tex")
        assert r"\textbf" in tex

    def test_custom_metrics_subset(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_results_table
        tex = export_results_table(_make_metrics_df(), tmp_path / "r.tex", metrics=["auc"])
        assert "AUC" in tex


class TestExportAblationTable:
    def test_writes_file(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_ablation_table
        df = _make_ablation_df()
        export_ablation_table(df, tmp_path / "abl.tex")
        assert (tmp_path / "abl.tex").exists()

    def test_has_booktabs_rules(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_ablation_table
        tex = export_ablation_table(_make_ablation_df(), tmp_path / "abl.tex")
        assert r"\toprule" in tex and r"\bottomrule" in tex

    def test_has_variant_column(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_ablation_table
        tex = export_ablation_table(_make_ablation_df(), tmp_path / "abl.tex")
        assert "Variant" in tex


class TestExportSignificanceTable:
    def test_writes_file(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_significance_table
        tex = export_significance_table(_make_sig_df(), tmp_path / "sig.tex")
        assert (tmp_path / "sig.tex").exists()

    def test_double_star_for_small_p(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_significance_table
        tex = export_significance_table(_make_sig_df(), tmp_path / "sig.tex")
        assert "**" in tex

    def test_single_star_for_moderate_p(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_significance_table
        tex = export_significance_table(_make_sig_df(), tmp_path / "sig.tex")
        assert "*" in tex

    def test_has_booktabs_rules(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_significance_table
        tex = export_significance_table(_make_sig_df(), tmp_path / "sig.tex")
        assert r"\toprule" in tex

    def test_ns_for_large_p(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_significance_table
        df = pd.DataFrame([{
            "model_a": "a", "model_b": "b", "metric": "auc",
            "statistic": 5.0, "p_value": 0.4, "p_bonferroni": 0.6,
            "effect_size_d": 0.1, "n_scenarios": 10,
        }])
        tex = export_significance_table(df, tmp_path / "sig.tex")
        assert "ns" in tex


class TestExportDatasetTable:
    def test_writes_file(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import export_dataset_table
        df = pd.DataFrame({
            "scenario_id": [1, 2],
            "family": ["Neris", "Rbot"],
            "total_flows": [1000, 2000],
            "botnet_flows": [100, 200],
            "botnet_frac": [0.1, 0.1],
            "unique_src_ips": [50, 80],
            "botnet_host_ips": [5, 8],
            "unique_dst_ports": [20, 30],
            "mean_duration": [1.5, 2.0],
            "max_duration": [60.0, 120.0],
        })
        tex = export_dataset_table(df, tmp_path / "ds.tex")
        assert (tmp_path / "ds.tex").exists()
        assert r"\toprule" in tex


# ---------------------------------------------------------------------------
# TestArchitectureDot
# ---------------------------------------------------------------------------


class TestDrawModelArchitecture:
    @pytest.mark.parametrize("model_type", ["hybrid", "gat", "graphsage", "mlp"])
    def test_writes_dot_file(self, tmp_path, model_type):
        from gbd_ctu.evaluation.paper_artifacts import draw_model_architecture
        path = draw_model_architecture(model_type, tmp_path / f"arch_{model_type}.dot")
        assert path.exists()

    @pytest.mark.parametrize("model_type", ["hybrid", "gat", "graphsage", "mlp"])
    def test_dot_content_valid_digraph(self, tmp_path, model_type):
        from gbd_ctu.evaluation.paper_artifacts import draw_model_architecture
        path = draw_model_architecture(model_type, tmp_path / "arch.dot")
        content = path.read_text()
        assert "digraph" in content
        assert "->" in content

    def test_unknown_model_type_raises(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import draw_model_architecture
        with pytest.raises(ValueError, match="Unknown"):
            draw_model_architecture("unknown_xyz", tmp_path / "x.dot")

    def test_returns_path(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import draw_model_architecture
        result = draw_model_architecture("hybrid", tmp_path / "arch.dot")
        assert isinstance(result, Path)

    def test_creates_parent_dirs(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import draw_model_architecture
        deep = tmp_path / "a" / "b" / "c" / "arch.dot"
        draw_model_architecture("gat", deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# TestFigures
# ---------------------------------------------------------------------------


@_skip_mpl
class TestScenarioHeatmapFigure:
    def test_creates_png(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import scenario_heatmap_figure
        df = _make_metrics_df()
        out = scenario_heatmap_figure(df, metric="auc", output_dir=tmp_path)
        assert out.exists()

    def test_output_named_correctly(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import scenario_heatmap_figure
        out = scenario_heatmap_figure(_make_metrics_df(), metric="f1", output_dir=tmp_path)
        assert out.name == "heatmap_f1.png"

    def test_missing_metric_raises(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import scenario_heatmap_figure
        with pytest.raises(ValueError, match="not found"):
            scenario_heatmap_figure(_make_metrics_df(), metric="nonexistent", output_dir=tmp_path)


@_skip_mpl
class TestAblationBarFigure:
    def test_creates_png(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import ablation_bar_figure
        out = ablation_bar_figure(_make_ablation_df(), output_dir=tmp_path)
        assert out.exists()

    def test_output_named_correctly(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import ablation_bar_figure
        out = ablation_bar_figure(_make_ablation_df(), output_dir=tmp_path)
        assert out.name == "ablation_bar.png"

    def test_no_metric_cols_raises(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import ablation_bar_figure
        df = pd.DataFrame({"name": ["a"]})
        with pytest.raises(ValueError, match="No metric"):
            ablation_bar_figure(df, metrics=["nonexistent"], output_dir=tmp_path)


@_skip_mpl_sklearn
class TestPerScenarioRocPrFigure:
    def test_creates_two_pngs(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import per_scenario_roc_pr_figure
        records = _make_records()
        roc_p, pr_p = per_scenario_roc_pr_figure(records, output_dir=tmp_path)
        assert roc_p.exists()
        assert pr_p.exists()

    def test_roc_path_named_correctly(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import per_scenario_roc_pr_figure
        roc_p, _ = per_scenario_roc_pr_figure(_make_records(), output_dir=tmp_path)
        assert roc_p.name == "per_scenario_roc.png"

    def test_pr_path_named_correctly(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import per_scenario_roc_pr_figure
        _, pr_p = per_scenario_roc_pr_figure(_make_records(), output_dir=tmp_path)
        assert pr_p.name == "per_scenario_pr.png"

    def test_empty_records_raises(self, tmp_path):
        from gbd_ctu.evaluation.paper_artifacts import per_scenario_roc_pr_figure
        with pytest.raises((ValueError, Exception)):
            per_scenario_roc_pr_figure([], output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Constants / style
# ---------------------------------------------------------------------------


class TestConstants:
    def test_palette_has_required_models(self):
        from gbd_ctu.evaluation.paper_artifacts import PALETTE
        for key in ("hybrid", "graphsage", "gat", "xgboost", "random_forest", "mlp"):
            assert key in PALETTE, f"PALETTE missing key {key!r}"

    def test_palette_values_are_hex(self):
        from gbd_ctu.evaluation.paper_artifacts import PALETTE
        for key, colour in PALETTE.items():
            assert colour.startswith("#"), f"PALETTE[{key!r}] not hex: {colour!r}"

    def test_lower_better_includes_fpr(self):
        from gbd_ctu.evaluation.paper_artifacts import LOWER_BETTER
        assert "fpr" in LOWER_BETTER

    def test_style_rc_has_font_size(self):
        from gbd_ctu.evaluation.paper_artifacts import STYLE_RC
        assert "font.size" in STYLE_RC

    def test_style_rc_has_dpi(self):
        from gbd_ctu.evaluation.paper_artifacts import STYLE_RC
        assert STYLE_RC.get("savefig.dpi") == 300


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def _env_with_src():
    import os
    env = os.environ.copy()
    src_root = str(Path(__file__).parents[3] / "src")
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


class TestCLI:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.export_paper", "--help"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode == 0

    def test_help_shows_key_flags(self):
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.export_paper", "--help"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        for flag in ("--results-dir", "--output-dir", "--skip-figures", "--skip-tables"):
            assert flag in result.stdout

    def test_missing_results_dir_exits_nonzero(self, tmp_path):
        missing = tmp_path / "no_such"
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.export_paper",
             "--results-dir", str(missing)],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode != 0

    def test_empty_results_dir_exits_zero(self, tmp_path):
        """CLI exits 0 even with no CSVs (warns and writes dot files only)."""
        result = subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.export_paper",
             "--results-dir", str(tmp_path),
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True, env=_env_with_src(),
        )
        assert result.returncode == 0

    def test_dot_files_written_even_without_csvs(self, tmp_path):
        out_dir = tmp_path / "out"
        subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.export_paper",
             "--results-dir", str(tmp_path),
             "--output-dir", str(out_dir),
             "--skip-figures"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        dot_files = list(out_dir.glob("architecture_*.dot"))
        assert len(dot_files) == 4

    def test_skip_figures_flag(self, tmp_path):
        out_dir = tmp_path / "out"
        subprocess.run(
            [sys.executable, "-m", "gbd_ctu.scripts.export_paper",
             "--results-dir", str(tmp_path),
             "--output-dir", str(out_dir),
             "--skip-figures"],
            capture_output=True, text=True, env=_env_with_src(),
        )
        png_files = list(out_dir.glob("*.png"))
        assert len(png_files) == 0
