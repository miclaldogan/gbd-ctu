"""GBD-CTU statistical significance tests.

Public API
----------
``wilcoxon_comparison``
    Paired Wilcoxon signed-rank test for one model pair × one metric,
    with Bonferroni correction and Cohen's d effect size.

``full_significance_table``
    All comparisons of a reference model vs every other model across
    multiple metrics; returns a tidy DataFrame with LaTeX-ready significance
    markers.

``bootstrap_ci``
    Bootstrap 95 % CI (BCa method) on per-scenario metric values for a
    single model.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

# Default number of Bonferroni comparisons (5 pairs × 3 metrics)
_DEFAULT_N_COMPARISONS = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scores(
    results_df: pd.DataFrame,
    model: str,
    metric: str,
) -> np.ndarray:
    """Extract per-scenario metric values for *model*, sorted by scenario_id."""
    col_model = "model"
    df = results_df.copy()
    df[col_model] = df[col_model].str.strip().str.lower()
    sub = df[df[col_model] == model.strip().lower()]
    if "scenario_id" in sub.columns:
        sub = sub.sort_values("scenario_id")
    return sub[metric].to_numpy(dtype=float)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean_a - mean_b) / pooled_std."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wilcoxon_comparison(
    results_df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "auc",
    n_comparisons: int = _DEFAULT_N_COMPARISONS,
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test: model_a vs model_b on *metric*.

    Parameters
    ----------
    results_df:
        Long-format DataFrame with columns ``model``, ``scenario_id``, and
        one column per metric.
    model_a:
        Name of the first model (case-insensitive, stripped).
    model_b:
        Name of the second (baseline) model.
    metric:
        Column name of the metric to test (default ``"auc"``).
    n_comparisons:
        Number of simultaneous comparisons for Bonferroni correction
        (default 15 = 5 pairs × 3 metrics).

    Returns
    -------
    dict with keys:
        ``model_a``, ``model_b``, ``metric``,
        ``statistic``, ``p_value``, ``p_bonferroni``,
        ``effect_size_d``, ``significant_at_0.05``,
        ``n_scenarios``, ``note``.
    """
    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError as exc:
        raise ImportError("scipy is required for wilcoxon_comparison.") from exc

    scores_a = _get_scores(results_df, model_a, metric)
    scores_b = _get_scores(results_df, model_b, metric)

    # Align on paired scenarios: keep only positions valid in both
    min_len = min(len(scores_a), len(scores_b))
    if min_len < 5:
        note = (
            f"Only {min_len} paired scenarios available (need ≥ 5 for reliable Wilcoxon test)."
        )
        _logger.warning(note)
        return {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "p_bonferroni": float("nan"),
            "effect_size_d": _cohens_d(scores_a, scores_b),
            "significant_at_0.05": False,
            "n_scenarios": min_len,
            "note": note,
        }

    a = scores_a[:min_len]
    b = scores_b[:min_len]
    diff = a - b

    # Skip if all differences are zero (Wilcoxon undefined)
    if np.all(diff == 0):
        note = "All paired differences are zero; Wilcoxon test undefined."
        return {
            "model_a": model_a,
            "model_b": model_b,
            "metric": metric,
            "statistic": float("nan"),
            "p_value": float("nan"),
            "p_bonferroni": float("nan"),
            "effect_size_d": 0.0,
            "significant_at_0.05": False,
            "n_scenarios": min_len,
            "note": note,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            stat, p_val = _wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        except Exception as exc:
            stat, p_val = float("nan"), float("nan")
            _logger.warning("Wilcoxon test failed: %s", exc)

    p_bonf = float(min(float(p_val) * n_comparisons, 1.0)) if not np.isnan(p_val) else float("nan")
    significant = bool(p_bonf < 0.05) if not np.isnan(p_bonf) else False

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "statistic": float(stat),
        "p_value": float(p_val),
        "p_bonferroni": p_bonf,
        "effect_size_d": _cohens_d(a, b),
        "significant_at_0.05": significant,
        "n_scenarios": min_len,
        "note": "Wilcoxon signed-rank (two-sided), Bonferroni-corrected.",
    }


def full_significance_table(
    results_df: pd.DataFrame,
    reference_model: str = "hybrid",
    metrics: list[str] | None = None,
    n_comparisons: int | None = None,
) -> pd.DataFrame:
    """All comparisons: reference_model vs each other model, across metrics.

    Parameters
    ----------
    results_df:
        Long-format DataFrame with ``model``, ``scenario_id``, and metric
        columns.
    reference_model:
        The model to use as model_a in every comparison (default
        ``"hybrid"``).
    metrics:
        Metric columns to test.  Defaults to ``["auc", "auprc", "mcc"]``
        (columns must exist in *results_df*).
    n_comparisons:
        Override Bonferroni denominator.  Defaults to
        ``len(other_models) × len(metrics)``.

    Returns
    -------
    pd.DataFrame with columns:
        ``model_a``, ``model_b``, ``metric``, ``statistic``,
        ``p_raw``, ``p_bonferroni``, ``effect_size``, ``significant``,
        ``n_scenarios``.

    A ``"significant_marker"`` column appends ``"*"`` when
    ``p_bonferroni < 0.05``.
    """
    if metrics is None:
        metrics = [m for m in ("auc", "auprc", "mcc") if m in results_df.columns]
    if not metrics:
        raise ValueError("No recognised metric columns found in results_df.")

    models_in_df = (
        results_df["model"].str.strip().str.lower().unique().tolist()
    )
    ref = reference_model.strip().lower()
    other_models = [m for m in models_in_df if m != ref]

    if n_comparisons is None:
        n_comparisons = max(len(other_models) * len(metrics), 1)

    rows = []
    for metric in metrics:
        if metric not in results_df.columns:
            _logger.warning("Metric '%s' not in results_df; skipping.", metric)
            continue
        for other in other_models:
            result = wilcoxon_comparison(
                results_df,
                model_a=ref,
                model_b=other,
                metric=metric,
                n_comparisons=n_comparisons,
            )
            rows.append({
                "model_a": result["model_a"],
                "model_b": result["model_b"],
                "metric": result["metric"],
                "statistic": result["statistic"],
                "p_raw": result["p_value"],
                "p_bonferroni": result["p_bonferroni"],
                "effect_size": result["effect_size_d"],
                "significant": result["significant_at_0.05"],
                "n_scenarios": result["n_scenarios"],
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["significant_marker"] = df["significant"].apply(lambda s: "*" if s else "")
    return df


def bootstrap_ci(
    results_df: pd.DataFrame,
    model: str,
    metric: str,
    n_resamples: int = 9999,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval (BCa) for per-scenario metric values.

    Parameters
    ----------
    results_df:
        Long-format DataFrame with ``model`` and metric columns.
    model:
        Model name to filter on (case-insensitive).
    metric:
        Metric column name.
    n_resamples:
        Number of bootstrap resamples (default 9999).
    confidence_level:
        Coverage probability (default 0.95 → 95 % CI).
    random_state:
        Seed for reproducibility.

    Returns
    -------
    ``(lower, upper)`` — the BCa bootstrap confidence interval bounds.

    Raises
    ------
    ValueError
        When fewer than 2 observations are available.
    """
    try:
        from scipy.stats import bootstrap as _bootstrap
    except ImportError as exc:
        raise ImportError("scipy >= 1.7 is required for bootstrap_ci.") from exc

    scores = _get_scores(results_df, model, metric)
    scores = scores[~np.isnan(scores)]

    if len(scores) < 2:
        raise ValueError(
            f"Need at least 2 valid scores for bootstrap CI; got {len(scores)} "
            f"for model='{model}', metric='{metric}'."
        )

    rng = np.random.default_rng(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _bootstrap(
            (scores,),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            method="BCa",
            random_state=rng,
        )

    return float(result.confidence_interval.low), float(result.confidence_interval.high)
