"""GBD-CTU metric tests.

These tests validate metric computation for binary classification outputs used in
CTU-13 scenario evaluation.
"""

from __future__ import annotations

from gbd_ctu.evaluation.metrics import classification_metrics


def test_classification_metrics_include_fpr_and_auc() -> None:
    """Metric output should include the requested evaluation fields."""

    report = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.4])
    assert report["auc"] > 0.9
    assert report["f1"] > 0.7
    assert 0.0 <= report["fpr"] <= 1.0
