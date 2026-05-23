"""GBD-CTU evaluation package.

This package contains metric computation, scenario-wise reporting, and report
comparison utilities for CTU-13 experiments.
"""

from gbd_ctu.evaluation.compare import compare_reports
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
from gbd_ctu.evaluation.scenario_eval import evaluate_gnn_checkpoint

__all__ = ["classification_metrics", "compare_reports", "evaluate_gnn_checkpoint", "metrics_frame"]
