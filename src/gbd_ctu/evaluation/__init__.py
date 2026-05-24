"""GBD-CTU evaluation package.

This package contains metric computation, scenario-wise reporting, and report
comparison utilities for CTU-13 experiments.
"""

from gbd_ctu.evaluation.compare import (
    build_comparison_table,
    compare_reports,
    generate_figures,
    run_comparison,
    wilcoxon_hybrid_vs_baseline,
)
from gbd_ctu.evaluation.metrics import (
    auc_roc,
    classification_metrics,
    confusion_matrix_plot,
    f1_botnet,
    false_positive_rate,
    metrics_frame,
    precision_recall,
)
from gbd_ctu.evaluation.scenario_eval import evaluate_all_scenarios, evaluate_gnn_checkpoint

__all__ = [
    "auc_roc",
    "build_comparison_table",
    "classification_metrics",
    "compare_reports",
    "confusion_matrix_plot",
    "evaluate_all_scenarios",
    "evaluate_gnn_checkpoint",
    "f1_botnet",
    "false_positive_rate",
    "generate_figures",
    "metrics_frame",
    "precision_recall",
    "run_comparison",
    "wilcoxon_hybrid_vs_baseline",
]
