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
    _fpr_at_tpr,
    auc_roc,
    classification_metrics,
    confusion_matrix_plot,
    f1_botnet,
    false_positive_rate,
    metrics_frame,
    precision_recall,
)
from gbd_ctu.evaluation.plots import (
    PALETTE,
    confusion_matrix_plot as plots_confusion_matrix_plot,
    roc_pr_curves,
    scenario_comparison_heatmap,
)
from gbd_ctu.evaluation.scenario_eval import evaluate_all_scenarios, evaluate_gnn_checkpoint

__all__ = [
    # compare
    "build_comparison_table",
    "compare_reports",
    "generate_figures",
    "run_comparison",
    "wilcoxon_hybrid_vs_baseline",
    # metrics
    "_fpr_at_tpr",
    "auc_roc",
    "classification_metrics",
    "confusion_matrix_plot",
    "f1_botnet",
    "false_positive_rate",
    "metrics_frame",
    "precision_recall",
    # plots
    "PALETTE",
    "plots_confusion_matrix_plot",
    "roc_pr_curves",
    "scenario_comparison_heatmap",
    # scenario_eval
    "evaluate_all_scenarios",
    "evaluate_gnn_checkpoint",
]
