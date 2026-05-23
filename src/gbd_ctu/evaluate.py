"""Backward-compatible evaluation helpers.

This module preserves the original flat import surface while delegating to the
production scenario evaluator under `gbd_ctu.evaluation.scenario_eval`.
"""

from gbd_ctu.evaluation.scenario_eval import evaluate_gnn_checkpoint as evaluate_gnn
