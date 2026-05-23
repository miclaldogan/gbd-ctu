"""Backward-compatible metric helpers.

This module preserves the original flat import surface while delegating to the
production metrics module under `gbd_ctu.evaluation.metrics`.
"""

from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame as summarize_records
