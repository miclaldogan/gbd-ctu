"""Backward-compatible comparison helpers.

This module preserves the original flat import surface while delegating to the
production comparison module under `gbd_ctu.evaluation.compare`.
"""

from gbd_ctu.evaluation.compare import compare_reports
