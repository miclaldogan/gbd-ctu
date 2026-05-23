"""Backward-compatible baseline helpers.

This module preserves the original flat import surface while delegating to the
production baseline trainer under `gbd_ctu.training.trainer_baseline`.
"""

from gbd_ctu.training.trainer_baseline import train_baselines as train_and_compare_baselines
