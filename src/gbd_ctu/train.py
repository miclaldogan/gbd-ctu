"""Backward-compatible training helpers.

This module preserves the original flat import surface while delegating to the
production GNN trainer under `gbd_ctu.training.trainer_gnn`.
"""

from gbd_ctu.training.trainer_gnn import train_gnn
