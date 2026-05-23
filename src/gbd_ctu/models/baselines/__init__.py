"""GBD-CTU classical baseline models.

This package contains thin wrappers for tabular baselines trained on per-node
features derived from CTU-13 communication graphs.
"""

from gbd_ctu.models.baselines.random_forest_clf import RandomForestBaseline
from gbd_ctu.models.baselines.xgboost_clf import XGBoostBaseline

__all__ = ["RandomForestBaseline", "XGBoostBaseline"]
