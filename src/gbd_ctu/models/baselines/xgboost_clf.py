"""GBD-CTU XGBoost baseline wrapper.

This module exposes a small typed wrapper around xgboost.XGBClassifier. Input is
a dense node-feature matrix and labels; output is a fitted estimator and class
probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional during static inspection
    XGBClassifier = None


class XGBoostBaseline:
    """Typed wrapper around XGBClassifier for consistent trainer integration."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        tree_method: str = "hist",
    ) -> None:
        if XGBClassifier is None:
            raise ImportError("xgboost is required to use the XGBoost baseline.")
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            tree_method=tree_method,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        """Fit the estimator on dense node features."""

        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for node features."""

        return self.model.predict_proba(x)

    def get_params(self) -> dict[str, Any]:
        """Expose estimator parameters for logging and checkpoint metadata."""

        return self.model.get_params()
