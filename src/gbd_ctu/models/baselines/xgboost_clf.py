"""GBD-CTU XGBoost baseline wrapper.

This module exposes a small typed wrapper around xgboost.XGBClassifier. Input is
a dense node-feature matrix and labels; output is a fitted estimator and class
probabilities.

``scale_pos_weight`` is computed automatically from the training labels to handle
class imbalance (ratio of negatives to positives).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional during static inspection
    XGBClassifier = None


class XGBoostBaseline:
    """Typed wrapper around XGBClassifier for consistent trainer integration.

    ``scale_pos_weight`` is computed from the label distribution during
    :meth:`fit` so that the model accounts for class imbalance.
    """

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
        self._init_kwargs: dict[str, Any] = dict(
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
        self.model: XGBClassifier | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "XGBoostBaseline":
        """Fit the estimator. ``scale_pos_weight`` is derived from ``y``."""
        n_neg = int(np.sum(y == 0))
        n_pos = int(np.sum(y == 1))
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        self.model = XGBClassifier(scale_pos_weight=scale_pos_weight, **self._init_kwargs)
        self.model.fit(x, y)
        self._feature_names = feature_names
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return hard class predictions (0 or 1)."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape ``(N, 2)``."""
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(x)

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame with columns ``feature`` and ``importance``.

        Importances are the normalised gain scores from XGBoost, sorted
        descending.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        scores = self.model.feature_importances_
        names = (
            self._feature_names
            if self._feature_names is not None
            else [f"f{i}" for i in range(len(scores))]
        )
        return (
            pd.DataFrame({"feature": names, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_params(self) -> dict[str, Any]:
        """Expose estimator parameters for logging and checkpoint metadata."""
        if self.model is not None:
            return self.model.get_params()
        return {**self._init_kwargs}

