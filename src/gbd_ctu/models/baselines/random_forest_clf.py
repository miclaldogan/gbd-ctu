"""GBD-CTU Random Forest baseline wrapper.

This module exposes a typed wrapper around sklearn.ensemble.RandomForestClassifier.
Input is a dense node-feature matrix and labels; output is a fitted estimator and
class probabilities.

``class_weight='balanced'`` is applied by default to down-weight the majority
(benign) class and improve recall on the minority (botnet) class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestBaseline:
    """Typed wrapper around RandomForestClassifier for consistent trainer integration."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=n_jobs,
        )
        self._feature_names: list[str] | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "RandomForestBaseline":
        """Fit the estimator on dense node features."""
        self.model.fit(x, y)
        self._feature_names = feature_names
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return hard class predictions (0 or 1)."""
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return class probability matrix of shape ``(N, 2)``."""
        return self.model.predict_proba(x)

    def feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame with columns ``feature`` and ``importance``.

        Importances are the mean decrease in impurity (Gini), sorted descending.
        """
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
        return self.model.get_params()

