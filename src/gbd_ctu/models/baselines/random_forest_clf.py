"""GBD-CTU Random Forest baseline wrapper.

This module exposes a typed wrapper around sklearn.ensemble.RandomForestClassifier.
Input is a dense node-feature matrix and labels; output is a fitted estimator and
class probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
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
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """Fit the estimator on dense node features."""

        self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for node features."""

        return self.model.predict_proba(x)

    def get_params(self) -> dict[str, Any]:
        """Expose estimator parameters for logging and checkpoint metadata."""

        return self.model.get_params()
