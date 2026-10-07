"""DNF model: P(driver retires) per driver per race.

LightGBM classifier on reliability/chaos features. The label is a retirement
(positionText 'R'); disqualifications and withdrawals are not DNFs.

Leakage cutoff: every feature is a shifted rolling rate known before any
session of race R — usable in both pre_quali and post_quali modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

DNF_FEATURES: list[str] = [
    "driver_dnf_rate",
    "team_dnf_rate",
    "circuit_dnf_rate",
    "driver_experience",
]

# Probabilities are clipped: no driver is ever a certain finisher or a
# near-certain retirement before the race starts.
P_DNF_MIN = 0.02
P_DNF_MAX = 0.60

_LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 50,
    "subsample": 0.9,
    "subsample_freq": 1,
    "verbosity": -1,
}


class DnfModel:
    """Binary classifier giving P(DNF) per entry."""

    def __init__(self, features: list[str] | None = None, seed: int = 0) -> None:
        self.features = features if features is not None else list(DNF_FEATURES)
        self._clf = LGBMClassifier(random_state=seed, **_LGBM_PARAMS)

    def fit(self, train: pd.DataFrame) -> DnfModel:
        """Fit on all entries (dnf=True only for retirements).

        Input: frame with self.features and the boolean dnf column.
        """
        self._clf.fit(train[self.features], train["dnf"].astype(int))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """P(DNF) per row, clipped to [P_DNF_MIN, P_DNF_MAX]."""
        p = self._clf.predict_proba(df[self.features])[:, 1]
        return np.clip(p, P_DNF_MIN, P_DNF_MAX)
