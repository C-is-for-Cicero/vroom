"""Fastest-lap model: a multinomial classifier over the field.

Exactly one driver per race sets the fastest lap, so per-race scores from a
binary LightGBM (is_fastest_lap) are normalised to sum to 1 across the
race's entrants — a softmax-style multinomial over the field, per
CLAUDE.md. Since 2025 the fastest lap earns no bonus point; this is a
standalone prediction market.

Leakage cutoff: post-quali features (grid, quali pace, form, FP long runs)
— all known before the race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from f1pred.models.pace import POST_QUALI_FEATURES

FASTEST_LAP_FEATURES: list[str] = list(POST_QUALI_FEATURES)

_LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "verbosity": -1,
}


class FastestLapModel:
    """P(sets the race's fastest lap) per driver, normalised per race."""

    def __init__(self, features: list[str] | None = None, seed: int = 0) -> None:
        self.features = features if features is not None else list(FASTEST_LAP_FEATURES)
        self._clf = LGBMClassifier(random_state=seed, **_LGBM_PARAMS)

    def fit(self, train: pd.DataFrame) -> FastestLapModel:
        """Fit on races where a fastest lap is recorded (is_fastest_lap)."""
        has_label = train.groupby(["season", "round"])["is_fastest_lap"].transform("any")
        rows = train[has_label]
        self._clf.fit(rows[self.features], rows["is_fastest_lap"].astype(int))
        return self

    def predict(self, race: pd.DataFrame) -> np.ndarray:
        """Per-driver fastest-lap probabilities for ONE race; sums to 1."""
        scores = self._clf.predict_proba(race[self.features])[:, 1]
        scores = np.clip(scores, 1e-6, None)
        return scores / scores.sum()
