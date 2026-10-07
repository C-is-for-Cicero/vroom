"""Post-quali pace model: LightGBM regression of race pace delta (s/lap),
plus a per-driver sigma for the simulator.

Target: pace_delta_s from features/race_pace.py (fuel-corrected clean-lap
median delta to the field best), trained on finishers with enough clean laps.

Sigma: a second LightGBM fitted on absolute residuals of the mean model, so
uncertainty varies with the same features (rookies and midfield cars get
wider distributions than a settled front-runner). Floored to keep the
simulator sane.

Leakage cutoff: every feature below is known at the end of qualifying for
race R. pre_quali gets its own separate model in build-order step 7 — never
this model with a flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

POST_QUALI_FEATURES: list[str] = [
    "grid",
    "quali_position",
    "quali_delta_pct",
    "driver_form_finish",
    "driver_form_vs_grid",
    "driver_experience",
    "team_season_finish",
    "pace_form_3",
    "team_pace_form_3",
]

# The primary product mode: everything here is known BEFORE qualifying
# (leakage cutoff: end of the previous race; FP-based features join in
# build-order steps 5/7). Trained as its own model instance — never the
# post-quali model with a flag.
PRE_QUALI_FEATURES: list[str] = [
    "driver_form_finish",
    "driver_form_vs_grid",
    "driver_experience",
    "team_season_finish",
    "pace_form_3",
    "team_pace_form_3",
    "season_points_prior",
    "prev_season_points",
]

SIGMA_FLOOR_S = 0.05

_LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbosity": -1,
}


def add_pace_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling pace-delta form from past races (requires pace_delta_s).

    pace_form_3 / team_pace_form_3: mean of the driver's / team's last 3
    observed race pace deltas, shifted so race R only sees races before R.
    Team pace form is season-scoped (regulation resets). Leakage cutoff:
    previous race's results.
    """
    df = df.sort_values(["season", "round"], kind="stable").copy()
    df["pace_form_3"] = df.groupby("driver_id", sort=False)["pace_delta_s"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    team_race = (
        df.groupby(["season", "round", "team"], sort=False)["pace_delta_s"]
        .mean()
        .rename("team_race_pace")
        .reset_index()
        .sort_values(["season", "round"], kind="stable")
    )
    team_race["team_pace_form_3"] = team_race.groupby(["season", "team"], sort=False)[
        "team_race_pace"
    ].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return df.merge(
        team_race[["season", "round", "team", "team_pace_form_3"]],
        on=["season", "round", "team"],
        how="left",
    )


class PaceModel:
    """Mean + sigma pace model (post_quali mode)."""

    def __init__(self, features: list[str] | None = None, seed: int = 0) -> None:
        self.features = features if features is not None else list(POST_QUALI_FEATURES)
        self._mean = LGBMRegressor(random_state=seed, **_LGBM_PARAMS)
        self._sigma = LGBMRegressor(random_state=seed, **_LGBM_PARAMS)

    def fit(self, train: pd.DataFrame) -> PaceModel:
        """Fit on rows where the target exists (finishers with clean laps).

        Input: frame with self.features and pace_delta_s. Output: self.
        """
        rows = train[train["pace_delta_s"].notna()]
        x = rows[self.features]
        y = rows["pace_delta_s"]
        self._mean.fit(x, y)
        resid = np.abs(y - self._mean.predict(x))
        self._sigma.fit(x, resid)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict (mu, sigma) in s/lap for each row; sigma floored."""
        x = df[self.features]
        mu = self._mean.predict(x)
        sigma = np.maximum(self._sigma.predict(x), SIGMA_FLOOR_S)
        return mu, sigma
