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

# Telemetry car profile × track fingerprint (step 5): from FP of race R,
# cutoff end of practice — valid in both modes. NaN where not yet built.
TELEMETRY_FEATURES: list[str] = [
    "tel_slow_s",
    "tel_med_s",
    "tel_fast_s",
    "tel_top_speed",
    "tel_fade",
    "ix_slow",
    "ix_med",
    "ix_fast",
    "ix_straight",
]

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
    "fp_longrun_delta_s",
    "fp_deg_slope",
    "fp_consistency",
    *TELEMETRY_FEATURES,
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
    # FP long runs happen before qualifying: the within-weekend signal
    "fp_longrun_delta_s",
    "fp_deg_slope",
    "fp_consistency",
    *TELEMETRY_FEATURES,
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

    def __init__(
        self, features: list[str] | None = None, seed: int = 0, sigma_folds: int = 4
    ) -> None:
        self.features = features if features is not None else list(POST_QUALI_FEATURES)
        self._seed = seed
        self._sigma_folds = sigma_folds
        self._sigma_fitted = False
        self._mean = LGBMRegressor(random_state=seed, **_LGBM_PARAMS)
        self._sigma = LGBMRegressor(random_state=seed, **_LGBM_PARAMS)

    def fit(self, train: pd.DataFrame) -> PaceModel:
        """Fit on rows where the target exists (finishers with clean laps).

        The sigma model trains on OUT-OF-FOLD absolute residuals from
        time-ordered folds (never residuals the mean model was fitted on),
        scaled by sqrt(pi/2) (E|N(0,s)| = s*sqrt(2/pi)), so sigma is an
        honest estimate of predictive, not in-sample, error.

        Input: frame with self.features, pace_delta_s, season, round.
        """
        rows = train[train["pace_delta_s"].notna()].sort_values(
            ["season", "round"], kind="stable"
        )
        x = rows[self.features]
        y = rows["pace_delta_s"]
        self._mean.fit(x, y)

        races = rows[["season", "round"]].drop_duplicates().itertuples(index=False, name=None)
        races = list(races)
        keys = pd.Series(list(zip(rows["season"], rows["round"], strict=True)), index=rows.index)
        oof_x, oof_resid = [], []
        bounds = np.linspace(0, len(races), self._sigma_folds + 1).astype(int)
        for k in range(1, self._sigma_folds):
            fit_races = set(races[: bounds[k]])
            val_races = set(races[bounds[k] : bounds[k + 1]])
            fit_mask = keys.isin(fit_races)
            val_mask = keys.isin(val_races)
            if fit_mask.sum() < 50 or val_mask.sum() == 0:
                continue
            fold_model = LGBMRegressor(random_state=self._seed, **_LGBM_PARAMS)
            fold_model.fit(x[fit_mask], y[fit_mask])
            oof_x.append(x[val_mask])
            oof_resid.append(np.abs(y[val_mask] - fold_model.predict(x[val_mask])))
        if oof_x:
            self._sigma.fit(pd.concat(oof_x), np.concatenate(oof_resid))
            self._sigma_fitted = True
        else:  # tiny training set: fall back to in-sample residuals
            self._sigma.fit(x, np.abs(y - self._mean.predict(x)))
            self._sigma_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict (mu, sigma) in s/lap for each row; sigma floored."""
        x = df[self.features]
        mu = self._mean.predict(x)
        # abs-residual -> Gaussian sigma: E|N(0,s)| = s*sqrt(2/pi)
        sigma = self._sigma.predict(x) * np.sqrt(np.pi / 2)
        return mu, np.maximum(sigma, SIGMA_FLOOR_S)
