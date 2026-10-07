"""Track fingerprint v1: history-based circuit features.

- circuit_grid_hold: how strongly grid order held at this circuit in past
  races — the expanding mean (shifted, so race R only sees earlier races)
  of the per-race Spearman correlation between grid and finishing position.
  High = processional (Monaco-like), low = chaotic/overtaking-friendly.
  The simulator scales its grid-effect weight by it.

Telemetry-based fingerprint features (% full throttle, corner-type shares)
join in the full build-order step 5.

Leakage cutoff: previous race at the same circuit.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


def _race_grid_hold(race: pd.DataFrame) -> float:
    mask = race["position"].notna() & (race["grid"] > 0)
    if mask.sum() < 5:
        return float("nan")
    rho = spearmanr(race.loc[mask, "grid"], race.loc[mask, "position"]).statistic
    return float(rho)


def add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit_grid_hold to a core-like frame (needs grid, position)."""
    df = df.sort_values(["season", "round"], kind="stable").copy()
    per_race = (
        df.groupby(["season", "round", "circuit_id"], sort=False)[["grid", "position"]]
        .apply(_race_grid_hold)
        .rename("race_grid_hold")
        .reset_index()
        .sort_values(["season", "round"], kind="stable")
    )
    per_race["circuit_grid_hold"] = per_race.groupby("circuit_id", sort=False)[
        "race_grid_hold"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    return df.merge(
        per_race[["season", "round", "circuit_id", "circuit_grid_hold"]],
        on=["season", "round", "circuit_id"],
        how="left",
    )


def grid_effect_scale(test: pd.DataFrame, train: pd.DataFrame) -> float:
    """Per-race multiplier for the simulator's fitted grid-effect weight.

    scale = this circuit's grid hold / training-average grid hold, clipped
    to [0.3, 2.0]; 1.0 when the circuit has no history.
    """
    hold = test["circuit_grid_hold"].iloc[0]
    base = train["circuit_grid_hold"].mean()
    if pd.isna(hold) or pd.isna(base) or base <= 0:
        return 1.0
    return float(min(max(hold / base, 0.3), 2.0))
