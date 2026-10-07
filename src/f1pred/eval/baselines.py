"""Baselines every model is compared against.

1. Grid baseline (post-quali): predicted finishing order = grid order.
2. Standings baseline (pre-quali): predicted order = championship points
   before this race; previous season's points break ties (and decide
   round 1). The grid does not exist before qualifying, so this is the
   naive guess for the primary pre_quali mode.
3. Bookmaker implied probabilities, where available (odds ingestion is a
   later step; the report table keeps the column slot).
"""

from __future__ import annotations

import pandas as pd


def grid_baseline_rank(race_df: pd.DataFrame) -> pd.Series:
    """Predicted rank = grid position for one race's drivers.

    Pit-lane starters (grid 0) rank behind everyone else. Leakage cutoff:
    grid is known once qualifying (and any penalties) are published —
    post_quali mode only.
    """
    grid = race_df["grid"].astype(float).replace(0.0, 99.0)
    return grid.rank(method="first")


def standings_baseline_rank(race_df: pd.DataFrame) -> pd.Series:
    """Predicted rank = championship order before this race, for one race.

    Needs season_points_prior and prev_season_points (features/form.py).
    Leakage cutoff: previous race's results — valid in pre_quali mode.
    """
    score = race_df["season_points_prior"] * 10_000 + race_df["prev_season_points"]
    return (-score).rank(method="first")
