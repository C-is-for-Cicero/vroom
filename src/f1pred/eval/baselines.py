"""Baselines every model is compared against.

1. Grid baseline (post-quali): predicted finishing order = grid order.
2. Bookmaker implied probabilities, where available (odds ingestion is a
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
