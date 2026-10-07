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


def rank_outcome_probs(train: pd.DataFrame, rank: pd.Series) -> pd.DataFrame:
    """Historical P(win / podium / points | predicted rank), Laplace-smoothed.

    This turns a deterministic baseline ordering into probabilities, so the
    baselines can be scored on log-loss like the model. Inputs: training rows
    with the position column, and that baseline's rank per row (grid or
    standings order). Output: frame indexed by integer rank with columns
    p_win, p_podium, p_points. Leakage: pass training races only.
    """
    df = pd.DataFrame(
        {
            "rank": rank.round().astype(int),
            "win": (train["position"] == 1).astype(float),
            "podium": (train["position"] <= 3).astype(float),
            "points": (train["position"] <= 10).astype(float),
        }
    )
    grouped = df.groupby("rank").agg(n=("win", "size"), w=("win", "sum"),
                                     p=("podium", "sum"), t=("points", "sum"))
    out = pd.DataFrame(
        {
            "p_win": (grouped["w"] + 1) / (grouped["n"] + 2),
            "p_podium": (grouped["p"] + 1) / (grouped["n"] + 2),
            "p_points": (grouped["t"] + 1) / (grouped["n"] + 2),
        }
    )
    return out


def apply_rank_probs(probs: pd.DataFrame, rank: pd.Series) -> pd.DataFrame:
    """Look up rank_outcome_probs for a test race's ranks; unseen ranks fall
    back to the worst (last) row of the table."""
    idx = rank.round().astype(int).clip(upper=int(probs.index.max()))
    idx = idx.clip(lower=int(probs.index.min()))
    out = probs.reindex(idx.values)
    out.index = rank.index
    return out
