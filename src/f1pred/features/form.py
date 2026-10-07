"""Rolling form features on the core table.

Every rolling feature uses .shift(1) within its group, so the value for race
R aggregates races strictly before R.

Leakage cutoff: all features here are computable the moment the *previous*
race's results are published — available in both pre_quali and post_quali
modes.

Team features are season-scoped (expanding mean within the current season,
shifted), so a regulation reset such as 2026 never leaks stale team pace:
round 1 of a season has no team form and the model falls back to driver
features. Driver features roll across seasons — driver skill carries over.
"""

from __future__ import annotations

import pandas as pd

DRIVER_FORM_WINDOW = 5
DNF_RATE_WINDOW = 10


def add_form_features(core: pd.DataFrame) -> pd.DataFrame:
    """Add rolling driver/team form columns to a core table.

    Input: core table (one row per season/round/driver) with columns
    position, grid, dnf, classified, team. Output: a copy with
    driver_form_finish, driver_form_vs_grid, driver_dnf_rate,
    driver_experience, team_season_finish added. Rows where a feature has no
    history hold NaN (LightGBM handles missing natively).
    """
    df = core.sort_values(["season", "round"], kind="stable").copy()

    # Positions gained vs grid; pit-lane starts (grid 0) count as last-ish —
    # leave NaN rather than invent a number.
    finish = df["position"].astype("Float64")
    grid = df["grid"].replace(0, pd.NA).astype("Float64")
    df["_vs_grid"] = grid - finish  # positive = gained places

    by_driver = df.groupby("driver_id", sort=False)
    df["driver_form_finish"] = by_driver["position"].transform(
        lambda s: s.shift(1).rolling(DRIVER_FORM_WINDOW, min_periods=1).mean()
    )
    df["driver_form_vs_grid"] = by_driver["_vs_grid"].transform(
        lambda s: s.shift(1).rolling(DRIVER_FORM_WINDOW, min_periods=1).mean()
    )
    df["driver_dnf_rate"] = by_driver["dnf"].transform(
        lambda s: s.astype(float).shift(1).rolling(DNF_RATE_WINDOW, min_periods=3).mean()
    )
    df["driver_experience"] = by_driver.cumcount()

    # Team form: expanding mean finish within the current season only, from
    # races before R (mean over both cars).
    team_race = (
        df.groupby(["season", "round", "team"], sort=False)["position"]
        .mean()
        .rename("team_race_finish")
        .reset_index()
        .sort_values(["season", "round"], kind="stable")
    )
    team_race["team_season_finish"] = (
        team_race.groupby(["season", "team"], sort=False)["team_race_finish"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df = df.merge(
        team_race[["season", "round", "team", "team_season_finish"]],
        on=["season", "round", "team"],
        how="left",
    )
    return df.drop(columns=["_vs_grid"])
