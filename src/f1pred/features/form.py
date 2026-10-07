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

    # Championship points accumulated BEFORE race R (race points only; good
    # enough for form/baseline purposes), plus the previous season's total as
    # the round-1 fallback. Both known before any session of race R.
    df["season_points_prior"] = (
        df.groupby(["season", "driver_id"], sort=False)["points"]
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0.0)
    )
    totals = df.groupby(["season", "driver_id"])["points"].sum().rename("prev_season_points")
    totals = totals.reset_index()
    totals["season"] += 1
    df = df.merge(totals, on=["season", "driver_id"], how="left")
    df["prev_season_points"] = df["prev_season_points"].fillna(0.0)

    # DNF-related rates for the DNF model. Car reliability belongs to the
    # team; rolling over the team's last 20 car-races (~10 weekends).
    df["team_dnf_rate"] = df.groupby("team", sort=False)["dnf"].transform(
        lambda s: s.astype(float).shift(1).rolling(20, min_periods=5).mean()
    )
    # Circuit chaos level: expanding mean of the share of the field that
    # DNF'd at this circuit in past races (strictly before race R).
    race_dnf = (
        df.groupby(["season", "round", "circuit_id"], sort=False)["dnf"]
        .mean()
        .rename("race_dnf_share")
        .reset_index()
        .sort_values(["season", "round"], kind="stable")
    )
    race_dnf["circuit_dnf_rate"] = race_dnf.groupby("circuit_id", sort=False)[
        "race_dnf_share"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    df = df.merge(
        race_dnf[["season", "round", "circuit_id", "circuit_dnf_rate"]],
        on=["season", "round", "circuit_id"],
        how="left",
    )
    return df.drop(columns=["_vs_grid"])
