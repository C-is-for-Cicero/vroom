"""Race-pace target: fuel-corrected clean-lap pace delta in seconds per lap.

This is the LABEL the pace model regresses (per docs/DECISIONS.md), computed
from FastF1 race laps:

1. keep clean laps: green-flag, no in/out laps, drop lap 1, drop laps slower
   than 107% of the driver's own clean median (traffic/damage);
2. remove the common race-long trend (fuel burn + track evolution) with one
   linear fit of lap time vs lap number across the whole field;
3. per driver, take the median corrected lap time;
4. pace_delta_s = driver median − field-best median (0 = fastest car).

Leakage: a race's pace delta exists only after that race — it is a label for
race R and a (shifted) form feature for races after R.

Results are cached per season in data/processed/race_pace/{season}.parquet;
one row per (season, round, driver_code).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.config import PROCESSED_DIR
from f1pred.ingest.fastf1_loader import load_session

MIN_CLEAN_LAPS = 5
SLOW_LAP_FACTOR = 1.07

RACE_PACE_DIR = PROCESSED_DIR / "race_pace"


def race_pace_for_round(season: int, round_number: int) -> pd.DataFrame:
    """Pace deltas for one race. Columns: season, round, driver_code,
    pace_delta_s, n_clean_laps. Drivers with < MIN_CLEAN_LAPS clean laps are
    dropped (their pace is unknowable from this race)."""
    session = load_session(season, round_number, "R", laps=True, telemetry=False)
    laps = session.laps.pick_wo_box().pick_track_status("1")
    laps = laps[laps["LapNumber"] > 1]
    laps = laps[laps["LapTime"].notna()]
    if laps.empty:
        return pd.DataFrame(
            columns=["season", "round", "driver_code", "pace_delta_s", "n_clean_laps"]
        )

    df = pd.DataFrame(
        {
            "driver_code": laps["Driver"].values,
            "lap_number": laps["LapNumber"].astype(float).values,
            "lap_s": laps["LapTime"].dt.total_seconds().values,
        }
    )
    # drop a driver's own outlier laps (traffic, damage)
    med = df.groupby("driver_code")["lap_s"].transform("median")
    df = df[df["lap_s"] <= med * SLOW_LAP_FACTOR]

    # one field-wide linear trend for fuel burn + track evolution
    slope = np.polyfit(df["lap_number"], df["lap_s"], 1)[0]
    df["corrected_s"] = df["lap_s"] - slope * df["lap_number"]

    agg = (
        df.groupby("driver_code")["corrected_s"]
        .agg(median_s="median", n_clean_laps="size")
        .reset_index()
    )
    agg = agg[agg["n_clean_laps"] >= MIN_CLEAN_LAPS].copy()
    agg["pace_delta_s"] = agg["median_s"] - agg["median_s"].min()
    agg["season"] = season
    agg["round"] = round_number
    return agg[["season", "round", "driver_code", "pace_delta_s", "n_clean_laps"]]


def build_race_pace(season: int, rounds: list[int]) -> pd.DataFrame:
    """Build/update the season's race-pace parquet for `rounds`.

    Rounds already in the parquet are kept as-is (raw pulls are immutable);
    rounds whose race data isn't available yet are skipped with a notice.
    """
    RACE_PACE_DIR.mkdir(parents=True, exist_ok=True)
    path = RACE_PACE_DIR / f"{season}.parquet"
    existing = pd.read_parquet(path) if path.exists() else None
    done = set(existing["round"].unique()) if existing is not None else set()

    frames = [] if existing is None else [existing]
    for rnd in rounds:
        if rnd in done:
            continue
        try:
            frame = race_pace_for_round(season, rnd)
        except Exception as exc:  # session missing / not yet run
            print(f"race_pace {season} round {rnd}: skipped ({type(exc).__name__}: {exc})")
            continue
        if not frame.empty:
            frames.append(frame)
            print(f"race_pace {season} round {rnd}: {len(frame)} drivers")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["season", "round"]).reset_index(drop=True)
        out.to_parquet(path, index=False)
    return out


def load_race_pace(seasons: list[int]) -> pd.DataFrame:
    """Concatenate the cached race-pace parquets for `seasons`."""
    frames = []
    for season in seasons:
        path = RACE_PACE_DIR / f"{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(
            columns=["season", "round", "driver_code", "pace_delta_s", "n_clean_laps"]
        )
    return pd.concat(frames, ignore_index=True)
