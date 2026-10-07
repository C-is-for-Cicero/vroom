"""Car profile v1: free-practice long-run features per driver per weekend.

From FP laps (no telemetry yet — lap times only):
- fp_longrun_delta_s: median long-run stint pace minus the field-best
  median, s/lap. The within-weekend pace signal that exists BEFORE quali.
- fp_deg_slope: mean within-stint lap-time slope (s per lap of tyre age;
  fuel burn is part of the slope — consistent across drivers in a session).
- fp_consistency: std of lap times around the stint fit.
- fp_longrun_laps: how much long running the value is based on.

Sessions: FP2 is the representative long-run session; FP3 then FP1 are
fallbacks (sprint weekends only have FP1). Telemetry-based corner-type
features join later in build-order step 5.

Leakage cutoff: end of free practice for race R — valid in pre_quali and
post_quali modes for that same weekend.

Cached per season in data/processed/fp_longrun/{season}.parquet, one row
per (season, round, driver_code).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.config import PROCESSED_DIR
from f1pred.ingest.fastf1_loader import load_session

MIN_STINT_LAPS = 5
SLOW_LAP_FACTOR = 1.05  # within a stint, drop obvious traffic laps
FP_SESSION_PRIORITY = ("FP2", "FP3", "FP1")

FP_LONGRUN_DIR = PROCESSED_DIR / "fp_longrun"

FP_COLUMNS = [
    "season",
    "round",
    "driver_code",
    "fp_longrun_delta_s",
    "fp_deg_slope",
    "fp_consistency",
    "fp_longrun_laps",
]


def _stint_features(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-driver long-run features from one FP session's laps."""
    df = pd.DataFrame(
        {
            "driver_code": laps["Driver"].values,
            "stint": laps["Stint"].astype(float).values,
            "lap_s": laps["LapTime"].dt.total_seconds().values,
            "tyre_life": laps["TyreLife"].astype(float).values,
        }
    ).dropna(subset=["lap_s", "stint"])

    rows: list[dict] = []
    for driver, drv in df.groupby("driver_code"):
        medians, slopes, resid_sds, n_laps = [], [], [], 0
        for _, stint in drv.groupby("stint"):
            stint = stint[stint["lap_s"] <= stint["lap_s"].median() * SLOW_LAP_FACTOR]
            if len(stint) < MIN_STINT_LAPS:
                continue
            slope, intercept = np.polyfit(stint["tyre_life"], stint["lap_s"], 1)
            fitted = slope * stint["tyre_life"] + intercept
            medians.append(float(stint["lap_s"].median()))
            slopes.append(float(slope))
            resid_sds.append(float((stint["lap_s"] - fitted).std()))
            n_laps += len(stint)
        if not medians:
            continue
        rows.append(
            {
                "driver_code": driver,
                "longrun_median_s": min(medians),  # best long-run stint
                "fp_deg_slope": float(np.mean(slopes)),
                "fp_consistency": float(np.mean(resid_sds)),
                "fp_longrun_laps": n_laps,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["fp_longrun_delta_s"] = out["longrun_median_s"] - out["longrun_median_s"].min()
    return out.drop(columns=["longrun_median_s"])


def fp_longrun_for_round(season: int, round_number: int) -> pd.DataFrame:
    """Long-run features for one weekend, trying FP2 → FP3 → FP1."""
    for code in FP_SESSION_PRIORITY:
        try:
            session = load_session(season, round_number, code, laps=True, telemetry=False)
        except Exception as exc:  # session missing, not yet run, or download hiccup
            print(f"fp_longrun {season} round {round_number} {code}: "
                  f"{type(exc).__name__}: {str(exc)[:80]}")
            continue
        laps = session.laps.pick_wo_box()
        if len(laps) == 0:
            continue
        out = _stint_features(laps)
        if not out.empty:
            out["season"] = season
            out["round"] = round_number
            return out[FP_COLUMNS]
    return pd.DataFrame(columns=FP_COLUMNS)


def build_fp_longrun(season: int, rounds: list[int]) -> pd.DataFrame:
    """Build/update the season's FP long-run parquet; existing rounds kept."""
    FP_LONGRUN_DIR.mkdir(parents=True, exist_ok=True)
    path = FP_LONGRUN_DIR / f"{season}.parquet"
    existing = pd.read_parquet(path) if path.exists() else None
    done = set(existing["round"].unique()) if existing is not None else set()

    frames = [] if existing is None else [existing]
    for rnd in rounds:
        if rnd in done:
            continue
        frame = fp_longrun_for_round(season, rnd)
        if frame.empty:
            print(f"fp_longrun {season} round {rnd}: no usable long runs")
            continue
        frames.append(frame)
        print(f"fp_longrun {season} round {rnd}: {len(frame)} drivers")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FP_COLUMNS)
    if not out.empty:
        out = out.sort_values(["season", "round"]).reset_index(drop=True)
        out.to_parquet(path, index=False)
    return out


def load_fp_longrun(seasons: list[int]) -> pd.DataFrame:
    """Concatenate cached FP long-run parquets for `seasons`."""
    frames = [
        pd.read_parquet(FP_LONGRUN_DIR / f"{s}.parquet")
        for s in seasons
        if (FP_LONGRUN_DIR / f"{s}.parquet").exists()
    ]
    if not frames:
        return pd.DataFrame(columns=FP_COLUMNS)
    return pd.concat(frames, ignore_index=True)
