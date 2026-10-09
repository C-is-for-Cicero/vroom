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
            if type(exc).__name__ == "RateLimitExceededError":
                raise  # out of API budget: let the builder stop the whole pass
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
    exhausted = False
    for rnd in rounds:
        if rnd in done:
            continue
        try:
            frame = fp_longrun_for_round(season, rnd)
        except Exception as exc:
            if type(exc).__name__ == "RateLimitExceededError":
                exhausted = True
                break
            raise
        if frame.empty:
            print(f"fp_longrun {season} round {rnd}: no usable long runs")
            continue
        frames.append(frame)
        print(f"fp_longrun {season} round {rnd}: {len(frame)} drivers")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FP_COLUMNS)
    if not out.empty:
        out = out.sort_values(["season", "round"]).reset_index(drop=True)
        out.to_parquet(path, index=False)
    if exhausted:
        from f1pred.ingest.fastf1_loader import FastF1BudgetExhausted

        raise FastF1BudgetExhausted(f"stopped during fp_longrun {season}; progress saved")
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


# ---------------------------------------------------------------------------
# Telemetry car profile (build-order step 5, full): per-driver lap-shape
# features from FP telemetry, aggregated per corner-class zone — never per
# 4 Hz sample (CLAUDE.md). All cutoffs = end of free practice for race R,
# so these feed BOTH prediction modes.
# ---------------------------------------------------------------------------

SLOW_MAX_KMH = 120.0   # corner class boundaries by apex minimum speed
MED_MAX_KMH = 200.0
ZONE_BEFORE_M = 200.0  # corner zone around the apex marker
ZONE_AFTER_M = 150.0
HEAVY_BRAKE_DROP_KMH = 120.0

TEL_DRIVER_DIR = PROCESSED_DIR / "tel_driver"
TEL_TRACK_DIR = PROCESSED_DIR / "tel_track"

CLASS_NAMES = ("slow", "med", "fast")


def corner_zones(corner_dists: list[float], lap_length: float) -> list[tuple[float, float]]:
    """Corner zones [apex-200m, apex+150m], clipped at midpoints between
    neighbouring apexes and at the lap bounds. Pure function."""
    zones = []
    for i, d in enumerate(corner_dists):
        start, end = d - ZONE_BEFORE_M, d + ZONE_AFTER_M
        if i > 0:
            start = max(start, (corner_dists[i - 1] + d) / 2)
        if i < len(corner_dists) - 1:
            end = min(end, (d + corner_dists[i + 1]) / 2)
        zones.append((max(start, 0.0), min(end, lap_length)))
    return zones


def classify_zones(
    zones: list[tuple[float, float]], dist: np.ndarray, speed: np.ndarray
) -> list[int]:
    """Class per zone (0 slow / 1 med / 2 fast) from the reference lap's
    minimum speed inside the zone."""
    classes = []
    for start, end in zones:
        mask = (dist >= start) & (dist <= end)
        vmin = float(np.min(speed[mask])) if mask.any() else float("nan")
        classes.append(0 if vmin < SLOW_MAX_KMH else 1 if vmin < MED_MAX_KMH else 2)
    return classes


def _time_at(dist: np.ndarray, time_s: np.ndarray, d: float) -> float:
    return float(np.interp(d, dist, time_s))


def lap_profile(
    dist: np.ndarray,
    time_s: np.ndarray,
    speed: np.ndarray,
    zones: list[tuple[float, float]],
    classes: list[int],
) -> dict[str, float]:
    """Aggregate one lap into per-class times, min speeds, top speed and
    energy fade on the longest straight. Pure function (unit tested)."""
    out: dict[str, float] = {}
    for ci, name in enumerate(CLASS_NAMES):
        total, vmins = 0.0, []
        for (start, end), cls in zip(zones, classes, strict=True):
            if cls != ci:
                continue
            total += _time_at(dist, time_s, end) - _time_at(dist, time_s, start)
            mask = (dist >= start) & (dist <= end)
            if mask.any():
                vmins.append(float(np.min(speed[mask])))
        out[f"time_{name}"] = total
        out[f"vmin_{name}"] = float(np.mean(vmins)) if vmins else float("nan")
    out["top_speed"] = float(np.max(speed))

    # straights = gaps between consecutive zones; fade = speed lost before
    # the end of the longest one (battery clipping signature)
    bounds = [0.0] + [b for z in zones for b in z] + [float(dist[-1])]
    gaps = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2)]
    gaps = [(a, b) for a, b in gaps if b - a > 50]
    if gaps:
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        mask = (dist >= a) & (dist <= b)
        v_end = float(np.interp(b - 20, dist, speed))
        out["fade"] = float(np.max(speed[mask])) - v_end if mask.any() else float("nan")
        out["longest_straight_m"] = b - a
    else:
        out["fade"] = float("nan")
        out["longest_straight_m"] = float("nan")
    return out


def _car_arrays(lap) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    car = lap.get_car_data().add_distance()
    return (
        car["Distance"].to_numpy(dtype=float),
        car["Time"].dt.total_seconds().to_numpy(dtype=float),
        car["Speed"].to_numpy(dtype=float),
        car["Throttle"].to_numpy(dtype=float),
    )


def telemetry_features_for_round(season: int, round_number: int) -> tuple[pd.DataFrame, dict]:
    """Per-driver telemetry profile deltas + the event's track fingerprint.

    Uses the first FP session (FP2 -> FP3 -> FP1) with telemetry; every
    driver's fastest lap is compared to the session-best lap per corner
    class. Raises on download failure (budget errors propagate)."""
    from f1pred.ingest.fastf1_loader import load_session

    session = None
    for code in FP_SESSION_PRIORITY:
        try:
            session = load_session(season, round_number, code, laps=True, telemetry=True)
            if len(session.laps):
                break
        except Exception as exc:
            if type(exc).__name__ == "RateLimitExceededError":
                raise
            session = None
    if session is None or not len(session.laps):
        raise LookupError(f"no FP session with laps for {season} round {round_number}")

    ref_lap = session.laps.pick_fastest()
    ref_d, ref_t, ref_v, ref_thr = _car_arrays(ref_lap)
    circuit_info = session.get_circuit_info()
    if circuit_info is None:
        raise LookupError(f"no circuit info for {season} round {round_number}")
    corners = circuit_info.corners
    corner_dists = sorted(float(x) for x in corners["Distance"].dropna())
    if len(corner_dists) < 4:
        raise LookupError(f"no corner markers for {season} round {round_number}")
    zones = corner_zones(corner_dists, float(ref_d[-1]))
    classes = classify_zones(zones, ref_d, ref_v)
    ref = lap_profile(ref_d, ref_t, ref_v, zones, classes)

    lap_time = float(ref_t[-1])
    heavy = 0
    for (start, end), _cls in zip(zones, classes, strict=True):
        mask = (ref_d >= start) & (ref_d <= end)
        if mask.any() and float(np.max(ref_v[mask]) - np.min(ref_v[mask])) > HEAVY_BRAKE_DROP_KMH:
            heavy += 1
    track = {
        "season": season,
        "round": round_number,
        "trk_slow_share": ref["time_slow"] / lap_time,
        "trk_med_share": ref["time_med"] / lap_time,
        "trk_fast_share": ref["time_fast"] / lap_time,
        "trk_straight_share": (
            1 - (ref["time_slow"] + ref["time_med"] + ref["time_fast"]) / lap_time
        ),
        "trk_full_throttle": float(np.mean(ref_thr > 95)),
        "trk_longest_straight_m": ref["longest_straight_m"],
        "trk_heavy_brakes": heavy,
    }

    rows = []
    for code in session.laps["Driver"].dropna().unique():
        laps = session.laps.pick_drivers(code)
        lap = laps.pick_fastest() if len(laps) else None
        if lap is None or lap.isna()["LapTime"]:
            continue
        try:
            d, t, v, _ = _car_arrays(lap)
        except Exception:
            continue
        prof = lap_profile(d, t, v, zones, classes)
        rows.append(
            {
                "season": season,
                "round": round_number,
                "driver_code": str(code),
                "tel_slow_s": prof["time_slow"] - ref["time_slow"],
                "tel_med_s": prof["time_med"] - ref["time_med"],
                "tel_fast_s": prof["time_fast"] - ref["time_fast"],
                "tel_top_speed": prof["top_speed"] - ref["top_speed"],
                "tel_fade": prof["fade"] - ref["fade"],
            }
        )
    return pd.DataFrame(rows), track


def build_telemetry_features(season: int, rounds: list[int]) -> pd.DataFrame:
    """Build/update tel_driver and tel_track parquets; rounds already built
    are kept, budget exhaustion persists progress and raises."""
    TEL_DRIVER_DIR.mkdir(parents=True, exist_ok=True)
    TEL_TRACK_DIR.mkdir(parents=True, exist_ok=True)
    dpath = TEL_DRIVER_DIR / f"{season}.parquet"
    tpath = TEL_TRACK_DIR / f"{season}.parquet"
    dframes = [pd.read_parquet(dpath)] if dpath.exists() else []
    tframes = [pd.read_parquet(tpath)] if tpath.exists() else []
    done = set(dframes[0]["round"].unique()) if dframes else set()

    exhausted = False
    for rnd in rounds:
        if rnd in done:
            continue
        try:
            drv, trk = telemetry_features_for_round(season, rnd)
        except Exception as exc:
            if type(exc).__name__ == "RateLimitExceededError":
                exhausted = True
                break
            print(f"telemetry {season} round {rnd}: skipped ({type(exc).__name__}: {exc})")
            continue
        if not drv.empty:
            dframes.append(drv)
            tframes.append(pd.DataFrame([trk]))
            print(f"telemetry {season} round {rnd}: {len(drv)} drivers")
    out = pd.concat(dframes, ignore_index=True) if dframes else pd.DataFrame()
    if not out.empty:
        out.sort_values(["season", "round"]).to_parquet(dpath, index=False)
        pd.concat(tframes, ignore_index=True).sort_values(["season", "round"]).to_parquet(
            tpath, index=False
        )
    if exhausted:
        from f1pred.ingest.fastf1_loader import FastF1BudgetExhausted

        raise FastF1BudgetExhausted(f"stopped during telemetry {season}; progress saved")
    return out


def load_telemetry_features(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(driver profiles, track fingerprints) for `seasons`, empty if unbuilt."""
    dfs, tfs = [], []
    for s in seasons:
        if (TEL_DRIVER_DIR / f"{s}.parquet").exists():
            dfs.append(pd.read_parquet(TEL_DRIVER_DIR / f"{s}.parquet"))
        if (TEL_TRACK_DIR / f"{s}.parquet").exists():
            tfs.append(pd.read_parquet(TEL_TRACK_DIR / f"{s}.parquet"))
    drv = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["season", "round", "driver_code", "tel_slow_s", "tel_med_s",
                 "tel_fast_s", "tel_top_speed", "tel_fade"])
    trk = pd.concat(tfs, ignore_index=True) if tfs else pd.DataFrame(
        columns=["season", "round", "trk_slow_share", "trk_med_share", "trk_fast_share",
                 "trk_straight_share", "trk_full_throttle", "trk_longest_straight_m",
                 "trk_heavy_brakes"])
    return drv, trk
