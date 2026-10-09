"""Observed fastest-lap telemetry for the web app's Telemetry view.

This serves MEASURED data (speed/throttle/brake vs distance from FastF1),
not model output — Vroom predicts race-level quantities (pace, DNFs,
finishing distributions), never telemetry traces. First access to a
session downloads its telemetry (tens of MB, then cached); channels are
resampled onto a uniform distance grid to keep responses small.

Telemetry is ~4 Hz: fine for lap-shape comparison plots, too coarse for
micro analysis (per CLAUDE.md, features aggregate per segment).
"""

from __future__ import annotations

import numpy as np

from f1pred.ingest.fastf1_loader import load_session

N_POINTS_DEFAULT = 500


def resample_channels(
    distance: np.ndarray, channels: dict[str, np.ndarray], n_points: int
) -> dict[str, list[float]]:
    """Linearly resample each channel onto a uniform distance grid.

    Inputs: a monotonically increasing distance array (metres) and channels
    sampled at those distances. Output: {"dist": grid, <name>: values} with
    floats rounded for JSON size. Pure function (unit tested).
    """
    distance = np.asarray(distance, dtype=float)
    grid = np.linspace(0.0, float(distance[-1]), n_points)
    out: dict[str, list[float]] = {"dist": [round(x, 1) for x in grid]}
    for name, values in channels.items():
        resampled = np.interp(grid, distance, np.asarray(values, dtype=float))
        out[name] = [round(float(v), 2) for v in resampled]
    return out


def fastest_lap_channels(
    season: int,
    round_number: int,
    session_code: str,
    driver_code: str,
    n_points: int = N_POINTS_DEFAULT,
) -> dict:
    """Speed/throttle/brake/time vs distance for a driver's fastest lap.

    Raises LookupError when the driver set no timed lap in that session.
    The session (with telemetry) is downloaded on first access and cached.
    """
    session = load_session(season, round_number, session_code, laps=True, telemetry=True)
    laps = session.laps.pick_drivers(driver_code)
    lap = laps.pick_fastest() if len(laps) else None
    if lap is None or lap.isna()["LapTime"]:
        raise LookupError(f"{driver_code}: no timed lap in {session_code}")

    car = lap.get_car_data().add_distance()
    data = resample_channels(
        car["Distance"].to_numpy(),
        {
            "speed": car["Speed"].to_numpy(),
            "throttle": car["Throttle"].to_numpy(),
            "brake": car["Brake"].to_numpy().astype(float) * 100.0,
            "time": car["Time"].dt.total_seconds().to_numpy(),
        },
        n_points,
    )
    return {
        "driver": driver_code,
        "session": session_code,
        "lap_time": round(float(lap["LapTime"].total_seconds()), 3),
        **data,
    }
