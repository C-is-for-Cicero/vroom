"""Core race table: one row per (season, round, driver), from raw Jolpica pulls.

Reads only the local raw cache (data/raw/jolpica) via JolpicaClient, which is
cache-first — run `python -m f1pred.ingest` beforehand. Output is written to
data/processed/core.parquet.

Leakage cutoffs per column group:
- identity/grid/quali columns: known at the end of qualifying for race R
  (usable in post_quali mode);
- outcome columns (position, status, dnf, points, pace target later): only
  known after race R — labels, never features for race R itself.
"""

from __future__ import annotations

import re

import pandas as pd

from f1pred.config import PROCESSED_DIR, canonical_constructor
from f1pred.ingest.jolpica import JolpicaClient

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d{1,2})\.(\d{3})$")

# positionText values for drivers who did not take / complete the race.
_RETIRED = "R"  # retired — the DNF signal
_DISQUALIFIED = "D"
_WITHDRAWN = "W"
_EXCLUDED = "E"
_FAILED_TO_QUALIFY = "F"


def time_str_to_seconds(value: str | None) -> float | None:
    """Parse a Jolpica lap/quali time string ("1:23.456" or "59.123") to seconds."""
    if not value:
        return None
    m = _TIME_RE.match(value.strip())
    if m is None:
        return None
    minutes = int(m.group(1) or 0)
    return minutes * 60 + int(m.group(2)) + int(m.group(3)) / 1000


def results_frame(seasons: list[int], client: JolpicaClient | None = None) -> pd.DataFrame:
    """Race results for `seasons` as a flat frame, one row per (season, round, driver).

    Outcome columns are labels (known only post-race). `dnf` is True for
    retirements (positionText == 'R'); disqualifications are flagged
    separately and count as neither finish nor DNF.
    """
    client = client or JolpicaClient()
    rows: list[dict] = []
    for season in seasons:
        for race in client.get_all(f"{season}/results"):
            for res in race.get("Results", []):
                position_text = res["positionText"]
                classified = position_text.isdigit()
                fl_rank = res.get("FastestLap", {}).get("rank")
                rows.append(
                    {
                        "fastest_lap_rank": int(fl_rank) if fl_rank else None,
                        "is_fastest_lap": fl_rank == "1",
                        "season": season,
                        "round": int(race["round"]),
                        "circuit_id": race["Circuit"]["circuitId"],
                        "race_name": race["raceName"],
                        "date": race["date"],
                        "driver_id": res["Driver"]["driverId"],
                        "driver_code": res["Driver"].get("code"),
                        "driver_name": (
                            f"{res['Driver'].get('givenName', '')} "
                            f"{res['Driver'].get('familyName', '')}".strip()
                        ),
                        "constructor_id": res["Constructor"]["constructorId"],
                        "team": canonical_constructor(res["Constructor"]["constructorId"]),
                        "grid": int(res["grid"]),  # 0 = pit-lane start
                        "position": int(res["position"]) if classified else None,
                        "position_text": position_text,
                        "classified": classified,
                        "dnf": position_text == _RETIRED,
                        "disqualified": position_text == _DISQUALIFIED,
                        "status": res["status"],
                        "points": float(res["points"]),
                        "laps": int(res["laps"]),
                    }
                )
    df = pd.DataFrame(rows)
    return df.sort_values(["season", "round", "position_text"]).reset_index(drop=True)


def qualifying_frame(seasons: list[int], client: JolpicaClient | None = None) -> pd.DataFrame:
    """Qualifying classification and best times, one row per (season, round, driver).

    Known at the end of qualifying for race R (post_quali features).
    `quali_delta_pct` is the driver's best time as % over the session best.
    """
    client = client or JolpicaClient()
    rows: list[dict] = []
    for season in seasons:
        for race in client.get_all(f"{season}/qualifying"):
            for q in race.get("QualifyingResults", []):
                times = [
                    time_str_to_seconds(q.get(key)) for key in ("Q1", "Q2", "Q3")
                ]
                times = [t for t in times if t is not None]
                rows.append(
                    {
                        "season": season,
                        "round": int(race["round"]),
                        "driver_id": q["Driver"]["driverId"],
                        "quali_position": int(q["position"]),
                        "best_quali_s": min(times) if times else None,
                    }
                )
    df = pd.DataFrame(rows)
    best = df.groupby(["season", "round"])["best_quali_s"].transform("min")
    df["quali_delta_pct"] = (df["best_quali_s"] / best - 1) * 100
    return df


def build_core_table(seasons: list[int], client: JolpicaClient | None = None) -> pd.DataFrame:
    """Join results and qualifying into the core table and write core.parquet."""
    client = client or JolpicaClient()
    core = results_frame(seasons, client).merge(
        qualifying_frame(seasons, client),
        on=["season", "round", "driver_id"],
        how="left",
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    core.to_parquet(PROCESSED_DIR / "core.parquet", index=False)
    return core
