"""FastF1 session loading with the on-disk cache always enabled.

FastF1 covers 2018+ (laps, sectors, tyres, stints, weather, telemetry,
circuit corners). Telemetry is ~4 Hz — downstream code aggregates per lap
segment, never per sample.

Leakage note: a session's data exists only after that session has run;
feature code must only load sessions that precede its prediction cutoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from f1pred.config import FASTF1_CACHE_DIR

if TYPE_CHECKING:  # pragma: no cover - typing only
    import fastf1

# Session codes in weekend order for conventional and sprint weekends.
CONVENTIONAL_SESSIONS = ("FP1", "FP2", "FP3", "Q", "R")
SPRINT_SESSIONS = ("FP1", "SQ", "S", "Q", "R")


def _enable_cache() -> None:
    import fastf1

    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))


def load_session(
    season: int,
    round_number: int,
    session_code: str,
    laps: bool = True,
    telemetry: bool = False,
    weather: bool = True,
    messages: bool = False,
) -> fastf1.core.Session:
    """Load one session with the FastF1 cache enabled; returns it loaded.

    Inputs: season, round number, and a FastF1 session code ("FP1".."FP3",
    "Q", "SQ", "S", "R"); flags choose what to download. Output: the loaded
    `fastf1.core.Session`. Raises whatever FastF1 raises for sessions that
    don't exist or haven't run yet.
    """
    import fastf1

    _enable_cache()
    session = fastf1.get_session(season, round_number, session_code)
    session.load(laps=laps, telemetry=telemetry, weather=weather, messages=messages)
    return session


def ingest_weekend(season: int, round_number: int, telemetry: bool = False) -> list[str]:
    """Cache every session of a race weekend that has already run.

    Tries sprint-format sessions as well; sessions that don't exist or
    haven't happened yet are skipped. Returns the session codes loaded.
    """
    loaded: list[str] = []
    for code in dict.fromkeys(CONVENTIONAL_SESSIONS + SPRINT_SESSIONS):
        try:
            load_session(season, round_number, code, telemetry=telemetry)
        except Exception:
            continue
        loaded.append(code)
    return loaded
