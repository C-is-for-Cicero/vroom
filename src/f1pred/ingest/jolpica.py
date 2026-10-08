"""Jolpica-F1 API client (successor to Ergast; Ergast itself is shut down).

Every HTTP response is cached verbatim as JSON under data/raw/jolpica/ before
anything reads it, so raw pulls are reproducible and the API is hit at most
once per (endpoint, page). Requests are throttled to stay well inside
Jolpica's rate limits.

Leakage note: this module only *stores* data; cutoff enforcement happens in
the feature layer. Files are keyed by endpoint path, so re-ingesting a season
after a race weekend refreshes only via `force=True`.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from f1pred.config import JOLPICA_RAW_DIR

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100
# Jolpica's unauthenticated burst limit is 4 req/s; stay comfortably below
# it. The sustained hourly quota still bites on a full multi-season pull -
# _http_fetch waits through 429s rather than crashing.
MIN_REQUEST_INTERVAL_S = 1.0
MAX_RETRIES = 8
MAX_RETRY_WAIT_S = 900.0
USER_AGENT = "vroom-f1pred/0.1 (hobby F1 prediction project)"

# (MRData table key, list key) per endpoint leaf, for pagination/merging.
TABLE_KEYS: dict[str, tuple[str, str]] = {
    "races": ("RaceTable", "Races"),
    "results": ("RaceTable", "Races"),
    "qualifying": ("RaceTable", "Races"),
    "sprint": ("RaceTable", "Races"),
    "pitstops": ("RaceTable", "Races"),
    "driverstandings": ("StandingsTable", "StandingsLists"),
    "constructorstandings": ("StandingsTable", "StandingsLists"),
    "drivers": ("DriverTable", "Drivers"),
    "constructors": ("ConstructorTable", "Constructors"),
    "circuits": ("CircuitTable", "Circuits"),
}


class JolpicaClient:
    """Cached, throttled access to the Jolpica-F1 REST API.

    Inputs: a cache directory (defaults to data/raw/jolpica) and optionally a
    `fetch` callable (url -> parsed JSON dict) injected for tests. Outputs:
    parsed MRData dicts; `get_all` merges paginated list entries.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        fetch: Callable[[str], dict[str, Any]] | None = None,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else JOLPICA_RAW_DIR
        self._fetch = fetch if fetch is not None else self._http_fetch
        self._min_interval_s = min_interval_s
        self._last_request_at = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT

    # -- low level ----------------------------------------------------------

    def _http_fetch(self, url: str) -> dict[str, Any]:
        """GET with throttling and polite 429 handling.

        On 429 the client waits (honouring Retry-After, capped at
        MAX_RETRY_WAIT_S) and retries up to MAX_RETRIES times, so a long
        ingest pauses through rate limits instead of crashing.
        """
        for attempt in range(MAX_RETRIES):
            wait = self._min_interval_s - (time.monotonic() - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            resp = self._session.get(url, timeout=30)
            self._last_request_at = time.monotonic()
            if resp.status_code == 429:
                try:
                    retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                except ValueError:
                    retry_after = 0.0
                delay = min(max(retry_after, 2.0 ** (attempt + 2)), MAX_RETRY_WAIT_S)
                print(f"jolpica: rate limited (429), waiting {delay:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(
            f"Jolpica still rate-limiting after {MAX_RETRIES} waits for {url} - "
            "re-run later; everything fetched so far is cached"
        )

    def _cache_path(self, endpoint: str, offset: int) -> Path:
        safe = re.sub(r"[^A-Za-z0-9/_-]", "_", endpoint.strip("/"))
        return self.cache_dir / safe / f"offset_{offset:06d}.json"

    def get_page(self, endpoint: str, offset: int = 0, force: bool = False) -> dict[str, Any]:
        """One page of `endpoint` (e.g. "2026/results"), cache-first.

        Returns the parsed MRData envelope exactly as the API sent it.
        """
        path = self._cache_path(endpoint, offset)
        if path.exists() and not force:
            return json.loads(path.read_text())
        url = f"{BASE_URL}/{endpoint.strip('/')}.json?limit={PAGE_LIMIT}&offset={offset}"
        payload = self._fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return payload

    # -- pagination ---------------------------------------------------------

    def get_all(self, endpoint: str, force: bool = False) -> list[dict[str, Any]]:
        """All list entries for `endpoint`, merged across pages.

        The list key is looked up from the endpoint leaf (e.g. ".../results"
        pages hold RaceTable.Races). Raises KeyError for unknown endpoints.
        """
        leaf = endpoint.strip("/").split("/")[-1].lower()
        table_key, list_key = TABLE_KEYS[leaf]
        entries: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self.get_page(endpoint, offset=offset, force=force)
            mrdata = payload["MRData"]
            entries.extend(mrdata[table_key].get(list_key, []))
            total = int(mrdata["total"])
            offset += int(mrdata["limit"])
            if offset >= total:
                return entries


# -- season-level convenience pulls ----------------------------------------


def ingest_season(client: JolpicaClient, season: int, force: bool = False) -> dict[str, int]:
    """Pull a season's schedule, results, quali, sprints, standings, pit stops.

    Returns a summary of entry counts per endpoint. Pit stops are per round,
    so the schedule is pulled first to enumerate rounds.
    """
    counts: dict[str, int] = {}
    races = client.get_all(f"{season}/races", force=force)
    counts["races"] = len(races)
    for leaf in ("results", "qualifying", "sprint"):
        counts[leaf] = len(client.get_all(f"{season}/{leaf}", force=force))
    counts["driverstandings"] = len(client.get_all(f"{season}/driverstandings", force=force))
    counts["constructorstandings"] = len(
        client.get_all(f"{season}/constructorstandings", force=force)
    )
    pitstop_rounds = 0
    for race in races:
        rnd = race["round"]
        stops = client.get_all(f"{season}/{rnd}/pitstops", force=force)
        if stops:
            pitstop_rounds += 1
    counts["pitstop_rounds"] = pitstop_rounds
    return counts
