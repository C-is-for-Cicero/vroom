"""Bookmaker odds via The Odds API (the-odds-api.com) — the odds baseline.

The free tier serves CURRENT odds only (no historical backfill), so this
module snapshots upcoming F1 race-winner markets and accumulates them:
run `python -m f1pred.ingest.odds` before each race weekend (cron on the
VPS) and the eval's odds-baseline column fills in as races complete.

Raw responses are stored untouched under data/raw/odds/; the processed
table (one row per season/round/driver with a consensus implied win
probability) is rebuilt from them.

De-vigging: per bookmaker, implied probabilities 1/price are normalised to
sum to 1 across the market (removes the overround); the consensus is the
mean over bookmakers. CLI: --snapshot (needs ODDS_API_KEY), --build.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from f1pred.config import PROCESSED_DIR, RAW_DIR

BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_RAW_DIR = RAW_DIR / "odds"
ODDS_TABLE_PATH = PROCESSED_DIR / "odds_win.parquet"

# a snapshot counts as pre-race only if taken before the race date;
# events are matched to schedule rounds within this window
MATCH_WINDOW_DAYS = 5


def normalise_name(name: str) -> str:
    """Accent-stripped, lowercased 'given family' for matching."""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


class OddsClient:
    """Thin client; `fetch` is injectable for tests."""

    def __init__(self, api_key: str | None = None, fetch=None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("ODDS_API_KEY", "")
        self._fetch = fetch if fetch is not None else self._http_fetch

    def _http_fetch(self, path: str, params: dict[str, str]) -> Any:
        resp = requests.get(
            f"{BASE_URL}/{path}", params={**params, "apiKey": self.api_key}, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def f1_sport_keys(self) -> list[str]:
        """Sport keys for Formula 1 (discovered, not hardcoded)."""
        sports = self._fetch("sports", {"all": "true"})
        return [
            s["key"]
            for s in sports
            if "formula 1" in (s.get("description", "") + " " + s.get("title", "")).lower()
        ]

    def snapshot(self) -> int:
        """Store the current F1 winner-market odds; returns events saved."""
        if not self.api_key:
            raise SystemExit("ODDS_API_KEY is not set - add it to the environment/.env")
        ODDS_RAW_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        n_events = 0
        for key in self.f1_sport_keys():
            events = self._fetch(
                f"sports/{key}/odds",
                {"regions": "eu,uk", "markets": "outrights", "oddsFormat": "decimal"},
            )
            if events:
                path = ODDS_RAW_DIR / f"{stamp}_{key}.json"
                path.write_text(json.dumps({"fetched_at": stamp, "events": events}))
                n_events += len(events)
            time.sleep(1)
        return n_events


# -- processing --------------------------------------------------------------


def devig_event(event: dict) -> pd.DataFrame:
    """Consensus implied win probability per outcome for one event.

    Per bookmaker: p = (1/price) / sum(1/price) over the market, then the
    mean across bookmakers. Output columns: outcome_name, p_win_odds.
    """
    per_bookmaker: list[pd.Series] = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            prices = {
                o["name"]: float(o["price"])
                for o in market.get("outcomes", [])
                if float(o.get("price", 0)) > 1.0
            }
            if len(prices) < 5:
                continue
            inv = pd.Series({k: 1.0 / v for k, v in prices.items()})
            per_bookmaker.append(inv / inv.sum())
    if not per_bookmaker:
        return pd.DataFrame(columns=["outcome_name", "p_win_odds"])
    consensus = pd.concat(per_bookmaker, axis=1).mean(axis=1)
    consensus = consensus / consensus.sum()
    return consensus.rename("p_win_odds").rename_axis("outcome_name").reset_index()


def _schedule(seasons: list[int]) -> pd.DataFrame:
    from f1pred.ingest.jolpica import JolpicaClient

    client = JolpicaClient()
    rows = []
    for season in seasons:
        for race in client.get_all(f"{season}/races"):
            rows.append(
                {
                    "season": season,
                    "round": int(race["round"]),
                    "race_date": pd.Timestamp(race["date"], tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def build_odds_table(seasons: list[int], raw_dir: Path | None = None) -> pd.DataFrame:
    """Rebuild data/processed/odds_win.parquet from raw snapshots.

    Maps each event to a schedule round by commence time (within
    MATCH_WINDOW_DAYS before the race date), keeps the LATEST pre-race
    snapshot per round, and maps outcome names to driver_ids via the core
    table's driver names. Unmatched outcomes are dropped with a notice.
    """
    raw_dir = raw_dir if raw_dir is not None else ODDS_RAW_DIR
    core = pd.read_parquet(PROCESSED_DIR / "core.parquet")
    name_map = {
        normalise_name(n): d
        for n, d in core.dropna(subset=["driver_name"])[["driver_name", "driver_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    schedule = _schedule(seasons)

    frames = []
    for path in sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []:
        payload = json.loads(path.read_text())
        fetched_at = pd.Timestamp(
            datetime.strptime(payload["fetched_at"], "%Y%m%dT%H%M%SZ"), tz="UTC"
        )
        for event in payload["events"]:
            commence = pd.Timestamp(event["commence_time"])
            if commence.tzinfo is None:
                commence = commence.tz_localize("UTC")
            gap = (schedule["race_date"] - commence).dt.total_seconds().abs()
            near = schedule[gap <= MATCH_WINDOW_DAYS * 86_400]
            if near.empty:
                continue
            race = near.iloc[gap.loc[near.index].argmin()]
            if fetched_at >= race["race_date"]:
                continue  # not a pre-race snapshot
            probs = devig_event(event)
            if probs.empty:
                continue
            probs["driver_id"] = probs["outcome_name"].map(
                lambda n: name_map.get(normalise_name(n))
            )
            unmatched = probs[probs["driver_id"].isna()]["outcome_name"].tolist()
            if unmatched:
                print(f"odds {path.name}: unmatched outcomes {unmatched}")
            probs = probs.dropna(subset=["driver_id"])
            probs["season"] = int(race["season"])
            probs["round"] = int(race["round"])
            probs["snapshot_time"] = fetched_at
            frames.append(probs)

    if not frames:
        out = pd.DataFrame(
            columns=["season", "round", "driver_id", "p_win_odds", "snapshot_time"]
        )
    else:
        out = pd.concat(frames, ignore_index=True)
        # latest pre-race snapshot per round wins
        out = (
            out.sort_values("snapshot_time")
            .groupby(["season", "round", "driver_id"], as_index=False)
            .last()
        )[["season", "round", "driver_id", "p_win_odds", "snapshot_time"]]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ODDS_TABLE_PATH, index=False)
    return out


def load_odds_table() -> pd.DataFrame:
    if ODDS_TABLE_PATH.exists():
        return pd.read_parquet(ODDS_TABLE_PATH)
    return pd.DataFrame(columns=["season", "round", "driver_id", "p_win_odds", "snapshot_time"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.ingest.odds", description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="fetch current odds (needs key)")
    parser.add_argument("--build", action="store_true", help="rebuild the processed odds table")
    parser.add_argument("--seasons", default="2025-2026")
    args = parser.parse_args(argv)

    from f1pred.features.__main__ import parse_seasons

    if args.snapshot:
        n = OddsClient().snapshot()
        print(f"snapshot: {n} events stored under {ODDS_RAW_DIR}")
    if args.build or not args.snapshot:
        table = build_odds_table(parse_seasons(args.seasons))
        rounds = table[["season", "round"]].drop_duplicates()
        print(f"odds_win.parquet: {len(table)} rows over {len(rounds)} rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
