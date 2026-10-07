"""CLI: python -m f1pred.features --seasons 2018-2026 [--race-pace]

Rebuilds data/processed/ feature tables from the raw pulls:
core.parquet (results + quali + form) and, with --race-pace, the per-season
race-pace target parquets (needs FastF1 downloads for races not yet cached).
"""

from __future__ import annotations

import argparse

from f1pred.config import CURRENT_SEASON, FIRST_DETAILED_SEASON, PROCESSED_DIR
from f1pred.features.base import build_core_table
from f1pred.features.form import add_form_features
from f1pred.features.track_fingerprint import add_track_features


def parse_seasons(spec: str) -> list[int]:
    """Parse "2024", "2018-2026", or "2022,2024" into a season list."""
    seasons: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seasons.update(range(int(lo), int(hi) + 1))
        else:
            seasons.add(int(part))
    return sorted(seasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.features", description=__doc__)
    parser.add_argument("--seasons", default=f"{FIRST_DETAILED_SEASON}-{CURRENT_SEASON}")
    parser.add_argument(
        "--race-pace", action="store_true", help="also build the FastF1 race-pace target"
    )
    args = parser.parse_args(argv)
    seasons = parse_seasons(args.seasons)

    core = build_core_table(seasons)
    core = add_form_features(core)
    core = add_track_features(core)
    core.to_parquet(PROCESSED_DIR / "core.parquet", index=False)
    print(f"core.parquet: {len(core)} rows, seasons {seasons[0]}-{seasons[-1]}")

    if args.race_pace:
        from f1pred.features.race_pace import build_race_pace

        for season in seasons:
            rounds = sorted(core.loc[core["season"] == season, "round"].unique())
            built = build_race_pace(season, [int(r) for r in rounds])
            print(f"race_pace {season}: {0 if built.empty else built['round'].nunique()} rounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
