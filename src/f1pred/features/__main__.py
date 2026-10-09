"""CLI: python -m f1pred.features --seasons 2018-2026 [--race-pace]

Rebuilds data/processed/ feature tables from the raw pulls:
core.parquet (results + quali + form + track), and with --race-pace the
FastF1-derived tables (race-pace targets and FP long runs; downloads races
not yet cached). FastF1 enforces an hourly API budget, so lap data is only
pulled from --race-pace-since (default 2023, the span the models use);
builders stop cleanly when the budget runs out - re-run later to resume.
"""

from __future__ import annotations

import argparse

from f1pred.config import (
    CURRENT_SEASON,
    FIRST_DETAILED_SEASON,
    FIRST_LAP_DATA_SEASON,
    PROCESSED_DIR,
)
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
        "--race-pace", action="store_true",
        help="also build the FastF1 race-pace targets and FP long runs",
    )
    parser.add_argument(
        "--race-pace-since", type=int, default=FIRST_LAP_DATA_SEASON,
        help="first season to pull FastF1 lap data for (earlier seasons cost "
             "API budget without being used by the current models)",
    )
    parser.add_argument(
        "--telemetry", action="store_true",
        help="also build telemetry car profiles + track fingerprints "
             "(heavy: downloads FP telemetry, ~50-150MB per weekend)",
    )
    args = parser.parse_args(argv)
    seasons = parse_seasons(args.seasons)

    core = build_core_table(seasons)
    core = add_form_features(core)
    core = add_track_features(core)
    core.to_parquet(PROCESSED_DIR / "core.parquet", index=False)
    print(f"core.parquet: {len(core)} rows, seasons {seasons[0]}-{seasons[-1]}")

    if args.race_pace:
        from f1pred.features.car_profile import build_fp_longrun
        from f1pred.features.race_pace import build_race_pace
        from f1pred.ingest.fastf1_loader import FastF1BudgetExhausted

        lap_seasons = [s for s in seasons if s >= args.race_pace_since]
        try:
            for season in lap_seasons:
                rounds = [
                    int(r) for r in sorted(core.loc[core["season"] == season, "round"].unique())
                ]
                built = build_race_pace(season, rounds)
                print(f"race_pace {season}: "
                      f"{0 if built.empty else built['round'].nunique()} rounds")
            for season in lap_seasons:
                rounds = [
                    int(r) for r in sorted(core.loc[core["season"] == season, "round"].unique())
                ]
                built = build_fp_longrun(season, rounds)
                print(f"fp_longrun {season}: "
                      f"{0 if built.empty else built['round'].nunique()} rounds")
            if args.telemetry:
                from f1pred.features.car_profile import build_telemetry_features

                for season in lap_seasons:
                    rounds = [
                        int(r)
                        for r in sorted(core.loc[core["season"] == season, "round"].unique())
                    ]
                    built = build_telemetry_features(season, rounds)
                    print(f"telemetry {season}: "
                          f"{0 if built.empty else built['round'].nunique()} rounds")
        except FastF1BudgetExhausted as exc:
            print(f"\nFastF1 hourly API budget exhausted ({exc}).")
            print("All progress so far is saved - run this exact command again in "
                  "about an hour to continue where it left off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
