"""CLI: python -m f1pred.ingest --season 2026 [--fastf1] [--force]

Pulls/updates a season's Jolpica data into data/raw/jolpica, and optionally
caches FastF1 sessions for every round that has run.
"""

from __future__ import annotations

import argparse

from f1pred.config import CURRENT_SEASON
from f1pred.ingest.jolpica import JolpicaClient, ingest_season


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.ingest", description=__doc__)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument(
        "--force", action="store_true", help="re-fetch pages even if cached (use after a race)"
    )
    parser.add_argument(
        "--fastf1", action="store_true", help="also cache FastF1 sessions for completed rounds"
    )
    parser.add_argument(
        "--telemetry", action="store_true", help="with --fastf1, also download telemetry"
    )
    args = parser.parse_args(argv)

    client = JolpicaClient()
    counts = ingest_season(client, args.season, force=args.force)
    print(f"Jolpica {args.season}:")
    for endpoint, n in counts.items():
        print(f"  {endpoint:22s} {n}")

    if args.fastf1:
        from f1pred.ingest.fastf1_loader import ingest_weekend

        races = client.get_all(f"{args.season}/results")
        rounds = sorted({int(r["round"]) for r in races})
        for rnd in rounds:
            loaded = ingest_weekend(args.season, rnd, telemetry=args.telemetry)
            print(f"FastF1 {args.season} round {rnd:2d}: {', '.join(loaded) or 'nothing loaded'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
