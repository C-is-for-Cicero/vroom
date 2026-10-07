"""Championship simulator.

Starts from the current official standings (Jolpica, so sprint points are
included), simulates every remaining race with the race simulator in
pre_quali mode (no grids exist for future rounds), applies the points
system from config, and counts title outcomes per simulation run.

v1 approximations, documented:
- form features are frozen at today for all remaining rounds (only
  circuit-dependent features vary per round);
- remaining sprint races are not simulated (race points only);
- drivers in the standings but no longer racing keep their points.

CLI: python -m f1pred.sim.championship --season 2026
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from f1pred.config import (
    N_SIMS_DEFAULT,
    PROCESSED_DIR,
    SIM_SEED_DEFAULT,
    canonical_constructor,
    race_points,
)
from f1pred.ingest.jolpica import JolpicaClient
from f1pred.models.dnf import DnfModel
from f1pred.models.pace import PRE_QUALI_FEATURES, PaceModel
from f1pred.predict import build_prediction_frame
from f1pred.sim.race_sim import simulate_race

CHAMPIONSHIP_DIR = PROCESSED_DIR / "predictions"


def _current_standings(client: JolpicaClient, season: int) -> tuple[dict, dict]:
    """Official (driver_points, team_points) after the latest round."""
    driver_lists = client.get_all(f"{season}/driverstandings")
    drivers = {
        s["Driver"]["driverId"]: float(s["points"]) for s in driver_lists[-1]["DriverStandings"]
    }
    team_lists = client.get_all(f"{season}/constructorstandings")
    teams = {
        canonical_constructor(s["Constructor"]["constructorId"]): float(s["points"])
        for s in team_lists[-1]["ConstructorStandings"]
    }
    return drivers, teams


def simulate_championship(
    season: int, n_sims: int = N_SIMS_DEFAULT, seed: int = SIM_SEED_DEFAULT
) -> dict:
    """Simulate the rest of `season`; returns driver and team title tables."""
    client = JolpicaClient()
    schedule_rounds = sorted(int(r["round"]) for r in client.get_all(f"{season}/races"))
    core = pd.read_parquet(PROCESSED_DIR / "core.parquet")
    done_rounds = set(core.loc[core["season"] == season, "round"].astype(int))
    remaining = [r for r in schedule_rounds if r not in done_rounds]

    cur_driver_pts, cur_team_pts = _current_standings(client, season)
    rng = np.random.default_rng(seed)

    sim_driver_pts: dict[str, np.ndarray] = {}
    driver_team: dict[str, str] = {}
    races_simulated = []
    for rnd in remaining:
        df = build_prediction_frame(season, rnd, "pre_quali")
        target = (df["season"] == season) & (df["round"] == rnd)
        train, test = df[~target], df[target]
        pace = PaceModel(features=PRE_QUALI_FEATURES, seed=seed).fit(train)
        dnf = DnfModel(seed=seed).fit(train)
        mu, sigma = pace.predict(test)
        p_dnf = dnf.predict(test)
        sim = simulate_race(
            mu, sigma, p_dnf, n_sims=n_sims, seed=seed * 1009 + rnd, return_ranks=True
        )
        n = len(test)
        pts_by_rank = np.array([race_points(p) for p in range(1, n + 1)], dtype=float)
        pts = pts_by_rank[sim.ranks]  # (n_sims, n_drivers)
        for j, (drv, team) in enumerate(
            zip(test["driver_id"].values, test["team"].values, strict=True)
        ):
            sim_driver_pts.setdefault(drv, np.zeros(n_sims))
            sim_driver_pts[drv] += pts[:, j]
            driver_team[drv] = team
        races_simulated.append(int(rnd))

    # totals per sim: everyone in the standings participates in the title
    all_drivers = sorted(set(cur_driver_pts) | set(sim_driver_pts))
    totals = np.zeros((n_sims, len(all_drivers)))
    for j, drv in enumerate(all_drivers):
        totals[:, j] = cur_driver_pts.get(drv, 0.0) + sim_driver_pts.get(drv, 0.0)
    totals += rng.random(totals.shape) * 1e-6  # break exact ties randomly
    champion = np.argmax(totals, axis=1)
    driver_table = pd.DataFrame(
        {
            "driver_id": all_drivers,
            "current_points": [cur_driver_pts.get(d, 0.0) for d in all_drivers],
            "exp_final_points": np.round(totals.mean(axis=0), 1),
            "p_title": np.round(np.bincount(champion, minlength=len(all_drivers)) / n_sims, 4),
        }
    ).sort_values("p_title", ascending=False)

    all_teams = sorted(
        set(cur_team_pts) | {driver_team[d] for d in sim_driver_pts}
    )
    team_totals = np.zeros((n_sims, len(all_teams)))
    for j, team in enumerate(all_teams):
        team_totals[:, j] = cur_team_pts.get(team, 0.0)
        for drv, pts_arr in sim_driver_pts.items():
            if driver_team[drv] == team:
                team_totals[:, j] += pts_arr
    team_totals += rng.random(team_totals.shape) * 1e-6
    team_champion = np.argmax(team_totals, axis=1)
    team_table = pd.DataFrame(
        {
            "team": all_teams,
            "current_points": [cur_team_pts.get(t, 0.0) for t in all_teams],
            "exp_final_points": np.round(team_totals.mean(axis=0), 1),
            "p_title": np.round(
                np.bincount(team_champion, minlength=len(all_teams)) / n_sims, 4
            ),
        }
    ).sort_values("p_title", ascending=False)

    return {
        "meta": {
            "season": season,
            "rounds_simulated": races_simulated,
            "n_sims": n_sims,
            "seed": seed,
            "mode": "pre_quali",
            "sprints_simulated": False,
        },
        "drivers": driver_table,
        "teams": team_table,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.sim.championship", description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--n-sims", type=int, default=N_SIMS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SIM_SEED_DEFAULT)
    args = parser.parse_args(argv)

    result = simulate_championship(args.season, n_sims=args.n_sims, seed=args.seed)
    meta = result["meta"]
    print(f"\n{args.season} championship, {len(meta['rounds_simulated'])} rounds remaining "
          f"({meta['rounds_simulated']}), {meta['n_sims']} sims\n")
    print("Drivers' title:")
    print(result["drivers"].head(10).to_string(index=False))
    print("\nConstructors' title:")
    print(result["teams"].to_string(index=False))

    CHAMPIONSHIP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHAMPIONSHIP_DIR / f"{args.season}_championship.json"
    out_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "drivers": result["drivers"].to_dict(orient="records"),
                "teams": result["teams"].to_dict(orient="records"),
            },
            indent=2,
        )
    )
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
