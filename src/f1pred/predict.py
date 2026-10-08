"""CLI: python -m f1pred.predict --season 2026 --round 20 --mode pre_quali

Trains on every race strictly before the target round, simulates the race
10,000 times, and prints/saves per-driver probabilities.

Entry list for the target round:
- round already raced: its real entry list (retrodiction);
- upcoming round, post_quali: the Jolpica qualifying classification
  (grid = quali position; penalties unknown);
- upcoming round, pre_quali: the previous round's drivers/teams.

Form features for upcoming rounds are computed by appending placeholder
rows and re-running the (shift-based) feature builders, so race R only sees
races before R, exactly as in training. Probabilities are the simulator's
raw output (isotonic calibration is wired into the webapp layer later).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from f1pred.config import N_SIMS_DEFAULT, PROCESSED_DIR, SIM_SEED_DEFAULT, canonical_constructor
from f1pred.features.base import time_str_to_seconds
from f1pred.features.car_profile import load_fp_longrun
from f1pred.features.form import add_form_features
from f1pred.features.race_pace import load_race_pace
from f1pred.features.track_fingerprint import add_track_features, grid_effect_scale
from f1pred.ingest.jolpica import JolpicaClient
from f1pred.models.dnf import DnfModel
from f1pred.models.pace import (
    POST_QUALI_FEATURES,
    PRE_QUALI_FEATURES,
    PaceModel,
    add_pace_form_features,
)
from f1pred.sim.race_sim import estimate_dnf_frailty, fit_grid_effect, simulate_race

PREDICTIONS_DIR = PROCESSED_DIR / "predictions"

# Columns of core.parquet that come from ingestion (not derived by the
# feature builders); placeholders are appended at this level.
_BASE_COLUMNS = [
    "season", "round", "circuit_id", "race_name", "date", "driver_id", "driver_code",
    "constructor_id", "team", "grid", "position", "position_text", "classified",
    "dnf", "disqualified", "status", "points", "laps",
    "quali_position", "best_quali_s", "quali_delta_pct",
]


def _base_core() -> pd.DataFrame:
    core = pd.read_parquet(PROCESSED_DIR / "core.parquet")
    return core[[c for c in _BASE_COLUMNS if c in core.columns]].copy()


def _placeholder_entries(
    base: pd.DataFrame, season: int, round_number: int, mode: str, client: JolpicaClient
) -> pd.DataFrame:
    """Placeholder rows for an upcoming round, at base-column level."""
    races = client.get_all(f"{season}/races")
    circuit = next(
        (r for r in races if int(r["round"]) == round_number), None
    )
    if circuit is None:
        raise SystemExit(f"round {round_number} not found in the {season} schedule")

    prior = base[
        (base["season"] < season)
        | ((base["season"] == season) & (base["round"] < round_number))
    ]
    if prior.empty:
        raise SystemExit("no historical data before the requested round")
    last_key = prior[["season", "round"]].iloc[-1]
    latest = prior[(prior["season"] == last_key["season"]) & (prior["round"] == last_key["round"])]
    identity = latest.set_index("driver_id")[["driver_code", "constructor_id", "team"]]

    rows = []
    if mode == "post_quali":
        quali = [
            q
            for race in client.get_all(f"{season}/qualifying")
            if int(race["round"]) == round_number
            for q in race.get("QualifyingResults", [])
        ]
        if not quali:
            raise SystemExit(
                f"no qualifying data for {season} round {round_number} yet - "
                "use --mode pre_quali (or re-ingest with --force after quali)"
            )
        times = {
            q["Driver"]["driverId"]: [
                t for t in (time_str_to_seconds(q.get(k)) for k in ("Q1", "Q2", "Q3")) if t
            ]
            for q in quali
        }
        best_overall = min(min(t) for t in times.values() if t)
        for q in quali:
            drv = q["Driver"]["driverId"]
            best = min(times[drv]) if times[drv] else None
            constructor = q["Constructor"]["constructorId"]
            rows.append(
                {
                    "driver_id": drv,
                    "driver_code": q["Driver"].get("code"),
                    "constructor_id": constructor,
                    "team": canonical_constructor(constructor),
                    "grid": int(q["position"]),
                    "quali_position": int(q["position"]),
                    "best_quali_s": best,
                    "quali_delta_pct": (best / best_overall - 1) * 100 if best else None,
                }
            )
    else:
        for drv, ident in identity.iterrows():
            rows.append(
                {
                    "driver_id": drv,
                    "driver_code": ident["driver_code"],
                    "constructor_id": ident["constructor_id"],
                    "team": ident["team"],
                    "grid": 0,
                }
            )

    entries = pd.DataFrame(rows)
    entries["season"] = season
    entries["round"] = round_number
    entries["circuit_id"] = circuit["Circuit"]["circuitId"]
    entries["race_name"] = circuit["raceName"]
    entries["date"] = circuit.get("date")
    entries["dnf"] = False
    entries["disqualified"] = False
    entries["classified"] = False
    entries["points"] = np.nan
    entries["position"] = np.nan
    return entries


def build_prediction_frame(season: int, round_number: int, mode: str) -> pd.DataFrame:
    """Full feature frame (history + target round) with target-round rows last.

    Leakage: identical to training — every feature for the target round is a
    shifted aggregate of earlier races, plus same-weekend quali in post mode.
    """
    client = JolpicaClient()
    base = _base_core().sort_values(["season", "round"], kind="stable")
    key = (season, round_number)
    is_target = (base["season"] == season) & (base["round"] == round_number)
    base = base[
        (base["season"] < season)
        | ((base["season"] == season) & (base["round"] <= round_number))
    ]
    if not is_target.any():
        base = pd.concat(
            [base, _placeholder_entries(base, season, round_number, mode, client)],
            ignore_index=True,
        )

    df = add_form_features(base)
    df = add_track_features(df)
    seasons = sorted(df["season"].unique())
    df = df.merge(load_race_pace(seasons), on=["season", "round", "driver_code"], how="left")
    df = df.merge(load_fp_longrun(seasons), on=["season", "round", "driver_code"], how="left")
    # the target race's own pace outcome must never be visible
    target_mask = (df["season"] == key[0]) & (df["round"] == key[1])
    df.loc[target_mask, "pace_delta_s"] = np.nan
    return add_pace_form_features(df)


def recent_results(history: pd.DataFrame, driver_id: str, n: int = 5) -> list[dict]:
    """The driver's last `n` race results before the target round, oldest
    first: {label, position (None for DNF/unclassified), dnf}. Feeds the
    'predicted vs previous' chart on the race page."""
    rows = history[history["driver_id"] == driver_id].sort_values(
        ["season", "round"], kind="stable"
    ).tail(n)
    return [
        {
            "label": f"{int(r.season) % 100}R{int(r.round)}",
            "position": int(r.position) if pd.notna(r.position) else None,
            "dnf": bool(r.dnf),
        }
        for r in rows.itertuples()
    ]


def predict_round(
    season: int,
    round_number: int,
    mode: str,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = SIM_SEED_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Train on strictly earlier races and simulate the target round."""
    features = PRE_QUALI_FEATURES if mode == "pre_quali" else POST_QUALI_FEATURES
    df = build_prediction_frame(season, round_number, mode)
    target_mask = (df["season"] == season) & (df["round"] == round_number)
    train, test = df[~target_mask], df[target_mask]

    pace = PaceModel(features=features, seed=seed).fit(train)
    dnf = DnfModel(seed=seed).fit(train)
    mu, sigma = pace.predict(test)
    p_dnf = dnf.predict(test)

    grid, grid_effect = None, 0.0
    if mode == "post_quali":
        grid = test["grid"].to_numpy(dtype=float)
        mu_train, _ = pace.predict(train)
        per_race = list(train.groupby(["season", "round"], sort=False).indices.values())
        grid_arr = train["grid"].to_numpy(dtype=float)
        pos_arr = train["position"].astype(float).to_numpy()
        grid_effect = fit_grid_effect(
            [mu_train[ix] for ix in per_race],
            [grid_arr[ix] for ix in per_race],
            [pos_arr[ix] for ix in per_race],
        ) * grid_effect_scale(test, train)

    frailty = estimate_dnf_frailty(
        train.groupby(["season", "round"], sort=False)["dnf"].sum().to_numpy()
    )
    sim = simulate_race(mu, sigma, p_dnf, grid=grid, grid_effect_s=grid_effect,
                        dnf_frailty_var=frailty, n_sims=n_sims, seed=seed)
    out = pd.DataFrame(
        {
            "driver_id": test["driver_id"].values,
            "driver_code": test["driver_code"].values,
            "team": test["team"].values,
            "grid": test["grid"].values,
            "pace_delta_s": np.round(mu, 3),
            "p_dnf": np.round(p_dnf, 3),
            "p_win": np.round(sim.p_win, 4),
            "p_podium": np.round(sim.p_podium, 4),
            "p_points": np.round(sim.p_points, 4),
            "exp_position": np.round(sim.exp_position, 2),
            # per-driver extras for the webapp's charts
            "pos_probs": [list(np.round(row, 4)) for row in sim.pos_probs],
            "recent": [recent_results(train, d) for d in test["driver_id"]],
        }
    ).sort_values("exp_position")
    meta = {
        "season": season,
        "round": round_number,
        "race_name": str(test["race_name"].iloc[0]),
        "mode": mode,
        "n_sims": n_sims,
        "seed": seed,
        "grid_effect_s": grid_effect,
        "trained_races": int(train[["season", "round"]].drop_duplicates().shape[0]),
        "calibrated": False,
    }
    return out, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.predict", description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--mode", choices=["pre_quali", "post_quali"], required=True)
    parser.add_argument("--n-sims", type=int, default=N_SIMS_DEFAULT)
    parser.add_argument("--seed", type=int, default=SIM_SEED_DEFAULT)
    args = parser.parse_args(argv)

    table, meta = predict_round(args.season, args.round, args.mode, args.n_sims, args.seed)
    print(f"\n{meta['race_name']} {args.season} (round {args.round}, {args.mode}, "
          f"{meta['trained_races']} training races)\n")
    print(table.drop(columns=["pos_probs", "recent"]).to_string(index=False))

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREDICTIONS_DIR / f"{args.season}_{args.round:02d}_{args.mode}.json"
    out_path.write_text(
        json.dumps({"meta": meta, "drivers": table.to_dict(orient="records")}, indent=2)
    )
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
