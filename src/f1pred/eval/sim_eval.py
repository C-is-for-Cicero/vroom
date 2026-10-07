"""Walk-forward evaluation of the full pipeline: pace model + DNF model +
race simulator, against probabilistic baselines.

CLI: python -m f1pred.eval.sim_eval [--mode pre_quali|post_quali|both]
     [--min-train-races 30] [--n-sims 10000]

This is where the log-loss guardrail (docs/DECISIONS.md) is measured: the
simulator's win/podium/points probabilities must beat the baseline's on
log-loss. Baseline orderings (grid / standings) are turned into
probabilities via historical frequencies on the training races only
(eval/baselines.rank_outcome_probs), so both sides are scored the same way.

Per test race the models are refitted on strictly earlier races; the
post_quali simulator's grid-effect weight is refitted the same way.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from f1pred.eval.baselines import (
    apply_rank_probs,
    grid_baseline_rank,
    rank_outcome_probs,
    standings_baseline_rank,
)
from f1pred.eval.metrics import (
    binary_log_loss,
    brier_score,
    per_race_summary,
    spearman_order,
    top3_hit_rate,
)
from f1pred.eval.pace_eval import load_dataset
from f1pred.eval.splits import walk_forward_by_race
from f1pred.features.track_fingerprint import grid_effect_scale
from f1pred.models.dnf import DnfModel
from f1pred.models.pace import POST_QUALI_FEATURES, PRE_QUALI_FEATURES, PaceModel
from f1pred.sim.race_sim import estimate_dnf_frailty, fit_grid_effect, simulate_race

MODES: dict[str, tuple[list[str], str, Callable[[pd.DataFrame], pd.Series]]] = {
    "pre_quali": (PRE_QUALI_FEATURES, "baseline_standings", standings_baseline_rank),
    "post_quali": (POST_QUALI_FEATURES, "baseline_grid", grid_baseline_rank),
}


def _outcomes(test: pd.DataFrame) -> dict[str, np.ndarray]:
    pos = test["position"].astype(float)
    return {
        "win": (pos == 1).to_numpy(dtype=float),
        "podium": (pos <= 3).to_numpy(dtype=float),
        "points": (pos <= 10).to_numpy(dtype=float),
    }


def _prob_metrics(y: dict[str, np.ndarray], p: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("win", "podium", "points"):
        out[f"ll_{key}"] = binary_log_loss(y[key], p[key])
        out[f"brier_{key}"] = brier_score(y[key], p[key])
    return out


# Displayed probabilities are isotonic-calibrated (docs/DECISIONS.md). In
# the walk-forward, race i's calibrator trains on the model's own
# OUT-OF-SAMPLE predictions for races < i, so no leakage. Early races with
# too little history stay raw.
MIN_CALIBRATION_RACES = 10


def _calibrated_rows(records: list[dict], mode: str) -> list[dict]:
    from sklearn.isotonic import IsotonicRegression

    rows: list[dict] = []
    for i, rec in enumerate(records):
        p_cal: dict[str, np.ndarray] = {}
        for key in ("win", "podium", "points"):
            if i < MIN_CALIBRATION_RACES:
                p_cal[key] = rec["p"][key]
                continue
            iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
            iso.fit(
                np.concatenate([r["p"][key] for r in records[:i]]),
                np.concatenate([r["y"][key] for r in records[:i]]),
            )
            p_cal[key] = iso.predict(rec["p"][key])
        rows.append(
            {
                "model": f"sim_{mode}_calibrated",
                "season": rec["season"],
                "round": rec["round"],
                # calibration is monotonic per outcome: order metrics unchanged
                "spearman": rec["spearman"],
                "top3_hit_rate": rec["top3_hit_rate"],
                **_prob_metrics(rec["y"], p_cal),
            }
        )
    return rows


def _odds_rows(odds: pd.DataFrame | None, test: pd.DataFrame, y: dict, season, rnd) -> list[dict]:
    """Bookmaker-implied win baseline for one race, where a pre-race
    snapshot exists. Drivers missing from the market share the leftover
    probability mass equally."""
    if odds is None or odds.empty:
        return []
    o = odds[(odds["season"] == season) & (odds["round"] == rnd)]
    if len(o) < 10:
        return []
    p = test[["driver_id"]].merge(o[["driver_id", "p_win_odds"]], on="driver_id", how="left")[
        "p_win_odds"
    ].to_numpy()
    missing = np.isnan(p)
    leftover = max(1.0 - np.nansum(p), 0.0)
    p[missing] = leftover / missing.sum() if missing.any() else 0.0
    p = np.clip(p / p.sum(), 1e-6, 1 - 1e-6)
    odds_rank = pd.Series(-p, index=test.index).rank(method="first")
    actual = test["position"].astype(float)
    return [
        {
            "model": "baseline_odds",
            "season": season,
            "round": rnd,
            "spearman": spearman_order(odds_rank, actual),
            "top3_hit_rate": top3_hit_rate(odds_rank, actual),
            "ll_win": binary_log_loss(y["win"], p),
            "brier_win": brier_score(y["win"], p),
        }
    ]


def evaluate_sim(
    df: pd.DataFrame,
    mode: str,
    min_train_races: int = 30,
    seed: int = 0,
    n_sims: int = 10_000,
    odds: pd.DataFrame | None = None,
) -> list[dict]:
    """Walk-forward loop; returns per-race metric rows for the raw and
    calibrated simulator, the mode's baseline, and (where snapshots
    exist) the bookmaker odds baseline."""
    features, baseline_name, baseline_rank = MODES[mode]
    rows: list[dict] = []
    records: list[dict] = []
    for i, (train_idx, test_idx, (season, rnd)) in enumerate(
        walk_forward_by_race(df, min_train_races)
    ):
        train, test = df.loc[train_idx], df.loc[test_idx]
        if test["pace_delta_s"].notna().sum() < 5:
            continue

        pace = PaceModel(features=features, seed=seed).fit(train)
        dnf = DnfModel(seed=seed).fit(train)
        mu, sigma = pace.predict(test)
        p_dnf = dnf.predict(test)

        grid = None
        grid_effect = 0.0
        if mode == "post_quali":
            grid = test["grid"].to_numpy(dtype=float)
            mu_train, _ = pace.predict(train)
            # .indices gives positional indices per (season, round) group
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
        sim = simulate_race(
            mu, sigma, p_dnf, grid=grid, grid_effect_s=grid_effect,
            dnf_frailty_var=frailty, n_sims=n_sims, seed=seed * 100_003 + i,
        )
        y = _outcomes(test)
        actual = test["position"].astype(float)
        exp_rank = pd.Series(sim.exp_position, index=test.index).rank(method="first")
        p_sim = {"win": sim.p_win, "podium": sim.p_podium, "points": sim.p_points}
        sim_spearman = spearman_order(exp_rank, actual)
        sim_top3 = top3_hit_rate(exp_rank, actual)
        records.append(
            {
                "season": season,
                "round": rnd,
                "y": y,
                "p": p_sim,
                "spearman": sim_spearman,
                "top3_hit_rate": sim_top3,
            }
        )
        rows.append(
            {
                "model": f"sim_{mode}",
                "season": season,
                "round": rnd,
                "spearman": sim_spearman,
                "top3_hit_rate": sim_top3,
                **_prob_metrics(y, p_sim),
            }
        )

        # Baseline probabilities from train-race frequencies at each rank.
        train_ranks = train.groupby(["season", "round"], sort=False, group_keys=False).apply(
            baseline_rank, include_groups=False
        )
        table = rank_outcome_probs(train, train_ranks)
        bl = apply_rank_probs(table, baseline_rank(test))
        bl_rank = baseline_rank(test)
        rows.append(
            {
                "model": baseline_name,
                "season": season,
                "round": rnd,
                "spearman": spearman_order(bl_rank, actual),
                "top3_hit_rate": top3_hit_rate(bl_rank, actual),
                **_prob_metrics(
                    y,
                    {
                        "win": bl["p_win"].to_numpy(),
                        "podium": bl["p_podium"].to_numpy(),
                        "points": bl["p_points"].to_numpy(),
                    },
                ),
            }
        )
        rows.extend(_odds_rows(odds, test, y, season, rnd))
    rows.extend(_calibrated_rows(records, mode))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.eval.sim_eval", description=__doc__)
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    parser.add_argument("--min-train-races", type=int, default=30)
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from f1pred.ingest.odds import load_odds_table

    df = load_dataset()
    odds = load_odds_table()
    if odds.empty:
        print("(no bookmaker odds snapshots yet - odds baseline column will be empty)")
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    for mode in list(MODES) if args.mode == "both" else [args.mode]:
        rows = evaluate_sim(
            df, mode, min_train_races=args.min_train_races, seed=args.seed,
            n_sims=args.n_sims, odds=odds,
        )
        print(f"\n== {mode} ==")
        print(per_race_summary(rows).to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
