"""Walk-forward evaluation of the pace models against their naive baselines.

CLI: python -m f1pred.eval.pace_eval [--mode pre_quali|post_quali|both]
     [--min-train-races 30]

Modes (separate models, per CLAUDE.md):
- pre_quali (the primary product): features known before qualifying;
  baseline = championship-standings order.
- post_quali: adds grid and quali pace; baseline = grid order.

For each test race (chronological, after a minimum training history) the
model is refitted on all earlier races only. Reported per model: mean
Spearman of predicted vs actual finishing order (classified finishers),
top-3 hit rate, and RMSE of the pace prediction on rows with an observed
target. Log-loss/Brier columns join the table in build-order step 3, when
the simulator turns pace+sigma into probabilities; the odds-baseline column
joins when odds ingestion lands.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np
import pandas as pd

from f1pred.config import PROCESSED_DIR
from f1pred.eval.baselines import grid_baseline_rank, standings_baseline_rank
from f1pred.eval.metrics import per_race_summary, spearman_order, top3_hit_rate
from f1pred.eval.splits import walk_forward_by_race
from f1pred.features.race_pace import load_race_pace
from f1pred.models.pace import (
    POST_QUALI_FEATURES,
    PRE_QUALI_FEATURES,
    PaceModel,
    add_pace_form_features,
)

# mode -> (model features, baseline name, baseline rank function)
MODES: dict[str, tuple[list[str], str, Callable[[pd.DataFrame], pd.Series]]] = {
    "pre_quali": (PRE_QUALI_FEATURES, "baseline_standings", standings_baseline_rank),
    "post_quali": (POST_QUALI_FEATURES, "baseline_grid", grid_baseline_rank),
}


def load_dataset() -> pd.DataFrame:
    """Core table + race-pace target + FP long runs, restricted to seasons
    with targets."""
    from f1pred.features.car_profile import load_fp_longrun

    core = pd.read_parquet(PROCESSED_DIR / "core.parquet")
    seasons = sorted(core["season"].unique())
    df = core.merge(load_race_pace(seasons), on=["season", "round", "driver_code"], how="left")
    df = df.merge(load_fp_longrun(seasons), on=["season", "round", "driver_code"], how="left")
    df = add_pace_form_features(df)
    seasons_with_target = sorted(df.loc[df["pace_delta_s"].notna(), "season"].unique())
    return df[df["season"].isin(seasons_with_target)].reset_index(drop=True)


def evaluate(
    df: pd.DataFrame, mode: str, min_train_races: int = 30, seed: int = 0
) -> list[dict]:
    """Run the walk-forward loop for one mode; returns per-race metric rows."""
    features, baseline_name, baseline_rank = MODES[mode]
    rows: list[dict] = []
    for train_idx, test_idx, (season, rnd) in walk_forward_by_race(df, min_train_races):
        train, test = df.loc[train_idx], df.loc[test_idx]
        if test["pace_delta_s"].notna().sum() < 5:
            continue  # race without usable target (e.g. not yet run)
        model = PaceModel(features=features, seed=seed).fit(train)
        mu, _ = model.predict(test)
        pred_rank = pd.Series(mu, index=test.index).rank(method="first")
        actual = test["position"].astype(float)

        labeled = test["pace_delta_s"].notna()
        rmse = float(
            np.sqrt(np.mean((mu[labeled.to_numpy()] - test.loc[labeled, "pace_delta_s"]) ** 2))
        )
        rows.append(
            {
                "model": f"pace_model_{mode}",
                "season": season,
                "round": rnd,
                "spearman": spearman_order(pred_rank, actual),
                "top3_hit_rate": top3_hit_rate(pred_rank, actual),
                "pace_rmse_s": rmse,
            }
        )
        rows.append(
            {
                "model": baseline_name,
                "season": season,
                "round": rnd,
                "spearman": spearman_order(baseline_rank(test), actual),
                "top3_hit_rate": top3_hit_rate(baseline_rank(test), actual),
                "pace_rmse_s": float("nan"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.eval.pace_eval", description=__doc__)
    parser.add_argument("--mode", choices=[*MODES, "both"], default="both")
    parser.add_argument("--min-train-races", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = load_dataset()
    n_target = int(df["pace_delta_s"].notna().sum())
    races = df.loc[df["pace_delta_s"].notna(), ["season", "round"]].drop_duplicates()
    print(f"dataset: {len(df)} rows, {n_target} with pace target over {len(races)} races")

    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    modes = list(MODES) if args.mode == "both" else [args.mode]
    for mode in modes:
        rows = evaluate(df, mode, min_train_races=args.min_train_races, seed=args.seed)
        print(f"\n== {mode} ==")
        print(per_race_summary(rows).to_string(index=False, float_format=fmt))
        per_season = (
            pd.DataFrame(rows)
            .groupby(["season", "model"])[["spearman", "top3_hit_rate", "pace_rmse_s"]]
            .mean()
            .reset_index()
        )
        print("\nby season:")
        print(per_season.to_string(index=False, float_format=fmt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
