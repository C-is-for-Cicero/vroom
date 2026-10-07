"""Walk-forward evaluation of the post-quali pace model vs the grid baseline.

CLI: python -m f1pred.eval.pace_eval [--min-train-races 60]

For each test race (chronological, after a minimum training history) the
model is refitted on all earlier races only, per the validation rules.
Reported per model: mean Spearman of predicted vs actual finishing order
(classified finishers), top-3 hit rate, and RMSE of the pace prediction on
rows with an observed target. Log-loss/Brier columns join the table in
build-order step 3, when the simulator turns pace+sigma into probabilities;
the odds-baseline column joins when odds ingestion lands.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from f1pred.config import PROCESSED_DIR
from f1pred.eval.baselines import grid_baseline_rank
from f1pred.eval.metrics import per_race_summary, spearman_order, top3_hit_rate
from f1pred.eval.splits import walk_forward_by_race
from f1pred.features.race_pace import load_race_pace
from f1pred.models.pace import PaceModel, add_pace_form_features


def load_dataset() -> pd.DataFrame:
    """Core table + race-pace target, restricted to seasons with targets."""
    core = pd.read_parquet(PROCESSED_DIR / "core.parquet")
    pace = load_race_pace(sorted(core["season"].unique()))
    df = core.merge(pace, on=["season", "round", "driver_code"], how="left")
    df = add_pace_form_features(df)
    seasons_with_target = sorted(df.loc[df["pace_delta_s"].notna(), "season"].unique())
    return df[df["season"].isin(seasons_with_target)].reset_index(drop=True)


def evaluate(df: pd.DataFrame, min_train_races: int = 60, seed: int = 0) -> pd.DataFrame:
    """Run the walk-forward loop; returns the per-model summary table."""
    rows: list[dict] = []
    for train_idx, test_idx, (season, rnd) in walk_forward_by_race(df, min_train_races):
        train, test = df.loc[train_idx], df.loc[test_idx]
        if test["pace_delta_s"].notna().sum() < 5:
            continue  # race without usable target (e.g. not yet run)
        model = PaceModel(seed=seed).fit(train)
        mu, _ = model.predict(test)
        pred_rank = pd.Series(mu, index=test.index).rank(method="first")
        actual = test["position"].astype(float)

        labeled = test["pace_delta_s"].notna()
        rmse = float(
            np.sqrt(np.mean((mu[labeled.to_numpy()] - test.loc[labeled, "pace_delta_s"]) ** 2))
        )
        rows.append(
            {
                "model": "pace_model_post_quali",
                "season": season,
                "round": rnd,
                "spearman": spearman_order(pred_rank, actual),
                "top3_hit_rate": top3_hit_rate(pred_rank, actual),
                "pace_rmse_s": rmse,
            }
        )
        grid_rank = grid_baseline_rank(test)
        rows.append(
            {
                "model": "baseline_grid",
                "season": season,
                "round": rnd,
                "spearman": spearman_order(grid_rank, actual),
                "top3_hit_rate": top3_hit_rate(grid_rank, actual),
                "pace_rmse_s": float("nan"),
            }
        )
    return per_race_summary(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.eval.pace_eval", description=__doc__)
    parser.add_argument("--min-train-races", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = load_dataset()
    n_target = int(df["pace_delta_s"].notna().sum())
    races = df.loc[df["pace_delta_s"].notna(), ["season", "round"]].drop_duplicates()
    print(f"dataset: {len(df)} rows, {n_target} with pace target over {len(races)} races\n")
    summary = evaluate(df, min_train_races=args.min_train_races, seed=args.seed)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
