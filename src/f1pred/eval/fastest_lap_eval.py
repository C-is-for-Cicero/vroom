"""Walk-forward evaluation of the fastest-lap classifier.

CLI: python -m f1pred.eval.fastest_lap_eval [--min-train-races 30]

Metrics per race: multinomial log-loss (-log of the probability assigned
to the actual fastest-lap setter), hit@1 and hit@3. Baselines: historical
P(fastest lap | grid position) from training races (Laplace-smoothed,
normalised per race) and the uniform 1/n guess.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from f1pred.eval.pace_eval import load_dataset
from f1pred.eval.splits import walk_forward_by_race
from f1pred.models.fastest_lap import FastestLapModel


def _grid_fl_table(train: pd.DataFrame) -> pd.Series:
    """P(fastest lap | grid slot) from training races, Laplace-smoothed."""
    grid = train["grid"].replace(0, 99).astype(int)
    tab = train.groupby(grid)["is_fastest_lap"].agg(["sum", "size"])
    return (tab["sum"] + 1) / (tab["size"] + 20)


def _metrics(p: np.ndarray, actual_idx: int) -> dict[str, float]:
    order = np.argsort(-p)
    return {
        "log_loss": float(-np.log(max(p[actual_idx], 1e-12))),
        "hit1": float(order[0] == actual_idx),
        "hit3": float(actual_idx in order[:3]),
    }


def evaluate_fastest_lap(
    df: pd.DataFrame, min_train_races: int = 30, seed: int = 0
) -> pd.DataFrame:
    rows: list[dict] = []
    for train_idx, test_idx, (season, rnd) in walk_forward_by_race(df, min_train_races):
        train, test = df.loc[train_idx], df.loc[test_idx]
        flags = test["is_fastest_lap"].to_numpy()
        if flags.sum() != 1:
            continue  # no (or ambiguous) fastest-lap record
        actual = int(np.argmax(flags))

        model = FastestLapModel(seed=seed).fit(train)
        p_model = model.predict(test)
        rows.append({"model": "fastest_lap_model", "season": season, "round": rnd,
                     **_metrics(p_model, actual)})

        table = _grid_fl_table(train)
        p_grid = table.reindex(test["grid"].replace(0, 99).astype(int)).fillna(1 / 20).to_numpy()
        p_grid = p_grid / p_grid.sum()
        rows.append({"model": "baseline_grid_freq", "season": season, "round": rnd,
                     **_metrics(p_grid, actual)})

        # fixed a-priori 50/50 ensemble with the frequency prior: the
        # standard hedge against an overconfident per-race classifier
        p_blend = 0.5 * p_model + 0.5 * p_grid
        rows.append({"model": "fastest_lap_blend", "season": season, "round": rnd,
                     **_metrics(p_blend / p_blend.sum(), actual)})

        p_uni = np.full(len(test), 1 / len(test))
        rows.append({"model": "baseline_uniform", "season": season, "round": rnd,
                     **_metrics(p_uni, actual)})
    out = pd.DataFrame(rows)
    summary = out.groupby("model")[["log_loss", "hit1", "hit3"]].mean()
    summary["n_races"] = out.groupby("model").size()
    return summary.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred.eval.fastest_lap_eval", description=__doc__)
    parser.add_argument("--min-train-races", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    df = load_dataset()
    summary = evaluate_fastest_lap(df, min_train_races=args.min_train_races, seed=args.seed)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
