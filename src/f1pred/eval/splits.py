"""Time-ordered walk-forward splits. Never shuffled, never K-fold.

Leakage cutoff: for a test race R, the train set is every race that finished
strictly before R in (season, round) order.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def race_order(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Distinct (season, round) pairs in chronological order."""
    pairs = df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    return list(pairs.itertuples(index=False, name=None))


def walk_forward_by_race(
    df: pd.DataFrame, min_train_races: int = 30
) -> Iterator[tuple[pd.Index, pd.Index, tuple[int, int]]]:
    """Yield (train_index, test_index, (season, round)) per test race.

    Input: any frame with season and round columns. The first
    `min_train_races` races are never tested, only trained on.
    """
    races = race_order(df)
    keys = list(zip(df["season"], df["round"], strict=True))
    key_series = pd.Series(keys, index=df.index)
    for i in range(min_train_races, len(races)):
        test_race = races[i]
        train_races = set(races[:i])
        train_idx = df.index[[k in train_races for k in key_series]]
        test_idx = df.index[key_series == test_race]
        yield train_idx, test_idx, test_race
