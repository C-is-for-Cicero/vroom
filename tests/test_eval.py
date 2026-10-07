import numpy as np
import pandas as pd

from f1pred.eval.baselines import grid_baseline_rank, standings_baseline_rank
from f1pred.eval.metrics import binary_log_loss, brier_score, spearman_order, top3_hit_rate
from f1pred.eval.splits import walk_forward_by_race


def test_spearman_perfect_and_reversed():
    pred = pd.Series([1, 2, 3, 4], index=list("abcd"))
    actual = pd.Series([1, 2, 3, 4], index=list("abcd"))
    assert spearman_order(pred, actual) == 1.0
    assert spearman_order(pred, actual[::-1].set_axis(list("abcd"))) == -1.0


def test_spearman_ignores_unclassified():
    pred = pd.Series([1, 2, 3, 4], index=list("abcd"))
    actual = pd.Series([1.0, 2.0, 3.0, np.nan], index=list("abcd"))
    assert spearman_order(pred, actual) == 1.0


def test_top3_hit_rate():
    pred = pd.Series([1, 2, 3, 4, 5], index=list("abcde"))
    actual = pd.Series([3, 1, 2, 4, 5], index=list("abcde"))  # same podium set
    assert top3_hit_rate(pred, actual) == 1.0
    actual2 = pd.Series([5, 4, 3, 2, 1], index=list("abcde"))
    assert top3_hit_rate(pred, actual2) == 1 / 3  # only 'c' overlaps


def test_probability_metrics():
    y = np.array([1, 0, 1, 0])
    p_good = np.array([0.9, 0.1, 0.8, 0.2])
    p_bad = np.array([0.5, 0.5, 0.5, 0.5])
    assert binary_log_loss(y, p_good) < binary_log_loss(y, p_bad)
    assert brier_score(y, p_good) < brier_score(y, p_bad)
    assert binary_log_loss(np.array([1]), np.array([1.0])) < 1e-9


def test_grid_baseline_pit_lane_starts_last():
    race = pd.DataFrame({"grid": [2, 1, 0, 3]}, index=list("abcd"))
    rank = grid_baseline_rank(race)
    assert rank["c"] == 4.0  # grid 0 = pit lane = ranked last
    assert rank["b"] == 1.0


def test_standings_baseline_orders_by_points_then_prev_season():
    race = pd.DataFrame(
        {
            "season_points_prior": [50.0, 80.0, 0.0, 0.0],
            "prev_season_points": [200.0, 10.0, 90.0, 120.0],
        },
        index=list("abcd"),
    )
    rank = standings_baseline_rank(race)
    # b leads on current points despite a's big previous season;
    # c and d (no points yet) fall back to previous season order.
    assert list(rank.sort_values().index) == ["b", "a", "d", "c"]


def _race_df() -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        for rnd in (1, 2, 3):
            for driver in ("a", "b"):
                rows.append({"season": season, "round": rnd, "driver_id": driver})
    return pd.DataFrame(rows)


def test_walk_forward_is_strictly_time_ordered():
    df = _race_df()
    splits = list(walk_forward_by_race(df, min_train_races=2))
    assert len(splits) == 4  # 6 races, first 2 train-only
    for train_idx, _test_idx, (season, rnd) in splits:
        test_key = (season, rnd)
        train_keys = set(zip(df.loc[train_idx, "season"], df.loc[train_idx, "round"], strict=True))
        assert test_key not in train_keys
        assert all(k < test_key for k in train_keys)  # tuple order = chronological
    # last test race is the newest
    assert splits[-1][2] == (2025, 3)
