import numpy as np
import pandas as pd

from f1pred.eval.baselines import apply_rank_probs, rank_outcome_probs
from f1pred.models.dnf import P_DNF_MAX, P_DNF_MIN, DnfModel


def test_dnf_model_learns_rates_and_clips():
    rng = np.random.default_rng(0)
    n = 2000
    fragile = rng.random(n) < 0.5
    df = pd.DataFrame(
        {
            "driver_dnf_rate": np.where(fragile, 0.4, 0.05) + rng.normal(0, 0.02, n),
            "team_dnf_rate": np.where(fragile, 0.35, 0.06) + rng.normal(0, 0.02, n),
            "circuit_dnf_rate": rng.uniform(0.05, 0.2, n),
            "driver_experience": rng.integers(0, 150, n).astype(float),
            "dnf": rng.random(n) < np.where(fragile, 0.4, 0.05),
        }
    )
    model = DnfModel(seed=0).fit(df)
    p = model.predict(df)
    assert np.all(p >= P_DNF_MIN) and np.all(p <= P_DNF_MAX)
    assert p[fragile].mean() > p[~fragile].mean() + 0.1


def test_rank_outcome_probs_and_lookup():
    # 10 races, 4 drivers, rank 1 always wins
    rows = []
    for r in range(10):
        for rank, pos in [(1, 1), (2, 2), (3, 3), (4, 4)]:
            rows.append({"race": r, "rank_val": rank, "position": pos})
    train = pd.DataFrame(rows)
    table = rank_outcome_probs(train, train["rank_val"])
    assert table.loc[1, "p_win"] == (10 + 1) / (10 + 2)
    assert table.loc[2, "p_win"] == (0 + 1) / (10 + 2)
    assert table.loc[4, "p_points"] == (10 + 1) / (10 + 2)

    test_rank = pd.Series([2, 1, 9], index=["x", "y", "z"])  # rank 9 unseen
    looked = apply_rank_probs(table, test_rank)
    assert looked.loc["y", "p_win"] == table.loc[1, "p_win"]
    assert looked.loc["z", "p_win"] == table.loc[4, "p_win"]  # clipped to worst
