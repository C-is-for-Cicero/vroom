"""Form features must only see races strictly before the current one."""

import pandas as pd

from f1pred.features.form import add_form_features


def toy_core() -> pd.DataFrame:
    rows = []
    # one driver, one team, finishes 1,3,5 in rounds 1-3
    for rnd, pos in [(1, 1), (2, 3), (3, 5)]:
        rows.append(
            {
                "season": 2026,
                "round": rnd,
                "driver_id": "a",
                "team": "red_bull",
                "position": pos,
                "grid": pos + 1,
                "dnf": False,
                "classified": True,
            }
        )
    return pd.DataFrame(rows)


def test_driver_form_is_shifted():
    df = add_form_features(toy_core())
    by_round = df.set_index("round")
    assert pd.isna(by_round.loc[1, "driver_form_finish"])  # no history at round 1
    assert by_round.loc[2, "driver_form_finish"] == 1.0  # only round 1 visible
    assert by_round.loc[3, "driver_form_finish"] == 2.0  # mean of rounds 1-2


def test_team_form_is_season_scoped_and_shifted():
    df = toy_core()
    prev = df.copy()
    prev["season"] = 2025
    prev["position"] = 20  # terrible previous season must never leak in
    df = add_form_features(pd.concat([prev, df], ignore_index=True))
    cur = df[df.season == 2026].set_index("round")
    assert pd.isna(cur.loc[1, "team_season_finish"])  # reset at season start
    assert cur.loc[2, "team_season_finish"] == 1.0
    assert cur.loc[3, "team_season_finish"] == 2.0


def test_experience_counts_prior_races():
    df = add_form_features(toy_core())
    assert list(df.sort_values("round")["driver_experience"]) == [0, 1, 2]
