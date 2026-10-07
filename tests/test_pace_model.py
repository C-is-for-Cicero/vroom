import numpy as np
import pandas as pd

from f1pred.models.pace import SIGMA_FLOOR_S, PaceModel, add_pace_form_features


def synthetic_dataset(n_races: int = 40, n_drivers: int = 10, seed: int = 1) -> pd.DataFrame:
    """Pace delta driven mostly by quali delta, with noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in range(1, n_races + 1):
        quali = np.sort(rng.normal(0, 1.0, n_drivers)) - 0  # ordered field
        quali -= quali.min()
        for d in range(n_drivers):
            rows.append(
                {
                    "season": 2024 + (rnd - 1) // 22,
                    "round": (rnd - 1) % 22 + 1,
                    "driver_id": f"d{d}",
                    "team": f"t{d // 2}",
                    "grid": d + 1,
                    "quali_position": d + 1,
                    "quali_delta_pct": quali[d],
                    "driver_form_finish": d + 1.0,
                    "driver_form_vs_grid": 0.0,
                    "driver_experience": rnd,
                    "team_season_finish": d + 1.0,
                    "fp_longrun_delta_s": np.nan,
                    "fp_deg_slope": np.nan,
                    "fp_consistency": np.nan,
                    "pace_delta_s": quali[d] * 0.8 + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_pace_model_learns_signal_and_floors_sigma():
    df = add_pace_form_features(synthetic_dataset())
    train = df[df["round"] <= 18] if df["season"].nunique() == 1 else df.iloc[: len(df) * 3 // 4]
    test = df.drop(train.index)
    model = PaceModel(seed=0).fit(train)
    mu, sigma = model.predict(test)
    assert mu.shape == (len(test),) and sigma.shape == (len(test),)
    assert np.all(sigma >= SIGMA_FLOOR_S)
    # predictions must correlate strongly with the true target
    corr = np.corrcoef(mu, test["pace_delta_s"])[0, 1]
    assert corr > 0.8


def test_pace_form_is_shifted():
    df = pd.DataFrame(
        {
            "season": [2026] * 3,
            "round": [1, 2, 3],
            "driver_id": ["a"] * 3,
            "team": ["x"] * 3,
            "pace_delta_s": [0.3, 0.6, 0.9],
        }
    )
    out = add_pace_form_features(df).set_index("round")
    assert pd.isna(out.loc[1, "pace_form_3"])
    assert out.loc[2, "pace_form_3"] == 0.3
    assert abs(out.loc[3, "pace_form_3"] - 0.45) < 1e-12
    assert pd.isna(out.loc[1, "team_pace_form_3"])
    assert out.loc[2, "team_pace_form_3"] == 0.3
