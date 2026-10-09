import numpy as np
import pandas as pd

from f1pred.models.fastest_lap import FastestLapModel


def _dataset(n_races: int = 30, n_drivers: int = 10, seed: int = 3) -> pd.DataFrame:
    """Fastest lap goes to the driver with the best quali delta."""
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in range(1, n_races + 1):
        quali = np.sort(rng.normal(1.0, 0.5, n_drivers))
        quali -= quali.min()
        fl = int(np.argmin(quali + rng.normal(0, 0.2, n_drivers)))
        for d in range(n_drivers):
            rows.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "grid": d + 1,
                    "quali_position": d + 1,
                    "quali_delta_pct": quali[d],
                    "driver_form_finish": d + 1.0,
                    "driver_form_vs_grid": 0.0,
                    "driver_experience": rnd,
                    "team_season_finish": d + 1.0,
                    "pace_form_3": quali[d],
                    "team_pace_form_3": quali[d],
                    "fp_longrun_delta_s": np.nan,
                    "fp_deg_slope": np.nan,
                    "fp_consistency": np.nan,
                    **{f: np.nan for f in (
                        "tel_slow_s", "tel_med_s", "tel_fast_s", "tel_top_speed",
                        "tel_fade", "ix_slow", "ix_med", "ix_fast", "ix_straight")},
                    "is_fastest_lap": d == fl,
                }
            )
    return pd.DataFrame(rows)


def test_probabilities_normalised_and_front_loaded():
    df = _dataset()
    train = df[df["round"] <= 25]
    test_race = df[df["round"] == 30]
    model = FastestLapModel(seed=0).fit(train)
    p = model.predict(test_race)
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.all(p > 0)
    # front of the field must be far likelier than the back
    assert p[:3].sum() > p[-3:].sum() * 2
