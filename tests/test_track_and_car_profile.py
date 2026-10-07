import numpy as np
import pandas as pd

from f1pred.features.car_profile import _stint_features
from f1pred.features.track_fingerprint import add_track_features, grid_effect_scale


def _race(season, rnd, circuit, finish_follows_grid: bool) -> pd.DataFrame:
    n = 10
    grid = np.arange(1, n + 1)
    pos = grid if finish_follows_grid else grid[::-1]
    return pd.DataFrame(
        {
            "season": season,
            "round": rnd,
            "circuit_id": circuit,
            "grid": grid,
            "position": pos.astype(float),
        }
    )


def test_circuit_grid_hold_is_shifted_per_circuit():
    df = pd.concat(
        [
            _race(2024, 1, "monaco", True),
            _race(2024, 2, "spa", False),
            _race(2025, 1, "monaco", True),
            _race(2025, 2, "spa", False),
        ],
        ignore_index=True,
    )
    out = add_track_features(df)
    first_monaco = out[(out.season == 2024) & (out.circuit_id == "monaco")]
    assert first_monaco["circuit_grid_hold"].isna().all()  # no history yet
    second_monaco = out[(out.season == 2025) & (out.circuit_id == "monaco")]
    assert np.allclose(second_monaco["circuit_grid_hold"], 1.0)  # processional
    second_spa = out[(out.season == 2025) & (out.circuit_id == "spa")]
    assert np.allclose(second_spa["circuit_grid_hold"], -1.0)  # total shuffle


def test_grid_effect_scale_clipped_and_defaults():
    train = pd.DataFrame({"circuit_grid_hold": [0.5, 0.7, 0.6]})
    processional = pd.DataFrame({"circuit_grid_hold": [0.9]})
    unknown = pd.DataFrame({"circuit_grid_hold": [np.nan]})
    assert grid_effect_scale(processional, train) == 0.9 / 0.6
    assert grid_effect_scale(unknown, train) == 1.0
    extreme = pd.DataFrame({"circuit_grid_hold": [5.0]})
    assert grid_effect_scale(extreme, train) == 2.0  # clipped


def _fake_fp_laps() -> pd.DataFrame:
    """Two drivers: A fast and consistent, B slower, degrading harder."""
    rows = []
    for driver, base, deg in [("AAA", 90.0, 0.05), ("BBB", 91.0, 0.15)]:
        for stint, n in [(1.0, 8), (2.0, 6)]:
            for life in range(1, n + 1):
                rows.append(
                    {
                        "Driver": driver,
                        "Stint": stint,
                        "LapTime": pd.Timedelta(seconds=base + deg * life),
                        "TyreLife": float(life),
                    }
                )
    return pd.DataFrame(rows)


def test_stint_features_extract_pace_and_degradation():
    out = _stint_features(_fake_fp_laps()).set_index("driver_code")
    assert out.loc["AAA", "fp_longrun_delta_s"] == 0.0  # field best
    assert out.loc["BBB", "fp_longrun_delta_s"] > 0.5
    assert out.loc["BBB", "fp_deg_slope"] > out.loc["AAA", "fp_deg_slope"]
    assert out.loc["AAA", "fp_consistency"] < 0.01  # perfectly linear laps
    assert out.loc["AAA", "fp_longrun_laps"] == 14
