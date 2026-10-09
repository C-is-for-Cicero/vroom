import numpy as np
import pandas as pd
import pytest

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
                        "LapTime": pd.to_timedelta(base + deg * life, unit="s"),
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


def test_builders_stop_pass_on_fastf1_rate_limit(tmp_path, monkeypatch):
    """Once FastF1's hourly budget trips, the pass stops instead of
    hammering every remaining round."""
    import f1pred.features.car_profile as cp
    import f1pred.features.race_pace as rp

    class RateLimitExceededError(Exception):
        pass

    calls = {"rp": 0, "cp": 0}

    def rp_boom(season, rnd):
        calls["rp"] += 1
        raise RateLimitExceededError("any API: 500 calls/h")

    def cp_boom(season, rnd):
        calls["cp"] += 1
        raise RateLimitExceededError("any API: 500 calls/h")

    monkeypatch.setattr(rp, "race_pace_for_round", rp_boom)
    monkeypatch.setattr(rp, "RACE_PACE_DIR", tmp_path / "rp")
    monkeypatch.setattr(cp, "fp_longrun_for_round", cp_boom)
    monkeypatch.setattr(cp, "FP_LONGRUN_DIR", tmp_path / "cp")

    from f1pred.ingest.fastf1_loader import FastF1BudgetExhausted

    with pytest.raises(FastF1BudgetExhausted):
        rp.build_race_pace(2024, [1, 2, 3, 4, 5])
    with pytest.raises(FastF1BudgetExhausted):
        cp.build_fp_longrun(2024, [1, 2, 3, 4, 5])
    assert calls == {"rp": 1, "cp": 1}  # stopped after the first budget error


def test_corner_zone_geometry_and_classes():
    from f1pred.features.car_profile import classify_zones, corner_zones

    zones = corner_zones([1000.0, 1300.0, 3000.0], 5000.0)
    # first two zones must split at their midpoint (1150), not overlap
    assert zones[0][1] == 1150.0 and zones[1][0] == 1150.0
    assert zones[0][0] == 800.0 and zones[2] == (2800.0, 3150.0)

    dist = np.linspace(0, 5000, 5001)
    speed = np.full_like(dist, 300.0)
    speed[(dist > 900) & (dist < 1100)] = 80    # slow
    speed[(dist > 1200) & (dist < 1400)] = 150  # medium
    speed[(dist > 2900) & (dist < 3100)] = 250  # fast
    assert classify_zones(zones, dist, speed) == [0, 1, 2]


def test_lap_profile_attributes_time_loss_to_the_right_class():
    from f1pred.features.car_profile import classify_zones, corner_zones, lap_profile

    dist = np.linspace(0, 5000, 5001)

    def make(speed_slow):
        speed = np.full_like(dist, 300.0)
        speed[(dist > 900) & (dist < 1100)] = speed_slow
        speed[(dist > 2900) & (dist < 3100)] = 250.0
        kmh_to_ms = 1 / 3.6
        dt = np.diff(dist) / (speed[:-1] * kmh_to_ms)
        time_s = np.concatenate([[0.0], np.cumsum(dt)])
        return speed, time_s

    zones = corner_zones([1000.0, 3000.0], 5000.0)
    ref_speed, ref_time = make(100.0)
    classes = classify_zones(zones, dist, ref_speed)
    ref = lap_profile(dist, ref_time, ref_speed, zones, classes)

    slow_speed, slow_time = make(80.0)  # slower ONLY in the slow corner
    drv = lap_profile(dist, slow_time, slow_speed, zones, classes)

    assert drv["time_slow"] - ref["time_slow"] > 0.3     # loses real time there
    assert abs(drv["time_fast"] - ref["time_fast"]) < 1e-6  # identical elsewhere
    assert drv["top_speed"] == ref["top_speed"]
    assert ref["longest_straight_m"] > 1500  # the 1150->2800 gap


def test_telemetry_interactions():
    from f1pred.features.interactions import add_telemetry_interactions

    df = pd.DataFrame(
        {
            "tel_slow_s": [0.4, np.nan],
            "tel_med_s": [0.1, 0.1],
            "tel_fast_s": [0.0, 0.0],
            "tel_top_speed": [-6.0, -6.0],
            "trk_slow_share": [0.25, 0.25],
            "trk_med_share": [0.2, 0.2],
            "trk_fast_share": [0.1, 0.1],
            "trk_straight_share": [0.45, 0.45],
        }
    )
    out = add_telemetry_interactions(df)
    assert abs(out.loc[0, "ix_slow"] - 0.1) < 1e-12
    assert np.isnan(out.loc[1, "ix_slow"])  # missing telemetry stays missing
    assert abs(out.loc[0, "ix_straight"] - 2.7) < 1e-12  # deficit x share
