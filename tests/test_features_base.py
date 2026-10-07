from f1pred.features.base import qualifying_frame, results_frame, time_str_to_seconds
from f1pred.ingest.jolpica import JolpicaClient


def test_time_str_to_seconds():
    assert time_str_to_seconds("1:23.456") == 83.456
    assert time_str_to_seconds("59.123") == 59.123
    assert time_str_to_seconds(None) is None
    assert time_str_to_seconds("") is None
    assert time_str_to_seconds("DNF") is None


def _client_with(tmp_path, endpoint_payloads: dict[str, dict]) -> JolpicaClient:
    def fetch(url: str) -> dict:
        for fragment, payload in endpoint_payloads.items():
            if fragment in url:
                return payload
        raise AssertionError(f"unexpected url {url}")

    return JolpicaClient(cache_dir=tmp_path, fetch=fetch, min_interval_s=0)


def _results_payload() -> dict:
    def result(driver, code, constructor, grid, pos_text, status, points="0", laps="57"):
        return {
            "Driver": {"driverId": driver, "code": code},
            "Constructor": {"constructorId": constructor},
            "grid": grid,
            "position": pos_text if pos_text.isdigit() else "19",
            "positionText": pos_text,
            "status": status,
            "points": points,
            "laps": laps,
        }

    race = {
        "round": "1",
        "raceName": "Test GP",
        "date": "2026-03-08",
        "Circuit": {"circuitId": "testring"},
        "Results": [
            result("max_verstappen", "VER", "red_bull", "1", "1", "Finished", "25"),
            result("gabriel_bortoleto", "BOR", "sauber", "5", "2", "Finished", "18"),
            result("lando_norris", "NOR", "mclaren", "2", "R", "Collision"),
            result("lewis_hamilton", "HAM", "ferrari", "3", "D", "Disqualified"),
        ],
    }
    return {
        "MRData": {
            "total": "1",
            "limit": "100",
            "offset": "0",
            "RaceTable": {"Races": [race]},
        }
    }


def test_results_frame_classification(tmp_path):
    client = _client_with(tmp_path, {"/2026/results.json": _results_payload()})
    df = results_frame([2026], client)
    assert len(df) == 4
    ver = df[df.driver_id == "max_verstappen"].iloc[0]
    assert ver.position == 1 and ver.classified and not ver.dnf
    assert ver.team == "red_bull"
    # 2026 reset: sauber entries count as audi lineage
    assert df[df.driver_id == "gabriel_bortoleto"].iloc[0].team == "audi"
    import pandas as pd

    nor = df[df.driver_id == "lando_norris"].iloc[0]
    assert nor.dnf and not nor.classified and pd.isna(nor.position)
    ham = df[df.driver_id == "lewis_hamilton"].iloc[0]
    assert ham.disqualified and not ham.dnf


def test_qualifying_frame_delta(tmp_path):
    payload = {
        "MRData": {
            "total": "1",
            "limit": "100",
            "offset": "0",
            "RaceTable": {
                "Races": [
                    {
                        "round": "1",
                        "QualifyingResults": [
                            {
                                "Driver": {"driverId": "a"},
                                "position": "1",
                                "Q1": "1:31.000",
                                "Q2": "1:30.500",
                                "Q3": "1:30.000",
                            },
                            {
                                "Driver": {"driverId": "b"},
                                "position": "2",
                                "Q1": "1:31.200",
                                "Q2": "1:30.900",
                            },
                        ],
                    }
                ]
            },
        }
    }
    client = _client_with(tmp_path, {"/2026/qualifying.json": payload})
    df = qualifying_frame([2026], client)
    a, b = df[df.driver_id == "a"].iloc[0], df[df.driver_id == "b"].iloc[0]
    assert a.best_quali_s == 90.0 and a.quali_delta_pct == 0.0
    assert b.best_quali_s == 90.9
    assert abs(b.quali_delta_pct - 1.0) < 1e-9
