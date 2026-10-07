import numpy as np

from f1pred.ingest.odds import OddsClient, devig_event, normalise_name


def test_normalise_name_strips_accents_and_case():
    assert normalise_name("Sergio Pérez") == "sergio perez"
    assert normalise_name("Nico Hülkenberg") == "nico hulkenberg"
    assert normalise_name("  Max  Verstappen ") == "max  verstappen".strip()


def _event(bookmaker_prices: list[dict[str, float]]) -> dict:
    return {
        "commence_time": "2026-10-25T18:00:00Z",
        "bookmakers": [
            {
                "key": f"bm{i}",
                "markets": [
                    {
                        "key": "outrights",
                        "outcomes": [{"name": n, "price": p} for n, p in prices.items()],
                    }
                ],
            }
            for i, prices in enumerate(bookmaker_prices)
        ],
    }


def test_devig_removes_overround_and_averages():
    # one bookmaker, heavy overround: raw implied probs sum to ~1.2
    prices = {f"Driver {i}": 5.0 for i in range(6)}  # 6 * 0.2 = 1.2
    out = devig_event(_event([prices])).set_index("outcome_name")
    assert abs(out["p_win_odds"].sum() - 1.0) < 1e-9
    assert np.allclose(out["p_win_odds"], 1 / 6)

    # two bookmakers disagreeing on the favourite -> consensus in between
    bm1 = {"A": 1.5, "B": 4.0, "C": 8.0, "D": 10.0, "E": 12.0}
    bm2 = {"A": 2.5, "B": 3.0, "C": 8.0, "D": 10.0, "E": 12.0}
    out = devig_event(_event([bm1, bm2])).set_index("outcome_name")
    p1 = devig_event(_event([bm1])).set_index("outcome_name")["p_win_odds"]["A"]
    p2 = devig_event(_event([bm2])).set_index("outcome_name")["p_win_odds"]["A"]
    assert min(p1, p2) < out["p_win_odds"]["A"] < max(p1, p2)


def test_devig_skips_thin_and_broken_markets():
    thin = _event([{"A": 2.0, "B": 3.0}])  # fewer than 5 outcomes
    assert devig_event(thin).empty
    broken = _event([{f"D{i}": 0.0 for i in range(6)}])  # invalid prices
    assert devig_event(broken).empty


def test_sport_key_discovery_uses_injected_fetch():
    def fetch(path, params):
        assert path == "sports"
        return [
            {"key": "soccer_epl", "title": "EPL", "description": "English Premier League"},
            {"key": "motorsport_f1_x", "title": "F1 Winner",
             "description": "Formula 1 race winner"},
        ]

    client = OddsClient(api_key="k", fetch=fetch)
    assert client.f1_sport_keys() == ["motorsport_f1_x"]
