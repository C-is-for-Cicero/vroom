"""JolpicaClient tests: caching, pagination, no-network via injected fetch."""

import json

import pytest

from f1pred.ingest.jolpica import JolpicaClient


def mrdata_page(total: int, limit: int, offset: int, races: list[dict]) -> dict:
    return {
        "MRData": {
            "total": str(total),
            "limit": str(limit),
            "offset": str(offset),
            "RaceTable": {"Races": races},
        }
    }


def make_client(tmp_path, pages: dict[str, dict], calls: list[str]) -> JolpicaClient:
    def fetch(url: str) -> dict:
        calls.append(url)
        return pages[url]

    return JolpicaClient(cache_dir=tmp_path, fetch=fetch, min_interval_s=0)


def test_get_page_caches_response(tmp_path):
    url = "https://api.jolpi.ca/ergast/f1/2026/results.json?limit=100&offset=0"
    pages = {url: mrdata_page(1, 100, 0, [{"round": "1"}])}
    calls: list[str] = []
    client = make_client(tmp_path, pages, calls)

    first = client.get_page("2026/results")
    second = client.get_page("2026/results")

    assert first == second
    assert calls == [url], "second read must come from the on-disk cache"
    cached = list(tmp_path.rglob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text()) == first


def test_get_page_force_refetches(tmp_path):
    url = "https://api.jolpi.ca/ergast/f1/2026/results.json?limit=100&offset=0"
    pages = {url: mrdata_page(1, 100, 0, [{"round": "1"}])}
    calls: list[str] = []
    client = make_client(tmp_path, pages, calls)

    client.get_page("2026/results")
    client.get_page("2026/results", force=True)
    assert calls == [url, url]


def test_get_all_merges_pages(tmp_path):
    base = "https://api.jolpi.ca/ergast/f1/2026/results.json?limit=100&offset="
    races_a = [{"round": str(i)} for i in range(1, 101)]
    races_b = [{"round": str(i)} for i in range(101, 151)]
    pages = {
        base + "0": mrdata_page(150, 100, 0, races_a),
        base + "100": mrdata_page(150, 100, 100, races_b),
    }
    calls: list[str] = []
    client = make_client(tmp_path, pages, calls)

    entries = client.get_all("2026/results")
    assert len(entries) == 150
    assert entries[0]["round"] == "1"
    assert entries[-1]["round"] == "150"
    assert len(calls) == 2


def test_get_all_empty_endpoint(tmp_path):
    # e.g. the sprint endpoint for a season without sprints
    url = "https://api.jolpi.ca/ergast/f1/2017/sprint.json?limit=100&offset=0"
    pages = {url: mrdata_page(0, 100, 0, [])}
    client = make_client(tmp_path, pages, [])
    assert client.get_all("2017/sprint") == []


def test_get_all_unknown_endpoint_raises(tmp_path):
    client = make_client(tmp_path, {}, [])
    with pytest.raises(KeyError):
        client.get_all("2026/lapcharts")
