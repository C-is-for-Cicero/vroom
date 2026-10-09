import numpy as np
import pytest
from fastapi.testclient import TestClient

from f1pred.ingest.telemetry import resample_channels


def test_resample_channels_uniform_grid():
    dist = np.array([0.0, 100.0, 300.0, 600.0])
    out = resample_channels(dist, {"speed": np.array([0.0, 100.0, 300.0, 600.0])}, 7)
    assert len(out["dist"]) == 7 and len(out["speed"]) == 7
    assert out["dist"][0] == 0.0 and out["dist"][-1] == 600.0
    # the channel equals distance here, so interpolation must be identity
    assert out["speed"] == out["dist"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VROOM_DB_PATH", str(tmp_path / "t.sqlite"))
    monkeypatch.setenv("VROOM_SECRET_KEY", "t")
    import vroom.auth as auth
    import vroom.db as db

    db.reset_engine()
    auth.reset_serializer()
    from vroom.app import app
    from vroom.db import InviteCode, db_session, new_invite_code

    c = TestClient(app)
    code = new_invite_code()
    with db_session() as s:
        s.add(InviteCode(code=code))
        s.commit()
    c.post("/register", data={"invite": code, "username": "tt", "password": "longenough"})
    yield c
    db.reset_engine()
    auth.reset_serializer()


def test_telemetry_api_validation_and_mapping(client, monkeypatch):
    import f1pred.ingest.telemetry as tel

    assert client.get("/api/telemetry/2026/16/XX/VER").status_code == 400

    monkeypatch.setattr(tel, "fastest_lap_channels",
                        lambda *a, **k: (_ for _ in ()).throw(LookupError("no lap")))
    assert client.get("/api/telemetry/2026/16/Q/VER").status_code == 404

    monkeypatch.setattr(
        tel, "fastest_lap_channels",
        lambda season, rnd, code, drv, **k: {"driver": drv, "lap_time": 91.2,
                                             "dist": [0, 1], "speed": [0, 1],
                                             "throttle": [0, 1], "brake": [0, 0],
                                             "time": [0, 1], "session": code},
    )
    r = client.get("/api/telemetry/2026/16/Q/VER")
    assert r.status_code == 200 and r.json()["driver"] == "VER"


def test_telemetry_page_lists_events(client):
    r = client.get("/telemetry")
    assert r.status_code == 200
    assert "events-data" in r.text and "Add driver" in r.text
