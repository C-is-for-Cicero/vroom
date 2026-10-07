"""Auth flow tests: hashing, invite registration, login gating."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VROOM_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("VROOM_SECRET_KEY", "test-secret")
    monkeypatch.setenv("VROOM_COOKIE_SECURE", "0")
    import vroom.auth as auth
    import vroom.db as db

    db.reset_engine()
    auth.reset_serializer()
    from vroom.app import app

    yield TestClient(app)
    db.reset_engine()
    auth.reset_serializer()


def _mint_invite() -> str:
    from vroom.db import InviteCode, db_session, new_invite_code

    code = new_invite_code()
    with db_session() as session:
        session.add(InviteCode(code=code))
        session.commit()
    return code


def test_password_hash_roundtrip():
    from vroom.auth import hash_password, verify_password

    h = hash_password("correct horse battery")
    assert h != "correct horse battery"
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x", "not-a-hash")


def test_site_is_gated(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/login")
    assert client.get("/health").status_code == 200  # health stays public
    assert client.get("/login").status_code == 200


def test_register_login_logout_flow(client):
    code = _mint_invite()
    r = client.post(
        "/register",
        data={"invite": code, "username": "dieter", "password": "longenough"},
        follow_redirects=False,
    )
    assert r.status_code == 303 and "vroom_session" in r.cookies

    # invite is single-use
    r2 = client.post(
        "/register",
        data={"invite": code, "username": "other", "password": "longenough"},
    )
    assert r2.status_code == 403

    # logged-in user passes the gate (no prediction data -> index page 200)
    r3 = client.get("/", follow_redirects=False)
    assert r3.status_code in (200, 307)

    # logout clears the session
    client.get("/logout")
    r4 = client.get("/", follow_redirects=False)
    assert r4.status_code == 303

    # login works with the right password only
    bad = client.post("/login", data={"username": "dieter", "password": "nope", "next": "/"})
    assert bad.status_code == 401
    good = client.post(
        "/login",
        data={"username": "dieter", "password": "longenough", "next": "/"},
        follow_redirects=False,
    )
    assert good.status_code == 303 and "vroom_session" in good.cookies


def test_register_validation(client):
    code = _mint_invite()
    short = client.post(
        "/register", data={"invite": code, "username": "ok", "password": "short"}
    )
    assert short.status_code == 400
    bad_name = client.post(
        "/register", data={"invite": code, "username": "x y!", "password": "longenough"}
    )
    assert bad_name.status_code == 400
    bad_code = client.post(
        "/register", data={"invite": "nope", "username": "fine", "password": "longenough"}
    )
    assert bad_code.status_code == 403


def test_open_redirect_blocked(client):
    code = _mint_invite()
    client.post(
        "/register",
        data={"invite": code, "username": "redir", "password": "longenough"},
    )
    r = client.post(
        "/login",
        data={"username": "redir", "password": "longenough", "next": "//evil.example"},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/"


def test_manage_cli_invite_and_list(client, capsys):
    from vroom.db import InviteCode, db_session
    from vroom.manage import invite

    invite(3)
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3
    with db_session() as session:
        assert len(list(session.scalars(select(InviteCode)))) == 3
