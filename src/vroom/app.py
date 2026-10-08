"""Vroom web app.

Read-only views over the pipeline's outputs:
- /                      latest race prediction (or an ingest hint)
- /race/{season}/{round} one race's probabilities, pre/post-quali toggle
- /championship          title probabilities from the championship sim
- /track-record          the walk-forward eval log (model vs baselines)

Pages read the JSON files written by `python -m f1pred.predict` and
`python -m f1pred.sim.championship`; the scheduled refresh on the VPS
regenerates them. Auth (invite-only accounts) is wired in before exposure
on the public domain.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from f1pred.config import PROCESSED_DIR, REPO_ROOT
from vroom.auth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE_S,
    hash_password,
    make_session_token,
    user_from_token,
    verify_password,
)
from vroom.db import InviteCode, User, db_session, utcnow

TEMPLATES_DIR = Path(__file__).parent / "templates"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
EVAL_LOG = REPO_ROOT / "docs" / "EVAL_LOG.md"

_PRED_FILE = re.compile(r"^(\d{4})_(\d{2})_(pre_quali|post_quali)\.json$")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="Vroom", docs_url=None, redoc_url=None)

# ---------------------------------------------------------------------------
# Auth: the whole site is invite-only; login gates everything but these.
# ---------------------------------------------------------------------------

PUBLIC_PATHS = {"/login", "/register", "/health"}
MIN_PASSWORD_LEN = 8


def _cookie_kwargs() -> dict:
    return {
        "httponly": True,
        "samesite": "lax",
        "max_age": SESSION_MAX_AGE_S,
        # VROOM_COOKIE_SECURE=1 in deploy/.env (HTTPS via Caddy); off for dev
        "secure": os.environ.get("VROOM_COOKIE_SECURE", "0") == "1",
    }


def _safe_next(target: str | None) -> str:
    return target if target and target.startswith("/") and not target.startswith("//") else "/"


@app.middleware("http")
async def require_login(request: Request, call_next):
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    user = user_from_token(request.cookies.get(SESSION_COOKIE))
    if user is None:
        return RedirectResponse(f"/login?next={request.url.path}", status_code=303)
    request.state.user = user
    return await call_next(request)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request, next: str = "/", error: str | None = None):
    return templates.TemplateResponse(
        request, "login.html", {"next": _safe_next(next), "error": error}
    )


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...),
          next: str = Form("/")):
    with db_session() as session:
        user = session.scalar(select(User).where(User.username == username.strip()))
    if user is None or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"next": _safe_next(next), "error": "wrong username or password"},
            status_code=401,
        )
    response = RedirectResponse(_safe_next(next), status_code=303)
    response.set_cookie(SESSION_COOKIE, make_session_token(user.id), **_cookie_kwargs())
    return response


@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request, error: str | None = None):
    return templates.TemplateResponse(request, "register.html", {"error": error})


@app.post("/register")
def register(request: Request, invite: str = Form(...), username: str = Form(...),
             password: str = Form(...)):
    def fail(message: str, code: int = 400):
        return templates.TemplateResponse(
            request, "register.html", {"error": message}, status_code=code
        )

    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", username):
        return fail("username: 2-32 letters, digits, _ . -")
    if len(password) < MIN_PASSWORD_LEN:
        return fail(f"password must be at least {MIN_PASSWORD_LEN} characters")
    with db_session() as session:
        code = session.get(InviteCode, invite.strip())
        if code is None or code.used_by is not None:
            return fail("invalid or already-used invite code", 403)
        if session.scalar(select(User).where(User.username == username)):
            return fail("username already taken")
        user = User(username=username, password_hash=hash_password(password))
        session.add(user)
        session.flush()
        code.used_by = user.id
        code.used_at = utcnow()
        session.commit()
        user_id = user.id
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(SESSION_COOKIE, make_session_token(user_id), **_cookie_kwargs())
    return response


@app.get("/logout")
def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


def _prediction_index() -> list[dict]:
    """Available race predictions, newest first."""
    entries = []
    if PREDICTIONS_DIR.exists():
        for path in PREDICTIONS_DIR.iterdir():
            m = _PRED_FILE.match(path.name)
            if m:
                entries.append(
                    {"season": int(m.group(1)), "round": int(m.group(2)), "mode": m.group(3)}
                )
    return sorted(entries, key=lambda e: (e["season"], e["round"]), reverse=True)


def _load_prediction(season: int, round_number: int, mode: str) -> dict | None:
    path = PREDICTIONS_DIR / f"{season}_{round_number:02d}_{mode}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _landing_prediction() -> dict | None:
    """The prediction the landing page should show: the NEXT upcoming race
    (smallest not-yet-raced round with a prediction), falling back to the
    newest prediction file when everything predicted has already raced."""
    preds = _prediction_index()
    if not preds:
        return None
    try:
        import pandas as pd

        core = pd.read_parquet(PROCESSED_DIR / "core.parquet")
        season = max(p["season"] for p in preds)
        raced = set(core.loc[core["season"] == season, "round"].astype(int))
        upcoming = sorted(
            p["round"] for p in preds if p["season"] == season and p["round"] not in raced
        )
        if upcoming:
            return {"season": season, "round": upcoming[0]}
    except Exception:
        pass
    return preds[0]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    landing = _landing_prediction()
    if landing is None:
        return templates.TemplateResponse(request, "index.html", {"predictions": []})
    return RedirectResponse(f"/race/{landing['season']}/{landing['round']}")


@app.get("/race/{season}/{round_number}", response_class=HTMLResponse)
def race(request: Request, season: int, round_number: int, mode: str | None = None):
    available = {
        e["mode"] for e in _prediction_index()
        if e["season"] == season and e["round"] == round_number
    }
    if not available:
        raise HTTPException(404, "no prediction for this round yet")
    # post-quali supersedes pre-quali once it exists
    use_mode = mode if mode in available else (
        "post_quali" if "post_quali" in available else "pre_quali"
    )
    data = _load_prediction(season, round_number, use_mode)
    return templates.TemplateResponse(
        request,
        "race.html",
        {
            "meta": data["meta"],
            "drivers": data["drivers"],
            "modes": sorted(available),
            "mode": use_mode,
            "all_predictions": _prediction_index(),
        },
    )


@app.get("/championship", response_class=HTMLResponse)
def championship(request: Request, season: int | None = None):
    files = sorted(PREDICTIONS_DIR.glob("*_championship.json")) if PREDICTIONS_DIR.exists() else []
    if season is not None:
        files = [f for f in files if f.name.startswith(str(season))]
    if not files:
        raise HTTPException(404, "no championship simulation yet")
    data = json.loads(files[-1].read_text())
    return templates.TemplateResponse(
        request,
        "championship.html",
        {"meta": data["meta"], "drivers": data["drivers"], "teams": data["teams"]},
    )


@app.get("/track-record", response_class=HTMLResponse)
def track_record(request: Request):
    try:
        import markdown

        body = markdown.markdown(EVAL_LOG.read_text(), extensions=["tables"])
    except FileNotFoundError:
        body = "<p>No evaluation log yet.</p>"
    return templates.TemplateResponse(request, "track_record.html", {"body": body})
