"""Vroom web app shell.

Serves the landing page and a health endpoint. The real views (race
prediction, championship sim, model track record, driver/team profiles,
track fingerprints) land as their pipeline stages are built; auth (invite-only
accounts, viewing only) comes with the first real view. See docs/DECISIONS.md.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from f1pred.config import CURRENT_SEASON, JOLPICA_RAW_DIR

TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="Vroom", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    ingested = JOLPICA_RAW_DIR.exists() and any(JOLPICA_RAW_DIR.iterdir())
    return templates.TemplateResponse(
        request,
        "index.html",
        {"season": CURRENT_SEASON, "ingested": ingested},
    )
