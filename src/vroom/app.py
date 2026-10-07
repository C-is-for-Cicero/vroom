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
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from f1pred.config import PROCESSED_DIR, REPO_ROOT

TEMPLATES_DIR = Path(__file__).parent / "templates"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
EVAL_LOG = REPO_ROOT / "docs" / "EVAL_LOG.md"

_PRED_FILE = re.compile(r"^(\d{4})_(\d{2})_(pre_quali|post_quali)\.json$")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="Vroom", docs_url=None, redoc_url=None)


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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    preds = _prediction_index()
    if not preds:
        return templates.TemplateResponse(request, "index.html", {"predictions": []})
    latest = preds[0]
    return RedirectResponse(f"/race/{latest['season']}/{latest['round']}")


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
