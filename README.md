# Vroom 🏎️

F1 prediction app: probabilities for **finishing order**, **fastest lap**, and
the **championships**, from two models (DNF classifier + pace regression) fed
into a Monte Carlo race simulator — served as a web app.

- Modelling approach, data sources, validation rules: [`CLAUDE.md`](CLAUDE.md)
- Product and modelling decisions (UI stack, objective function, guardrails):
  [`docs/DECISIONS.md`](docs/DECISIONS.md)
- VPS deployment (Docker Compose + Caddy, duckdns): [`deploy/`](deploy/README.md)

## Layout

- `src/f1pred/` — the prediction pipeline (ingest → features → models → sim → eval)
- `src/vroom/` — the FastAPI + Jinja + HTMX web app (depends on `f1pred`, never the reverse)
- `data/` — raw pulls, caches, processed parquet (gitignored)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m f1pred.ingest --season 2026          # pull/update data
pytest
uvicorn vroom.app:app --reload                 # local web app
```

## Status

Build order (from `CLAUDE.md`): **1. ingestion ✅ (Jolpica; FastF1 loader ready)**
→ 2. post-quali pace model + grid baseline → 3. DNF model + race simulator
→ 4. championship simulator → 5. car profile + track fingerprint
→ 6. fastest-lap classifier → 7. pre-quali model.
