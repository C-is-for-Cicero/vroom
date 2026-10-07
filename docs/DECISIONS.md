# Vroom — product and modelling decisions

Agreed 2026-10-07 between the project owner and Claude. These refine (never
override) the modelling approach in `CLAUDE.md`. Change them the same way they
were made: by discussion, recorded here.

## UI and product

| Decision | Choice | Notes |
|---|---|---|
| Frontend stack | FastAPI + Jinja2 + HTMX | Server-rendered, one Python process behind Caddy. Charts via a small embedded JS library (Plotly/Chart.js). No Node build chain. |
| Audience & auth | Me + friends, user accounts | Invite-only accounts. Login gates the whole site; accounts are for **viewing only** (no prediction league for now — schema should not preclude adding one later). |
| Views | All of them | 1. **Race prediction** (core): per-driver finishing-position distribution, win/podium/points probabilities, pre-/post-quali toggle. 2. **Championship simulator**: title probabilities and evolution over the season. 3. **Model track record**: log-loss/Brier/Spearman vs baselines per race, calibration plots — the validation table as a living page. 4. **Driver & team profiles**: car profile and form. 5. **Track fingerprints**: per-circuit features, SC/DNF rates, overtaking difficulty, similar tracks. |
| Refresh | Scheduled auto-refresh | Cron on the VPS pulls new session data after each session and regenerates predictions. The site always shows the newest available mode (post-quali supersedes pre-quali). No OpenF1 near-live for now. |

## Modelling

| Decision | Choice | Notes |
|---|---|---|
| **Primary product mode** | **pre_quali** (clarified 2026-10-07) | The headline prediction is the finishing order guessed *before qualifying*. post_quali remains a separate model (per CLAUDE.md) shown once quali has run. Consequence: the pre-quali eval cannot use the grid baseline; its naive baseline is championship-standings order (previous season's order at round 1). The grid baseline stays as the post-quali bar. A first pre-quali model is pulled forward from build-order step 7; the full FP-based version (practice long runs, car profile) still lands with step 5/7. |
| Primary objective | **Spearman rank correlation** of expected finishing order | The tie-breaker for hyperparameters and model-version selection on walk-forward eval. |
| Log-loss role | **Hard guardrail + calibration** | A model version only ships if its win/podium/points log-loss beats the grid baseline on walk-forward eval. Displayed probabilities get post-hoc isotonic calibration, fitted walk-forward (no leakage). Spearman decides among models that pass the gate. |
| Pace model target | **Pace delta in seconds** | Fuel/tyre-corrected lap-time delta to the field best, given the driver finishes. Sigma is interpretable (s/lap). Grid position enters as a feature weighted by track overtaking difficulty. |
| Training window | **2018+ full detail** | FastF1 telemetry era for all features. Pre-2018 Jolpica results used only for long-horizon priors (track DNF/SC rates, driver experience). 2026 reset handling per `CLAUDE.md`. |
| Odds baseline | **Free odds API** (The Odds API or similar) | Pre-race win/podium markets pulled automatically where covered; implied probabilities (overround-corrected) as an eval baseline column. Needs `ODDS_API_KEY`. Not in the critical path of the model pipeline. **2026-10-07 live check: The Odds API has NO Formula 1 (or any motorsport) coverage — verified against its live /v4/sports catalog; F1 is an open feature request on their tracker. The integration code and key both work; the baseline stays dormant until they add F1 or we pick a provider that has it (new data source = ask first, per CLAUDE.md).** |

## Infrastructure

| Decision | Choice | Notes |
|---|---|---|
| App database | **SQLite** via SQLAlchemy | Users, sessions, cached prediction snapshots. Single file, backed up by copying. A later Postgres move is a connection-string change. Model data stays in parquet. |
| Deployment | **Docker Compose + Caddy** | App container (uvicorn) + Caddy with automatic HTTPS for the duckdns domain. Scheduled refresh via host cron invoking the pipeline in the app container. Files in `deploy/`. |

## Derived conventions

- The eval report table (per `CLAUDE.md`) gets columns: model, grid baseline,
  odds baseline (blank where no market coverage) × metrics (Spearman, log-loss,
  Brier, top-3 hit rate) — never a single number on its own.
- DNF model trains on binary log-loss; pace model on Gaussian NLL so sigma is
  learned, not bolted on. These are *training* losses; model *selection* uses
  the Spearman + guardrail rule above.
- Web app code lives in `src/vroom/` (package `vroom`); the model pipeline
  stays in `src/f1pred/` and never imports from `vroom`.
