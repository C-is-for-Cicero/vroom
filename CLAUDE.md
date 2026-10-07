# F1 Prediction Model

Predicts Formula 1 race outcomes: **finishing order**, **fastest lap**, and **constructors' championship**. Outputs are probabilities, not single picks.

## Approach (do not change without discussion)

The model is built as **two models plus a Monte Carlo race simulator**. LambdaRank or a single ranking model is not the approach.

1. **DNF model** – a classifier that gives P(DNF) per driver per race.
2. **Pace model** – a regression of expected pace or finishing position *given the driver finishes*, plus an uncertainty (sigma) per driver.
3. **Race simulator** – runs N sims (default 10,000). Each run samples DNFs, adds pace noise, then sorts the field. The output is the probability distribution over finishing positions for every driver.
4. **Championship simulator** – starts from current standings, simulates every remaining race with the race simulator, applies the points system, and counts title outcomes.
5. **Fastest-lap model** – a separate multinomial classifier over the field. Since 2025 fastest lap earns no bonus point.

There are two prediction modes. Keep them as **separate models**, never one model with a flag:
- `pre_quali`: uses data up to the end of FP.
- `post_quali`: adds grid and quali pace.

## Data sources

| Source | Use | Notes |
|---|---|---|
| Jolpica-F1 API | Results, grid, quali, standings, pit stops (1950+) | Successor to Ergast (Ergast is shut down; never use it). Respect its rate limits and cache all responses locally. |
| FastF1 | Laps, sectors, tyres, stints, weather, telemetry, circuit corners (2018+) | Always enable the cache: `fastf1.Cache.enable_cache("data/cache/fastf1")` |
| OpenF1 | Near-live session data | Optional, for in-weekend updates only |

Raw pulls are never edited by hand. Everything downstream is rebuilt from `data/raw/`.

## Telemetry and track features

Telemetry is sampled at about 4 Hz, which is too coarse for micro analysis. **Always aggregate it per lap segment**, never per sample.

**Car profile** (per driver, per weekend, relative to the session best):
- Corner-type pace: split the lap into slow, medium and fast corners and straights using `session.get_circuit_info()` corner markers. Measure min speed and time lost per type.
- Straight-line speed and energy fade: speed drop before the end of long straights shows battery clipping.
- Long-run degradation: the slope of a fuel-corrected lap-time vs tyre-age fit, from FP stints.
- Consistency: the spread of long-run lap times.

**Track fingerprint** (per circuit):
- % full throttle, time share by corner type, number of heavy braking zones, longest straight.
- From history: position changes per race, safety-car rate, DNF rate, pit-lane time loss.

**Where they meet:**
- Interaction features: car profile × track fingerprint.
- Similar-track lookup: cosine similarity of fingerprints, then use team results at the most similar tracks this season.
- Simulator settings by track: overtaking difficulty controls how strongly grid order holds; SC/DNF rates set the chaos level.

## 2026 regulation reset — important

2026 brought new power units, active aero (DRS removed), and new entrants (Cadillac; Audi replacing Sauber).
- Pre-2026 data has **weak value for car and team pace**. Driver skill ratings carry over better.
- Weight the current season heavily and shrink older team-level features toward the field mean.
- The FastF1 `DRS` channel does **not** mean what it did before 2026. Check what it contains for 2026 sessions before using it as a feature.
- Team identity across the reset: map constructors explicitly (e.g. Sauber → Audi). Don't rely on name matching.

## Validation rules (non-negotiable)

- **Time-ordered splits only.** Use walk-forward by race or season. Never shuffle and never use random K-fold.
- **No leakage.** Every feature for race R must be computable from data available *before* the relevant session of race R. Rolling features use `.shift(1)`.
- **Baselines every model must be compared against:**
  1. Finish = grid position (post-quali).
  2. Bookmaker implied probabilities, where available.
- **Metrics:**
  - Spearman rank correlation for order.
  - Log-loss and Brier score for win, podium and points probabilities.
  - Top-3 hit rate.
  - Calibration plots for probability outputs.
- **Report results** as a table against the baselines, never as a single number on its own.

## Project layout

```
data/
  raw/          # untouched API/FastF1 pulls (gitignored)
  cache/        # FastF1 + HTTP cache (gitignored)
  processed/    # parquet feature tables
src/f1pred/
  ingest/       # jolpica.py, fastf1_loader.py
  features/     # car_profile.py, track_fingerprint.py, form.py, interactions.py
  models/       # dnf.py, pace.py, fastest_lap.py
  sim/          # race_sim.py, championship_sim.py
  eval/         # splits.py, metrics.py, baselines.py
  config.py     # points system, season constants, team mappings
notebooks/      # exploration only — no logic that the pipeline depends on
tests/
```

**Core table grain:** one row per `(season, round, driver)`. Lap-level and telemetry-derived data live in separate tables keyed by `(season, round, session, driver, lap)`.

## Stack and conventions

- **Language and libraries:** Python 3.11+, pandas, numpy, LightGBM, scikit-learn, fastf1, pyarrow; matplotlib for plots.
- **Data formats:** store processed data as parquet, not CSV.
- **Types and docs:** type hints on all public functions, with a short docstring stating inputs, outputs and the leakage cutoff.
- **Randomness:** seed every simulation run (`numpy.random.default_rng(seed)`). Sims must be reproducible.
- **Points system:** keep it in `config.py`, not inline. Sprint points are separate.
- **Vectorise the simulator.** Use numpy arrays shaped `(n_sims, n_drivers)` and never loop over sims in Python.
- **Dev environment:** Windows with WSL2. Run the pipeline inside WSL, use relative paths via `pathlib`, and never hardcode drive letters.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m f1pred.ingest --season 2026          # pull/update data
python -m f1pred.features --season 2026        # rebuild feature tables
python -m f1pred.predict --season 2026 --round <R> --mode post_quali
python -m f1pred.sim.championship --season 2026
pytest
```

## Working rules for Claude

- Before adding a feature, state its leakage cutoff and confirm it is available at prediction time.
- When changing a model, rerun the walk-forward evaluation and show the before/after against the baselines.
- Don't add new data sources or swap the modelling approach without asking.
- Keep notebooks disposable. Move anything reused into `src/`.
- When unsure whether 2026 data behaves like earlier seasons, check the 2026 data first instead of assuming.

## Build order

1. Ingestion (Jolpica + FastF1) with caching
2. Post-quali pace model + grid baseline
3. DNF model + race simulator
4. Championship simulator
5. Telemetry car profile + track fingerprint + interactions
6. Fastest-lap classifier
7. Pre-quali model

## Product decisions

Product/UI/deployment decisions agreed on 2026-10-07 live in `docs/DECISIONS.md`.
Headlines: FastAPI + Jinja + HTMX web app, SQLite for app state, Spearman as the
model-selection tie-breaker with a hard log-loss guardrail vs the grid baseline
and isotonic calibration of displayed probabilities, pace model regresses
fuel/tyre-corrected pace delta in seconds, training window 2018+, deploy via
Docker Compose + Caddy on a Vultr VPS (duckdns domain).
