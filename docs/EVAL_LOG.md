# Walk-forward evaluation log

Every model change reruns the walk-forward eval and appends its table here
(CLAUDE.md working rules). Reproduce with the command shown per entry.

## 2026-10-07 — step 2: first post-quali pace model vs grid baseline

`python -m f1pred.eval.pace_eval --min-train-races 30` — data: 2023–2026
(2026 through round 16), 1668 rows with pace target over 86 races, 56 test
races, model refitted per test race on strictly earlier races.

| model | Spearman ↑ | top-3 hit rate ↑ | pace RMSE (s/lap) ↓ | races |
|---|---|---|---|---|
| baseline_grid | **0.7474** | **0.6786** | — | 56 |
| pace_model_post_quali | 0.7359 | 0.6667 | 0.5130 | 56 |

By season:

| season | model | Spearman | top-3 | RMSE |
|---|---|---|---|---|
| 2024 | baseline_grid | 0.7297 | 0.6250 | — |
| 2024 | pace_model_post_quali | 0.7223 | **0.6458** | 0.4755 |
| 2025 | baseline_grid | **0.7358** | **0.7500** | — |
| 2025 | pace_model_post_quali | 0.7102 | 0.6944 | 0.4565 |
| 2026 | baseline_grid | 0.7826 | 0.6250 | — |
| 2026 | pace_model_post_quali | **0.7882** | **0.6458** | 0.6352 |

Reading:

- Ranking drivers by predicted *underlying pace* alone slightly trails the
  grid baseline overall — expected before the race simulator exists: the
  grid encodes track position, which dominates finishes where overtaking is
  hard. Converting pace+sigma into finish order (grid-holding by overtaking
  difficulty, DNFs) is build-order step 3's job; this table is the step-2
  reference the simulator must beat.
- 2026 (regulation reset) is the one season where pace-ranking already edges
  the grid — consistent with a less settled field. 2026 pace RMSE is ~0.16 s
  higher than 2024–25: the new cars are genuinely noisier, matching the
  CLAUDE.md guidance to trust current-season data over carried-over priors.
- Log-loss / Brier columns (and the shipping guardrail) start at step 3,
  when probabilities exist. Odds-baseline column joins when odds ingestion
  lands.

## 2026-10-07 — pre_quali mode pulled forward (primary product mode)

Clarified: the headline product is the PRE-qualifying prediction. First
pre-quali model built from prior-race features only (form, past race pace,
championship points — no FP data yet). Its baseline is championship-standings
order; the grid does not exist before qualifying.

`python -m f1pred.eval.pace_eval --mode both --min-train-races 30` — same
dataset as above, 56 test races.

| mode | model | Spearman ↑ | top-3 hit rate ↑ | pace RMSE ↓ |
|---|---|---|---|---|
| pre_quali | baseline_standings | **0.6828** | **0.5774** | — |
| pre_quali | pace_model_pre_quali | 0.6653 | 0.5119 | 0.5702 |
| post_quali | baseline_grid | **0.7474** | **0.6786** | — |
| post_quali | pace_model_post_quali | 0.7359 | 0.6667 | 0.5130 |

Reading:

- The v1 pre-quali model trails the standings baseline. Expected: its
  features are rolling aggregates of past results — nearly the same
  information the championship table already summarises, with extra noise.
  It has no within-weekend signal yet.
- The known lever is FP data, which IS available before qualifying:
  long-run pace, degradation and car profile from practice (build-order
  step 5 features, feeding the step-7 pre-quali model). That is where the
  pre-quali edge over the standings table has to come from, plus the
  simulator (step 3) for order conversion and probabilities.
- Per season, 2026 again narrows the gap (0.7251 vs 0.7237 Spearman): in
  the reset season the standings table knows less.

## 2026-10-07 — step 3: DNF model + race simulator (first probabilities)

`python -m f1pred.eval.sim_eval --mode both --min-train-races 30` — 10,000
sims/race, 56 test races; isotonic calibration fitted walk-forward on the
model's own out-of-sample predictions (first 10 races stay raw). Baseline
orderings are converted to probabilities via train-race frequencies per rank
(Laplace-smoothed), so both sides score on log-loss fairly.

pre_quali (primary mode; baseline = standings order):

| model | Spearman | top-3 | ll win | ll podium | ll points |
|---|---|---|---|---|---|
| baseline_standings | **0.6828** | **0.5774** | 0.1649 | **0.2849** | **0.5515** |
| sim_pre_quali_calibrated | 0.6587 | 0.4940 | **0.1452** | 0.2918 | 0.5869 |

post_quali (baseline = grid order):

| model | Spearman | top-3 | ll win | ll podium | ll points |
|---|---|---|---|---|---|
| baseline_grid | 0.7474 | **0.6786** | 0.1184 | **0.2319** | **0.4967** |
| sim_post_quali_calibrated | **0.7518** | 0.6369 | **0.1085** | 0.2463 | 0.5505 |

Guardrail status (ship only if win/podium/points log-loss all beat the
baseline): **not passed yet** — win passes in both modes, podium is close,
points fails. Calibration already cuts points log-loss sharply
(post: 0.721 raw → 0.551).

Reading:

- First baseline win on the primary order metric: the post-quali simulator
  (pace + fitted grid-effect + DNFs) beats the grid baseline on Spearman —
  the conversion layer works.
- Both modes already predict the WINNER better than their baselines
  (win log-loss and Brier) — pace knowledge is sharpest at the front.
- Points-finish probabilities are the weak spot. Suspected causes: sigma
  underestimation in the midfield (trained on in-sample absolute
  residuals), independent DNF draws (no safety-car correlation), and random
  ordering among DNFs. These are the step-3 refinement targets, alongside
  per-track overtaking difficulty (step 5) replacing the single fitted
  grid-effect scalar.
