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

## 2026-10-07 — step 5 v1: FP long-run car profile + per-track grid effect

FP long-run features (best-stint pace delta, degradation slope,
consistency from FP2/FP3/FP1) for 85/86 weekends; simulator grid-effect
weight now scaled per circuit by circuit_grid_hold. Same 56 test races.

Pace model (`pace_eval --mode both --min-train-races 30`):

| mode | model | Spearman | top-3 | RMSE |
|---|---|---|---|---|
| pre_quali | baseline_standings | **0.6828** | **0.5774** | — |
| pre_quali | pace_model (now with FP) | 0.6718 (was 0.6653) | 0.5357 (was 0.5119) | 0.5703 |
| post_quali | baseline_grid | **0.7474** | **0.6786** | — |
| post_quali | pace_model (now with FP) | 0.7363 | 0.6548 | 0.5187 |

Simulator (`sim_eval --mode both --min-train-races 30`), calibrated:

| mode | model | Spearman | ll win | ll podium | ll points |
|---|---|---|---|---|---|
| pre_quali | baseline_standings | **0.6828** | 0.1649 | **0.2849** | **0.5515** |
| pre_quali | sim calibrated | 0.6652 | **0.1487** | 0.2991 | 0.6020 |
| post_quali | baseline_grid | 0.7474 | 0.1184 | 0.2319 | **0.4967** |
| post_quali | sim calibrated | **0.7505** | **0.1153** | **0.2292** | 0.5665 |

Guardrail status: post_quali now passes **win and podium** (podium for the
first time); points still fails in both modes. pre_quali passes win only.

Reading:

- FP long runs are real signal: pre-quali Spearman gap to the standings
  baseline narrowed (0.0175 → 0.0110) and top-3 jumped (+0.024), from
  within-weekend data alone, with 2025 now won outright. More FP signal is
  available (telemetry corner profile, step 5 full) — this was lap times only.
- The per-track grid effect + FP features keep the post-quali sim ahead on
  Spearman and pushed podium log-loss past the baseline.
- Points-finish probabilities remain the open guardrail item. The two
  known mechanisms not yet modelled: correlated retirements (safety-car
  chaos hits several cars at once) and midfield sigma underestimation
  (sigma trains on in-sample residuals). Next modelling targets.

## 2026-10-07 — honest sigma (OOF) + correlated DNFs (gamma frailty)

Sigma now trains on out-of-fold absolute residuals from time-ordered folds
with the half-normal correction; the simulator draws a per-sim chaos
multiplier (gamma, variance moment-matched to training DNF-count
overdispersion), so retirements correlate within a race.

`sim_eval --mode both --min-train-races 30`, same 56 test races (raw sim
rows; calibrated in parentheses where different):

| mode | model | Spearman | ll win | ll podium | ll points |
|---|---|---|---|---|---|
| pre_quali | baseline_standings | **0.6828** | 0.1649 | 0.2849 | **0.5515** |
| pre_quali | sim raw | 0.6626 | **0.1405** | **0.2741** | 0.5525 |
| post_quali | baseline_grid | **0.7474** | 0.1184 | **0.2319** | **0.4967** |
| post_quali | sim raw | 0.7454 | **0.1122** | 0.2381 | 0.5108 |

Guardrail: pre_quali now passes win AND podium and ties points (Δ 0.001);
post_quali passes win, misses podium by 0.006 and points by 0.014. Points
log-loss improved dramatically (pre raw: 0.743 → 0.553; post: 0.778 → 0.511).

Note on calibration: with honest sigma + frailty, the RAW simulator is
already well calibrated and isotonic now mostly degrades log-loss (it
overfits limited history). Proposal: make calibration adaptive — apply it
only where it improved walk-forward log-loss on past races; display layer
currently should use raw. Flagged for decision.

Trade-off: post-quali Spearman slipped from 0.7518 to 0.7454 (wider sigma
flattens expected-position differences). Net: probability quality bought
at a small order-metric cost; the probability metrics are the product.

## 2026-10-07 — step 6: fastest-lap classifier

Multinomial-over-the-field via per-race normalised binary LightGBM
(post-quali features). `python -m f1pred.eval.fastest_lap_eval
--min-train-races 30`, 56 test races:

| model | log-loss ↓ | hit@1 | hit@3 |
|---|---|---|---|
| uniform (1/n) | 3.0220 | 0.3393 | 0.3750 |
| grid-frequency baseline | 2.6284 | **0.3036** | **0.5179** |
| fastest-lap model (pure) | 3.0742 | 0.2679 | 0.5000 |
| **fastest-lap blend (0.5 model + 0.5 grid-freq, fixed a priori)** | **2.4558** | 0.2857 | 0.4821 |

The pure classifier is overconfident and loses to the frequency baseline;
the fixed 50/50 ensemble with that prior beats it on log-loss. The blend
is the shipping candidate for the fastest-lap market. Fastest lap is
intrinsically noisy (no bonus point since 2025, often set on a late free
pit stop) — hit@1 ≈ 0.30 against 20 drivers is the realistic ceiling area.

## 2026-10-09 — step 5 full: telemetry car profile + fingerprint (infrastructure)

Implements the car-profile × track-fingerprint design: FP telemetry is
segmented into corner zones from the circuit's corner markers, classified
slow/med/fast by apex minimum speed, and each driver's fastest FP lap is
compared to the session best per class (time deltas), plus top-speed delta
and energy fade on the longest straight. The same reference lap yields the
track fingerprint (corner-class time shares, straight share, % full
throttle, heavy-brake count). Interactions (deficit × share) feed both
pace models. Cutoff: end of FP — valid pre-quali.

Validated on real data (2026 Baku: NOR/ANT/VER lose least slow-corner
time; midfield loses 2.2-2.5 s). Eval is UNCHANGED at this entry
(pre-quali Spearman 0.6718): historical telemetry coverage is 2 events,
so the features are inert until the backfill runs
(`python -m f1pred.features --seasons 2018-2026 --race-pace --telemetry`,
~50-150 MB per weekend, FastF1-budget-aware and resumable). The
before/after table lands once coverage exists.
