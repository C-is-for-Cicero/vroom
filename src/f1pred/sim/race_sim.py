"""Monte Carlo race simulator.

Each simulation run samples DNFs, adds Gaussian pace noise per driver, and
sorts the field; the output is the probability distribution over finishing
positions for every driver. Fully vectorised over (n_sims, n_drivers) —
no Python loop over sims — and seeded, so runs are reproducible.

Model of one simulated race:
    score_d = mu_d + eps_d + grid_effect_s * grid_slot_d
with eps_d ~ N(0, sigma_d). Lower score = better finish. DNF'd cars are
placed behind every finisher, in random order among themselves (retirement
lap is not modelled). grid_effect_s expresses track position in s/lap
equivalents; 0 disables it (pre_quali mode, where no grid exists). Per-track
overtaking difficulty replaces the single scalar in build-order step 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1pred.config import N_SIMS_DEFAULT, SIM_SEED_DEFAULT

_DNF_SCORE_OFFSET = 1e6


@dataclass(frozen=True)
class RaceSimResult:
    """Simulation output; row order matches the input arrays.

    pos_probs[d, k] = P(driver d finishes in position k+1), counting DNFs
    as occupying the final positions.
    """

    pos_probs: np.ndarray  # (n_drivers, n_drivers)
    p_dnf: np.ndarray  # the input DNF probabilities (for reporting)
    p_win: np.ndarray
    p_podium: np.ndarray
    p_points: np.ndarray  # top 10
    exp_position: np.ndarray
    ranks: np.ndarray | None = None  # (n_sims, n_drivers), 0 = winner; on request


def simulate_race(
    mu: np.ndarray,
    sigma: np.ndarray,
    p_dnf: np.ndarray,
    grid: np.ndarray | None = None,
    grid_effect_s: float = 0.0,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = SIM_SEED_DEFAULT,
    return_ranks: bool = False,
) -> RaceSimResult:
    """Simulate one race n_sims times.

    Inputs: per-driver pace delta mu (s/lap), pace noise sigma (s/lap),
    DNF probability, and optionally grid slots (1-based; 0 = pit lane,
    treated as last slot) with a grid_effect_s weight. Output: RaceSimResult.
    All arrays must share the same length and order.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    p_dnf = np.asarray(p_dnf, dtype=float)
    n = len(mu)
    rng = np.random.default_rng(seed)

    score = mu[None, :] + rng.standard_normal((n_sims, n)) * sigma[None, :]
    if grid is not None and grid_effect_s != 0.0:
        slots = np.asarray(grid, dtype=float).copy()
        slots[slots == 0] = n  # pit-lane start treated as back of the field
        score = score + grid_effect_s * slots[None, :]

    dnf = rng.random((n_sims, n)) < p_dnf[None, :]
    score = np.where(dnf, _DNF_SCORE_OFFSET + rng.random((n_sims, n)), score)

    # rank per driver within each sim: 0 = winner
    ranks = np.argsort(np.argsort(score, axis=1, kind="stable"), axis=1)

    pos_probs = np.empty((n, n), dtype=float)
    for d in range(n):  # loop over drivers (~20), never over sims
        pos_probs[d] = np.bincount(ranks[:, d], minlength=n) / n_sims

    positions = np.arange(1, n + 1, dtype=float)
    return RaceSimResult(
        pos_probs=pos_probs,
        p_dnf=p_dnf,
        p_win=pos_probs[:, 0].copy(),
        p_podium=pos_probs[:, :3].sum(axis=1),
        p_points=pos_probs[:, :10].sum(axis=1),
        exp_position=pos_probs @ positions,
        ranks=ranks if return_ranks else None,
    )


def fit_grid_effect(
    mu_by_race: list[np.ndarray],
    grid_by_race: list[np.ndarray],
    finish_by_race: list[np.ndarray],
    candidates: np.ndarray | None = None,
) -> float:
    """Pick the grid weight (s/lap per grid slot) that best orders past races.

    Inputs: per-race arrays of predicted pace mu, grid slots, and actual
    finishing positions (NaN for non-finishers). Scores each candidate by the
    mean Spearman of rank(mu + a*grid) vs finish; returns the best candidate.
    Leakage: callers must pass training races only.
    """
    from scipy.stats import spearmanr

    if candidates is None:
        candidates = np.arange(0.0, 0.31, 0.02)
    best_a, best_score = 0.0, -np.inf
    for a in candidates:
        scores = []
        for mu, grid, finish in zip(mu_by_race, grid_by_race, finish_by_race, strict=True):
            slots = grid.astype(float).copy()
            slots[slots == 0] = len(slots)
            mask = ~np.isnan(finish)
            if mask.sum() < 3:
                continue
            rho = spearmanr((mu + a * slots)[mask], finish[mask]).statistic
            scores.append(rho)
        mean_rho = float(np.mean(scores)) if scores else -np.inf
        if mean_rho > best_score:
            best_a, best_score = float(a), mean_rho
    return best_a
