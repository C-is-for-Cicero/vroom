"""Evaluation metrics. Results are always reported as a table against the
baselines (see eval/baselines.py), never as a single number.

Order metrics (Spearman, top-3 hit rate) apply to a predicted ordering;
probability metrics (log-loss, Brier) apply to simulator outputs and carry
the shipping guardrail: win/podium/points log-loss must beat the grid
baseline (docs/DECISIONS.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_order(predicted_rank: pd.Series, actual_position: pd.Series) -> float:
    """Spearman correlation between a predicted ordering and actual finishing
    positions, over drivers with both values (classified finishers)."""
    mask = predicted_rank.notna() & actual_position.notna()
    if mask.sum() < 3:
        return float("nan")
    rho = spearmanr(predicted_rank[mask], actual_position[mask]).statistic
    return float(rho)


def top3_hit_rate(predicted_rank: pd.Series, actual_position: pd.Series) -> float:
    """Fraction of the actual podium found in the predicted top 3 (0, 1/3, 2/3, 1)."""
    pred_top = set(predicted_rank.nsmallest(3).index)
    actual_top = set(actual_position.nsmallest(3).index)
    return len(pred_top & actual_top) / 3


def binary_log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Mean log-loss for binary outcomes (win / podium / points-finish)."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between binary outcomes and predicted probabilities."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def per_race_summary(rows: list[dict]) -> pd.DataFrame:
    """Aggregate per-race metric dicts into the report table: one row per
    (model, metric) with the mean over races and the race count."""
    df = pd.DataFrame(rows)
    value_cols = [c for c in df.columns if c not in ("model", "season", "round")]
    out = df.groupby("model")[value_cols].mean()
    out["n_races"] = df.groupby("model").size()
    return out.reset_index()
