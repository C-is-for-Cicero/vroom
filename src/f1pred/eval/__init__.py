"""Walk-forward splits, metrics, and baselines.

Model selection: Spearman rank correlation is the tie-breaker; a model ships
only if win/podium/points log-loss beats the grid baseline (hard guardrail),
with isotonic calibration fitted walk-forward. See docs/DECISIONS.md.
"""
