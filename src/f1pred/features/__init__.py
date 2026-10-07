"""Feature tables: car profile, track fingerprint, form, interactions.

Core grain: one row per (season, round, driver). Every feature states its
leakage cutoff; rolling features use .shift(1). Built in later build-order
steps (2, 5).
"""
