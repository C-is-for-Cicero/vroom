"""Car profile × track fingerprint interactions (CLAUDE.md step 5).

A car's corner-class deficit matters in proportion to how much of the lap
the track spends in that class: ix_slow = time lost in slow corners at the
profiling event × this track's slow-corner time share, and so on. Straight
interaction pairs top-speed deficit with the track's straight share.

Leakage cutoff: both factors come from free practice of race R (driver
profile) and the same event's reference lap (track fingerprint) — known
before qualifying, valid in both modes. NaNs propagate (LightGBM handles
missing natively), so rounds without telemetry simply lack the features.
"""

from __future__ import annotations

import pandas as pd

INTERACTION_FEATURES = ["ix_slow", "ix_med", "ix_fast", "ix_straight"]


def add_telemetry_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add ix_* columns; requires tel_* and trk_* columns (NaN where absent)."""
    df = df.copy()
    for cls in ("slow", "med", "fast"):
        df[f"ix_{cls}"] = df[f"tel_{cls}_s"] * df[f"trk_{cls}_share"]
    # tel_top_speed is a delta (negative = slower); scale by straight share
    df["ix_straight"] = -df["tel_top_speed"] * df["trk_straight_share"]
    return df
