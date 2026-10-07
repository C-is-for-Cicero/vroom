"""Season constants, points systems, team mappings, and paths.

Everything here is static configuration: no I/O, no model logic. The points
system and team mappings live here (not inline in models/sims) per CLAUDE.md.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — relative to the repo root, overridable for deployment.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("F1PRED_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
FASTF1_CACHE_DIR = CACHE_DIR / "fastf1"
JOLPICA_RAW_DIR = RAW_DIR / "jolpica"

# ---------------------------------------------------------------------------
# Points systems. Sprint points are separate from race points.
# ---------------------------------------------------------------------------

# Standard race points, 2010–present.
RACE_POINTS: dict[int, int] = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Sprint points, 2022–present format.
SPRINT_POINTS: dict[int, int] = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

# Fastest-lap bonus (awarded only if the driver finished in the top 10).
# 2019–2024: 1 point. Since 2025: no bonus point.
FASTEST_LAP_BONUS_SEASONS = range(2019, 2025)


def race_points(position: int) -> int:
    """Championship points for a classified race finishing position."""
    return RACE_POINTS.get(position, 0)


def sprint_points(position: int) -> int:
    """Championship points for a sprint finishing position (2022+ format)."""
    return SPRINT_POINTS.get(position, 0)


def fastest_lap_bonus(season: int, position: int) -> int:
    """Fastest-lap bonus points for `season` given race finishing `position`."""
    if season in FASTEST_LAP_BONUS_SEASONS and position <= 10:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Team identity across seasons. Keys are Jolpica/Ergast constructorIds.
# Mapped explicitly (never by name matching) to a canonical lineage id, so
# team-level features aggregate across renames and the 2026 reset.
# ---------------------------------------------------------------------------

CONSTRUCTOR_LINEAGE: dict[str, str] = {
    # Stable identities
    "ferrari": "ferrari",
    "mclaren": "mclaren",
    "mercedes": "mercedes",
    "red_bull": "red_bull",
    "williams": "williams",
    "haas": "haas",
    # Renault → Alpine
    "renault": "alpine",
    "alpine": "alpine",
    # Force India → Racing Point → Aston Martin
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",
    # Toro Rosso → AlphaTauri → RB (Racing Bulls)
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "rb": "rb",
    # Sauber → Alfa Romeo (branding) → Sauber → Audi (2026)
    "sauber": "audi",
    "alfa": "audi",
    "audi": "audi",
    # New entrant 2026
    "cadillac": "cadillac",
}


def canonical_constructor(constructor_id: str) -> str:
    """Canonical lineage id for a Jolpica constructorId.

    Unknown ids map to themselves so a new entrant never crashes the pipeline;
    add them here explicitly once they appear.
    """
    return CONSTRUCTOR_LINEAGE.get(constructor_id, constructor_id)


# ---------------------------------------------------------------------------
# Season / modelling constants
# ---------------------------------------------------------------------------

CURRENT_SEASON = 2026

# 2026 regulation reset: pre-2026 team-level pace features are shrunk toward
# the field mean (see CLAUDE.md). Driver skill features carry over.
REGULATION_RESET_SEASONS: frozenset[int] = frozenset({2026, 2022, 2014})

# Training window: full-detail features from the FastF1 era onward.
# Earlier Jolpica data is used only for long-horizon priors.
FIRST_DETAILED_SEASON = 2018

# Simulation defaults. Every sim run must be seeded (numpy.random.default_rng).
N_SIMS_DEFAULT = 10_000
SIM_SEED_DEFAULT = 2026
