"""Alias so the CLAUDE.md command `python -m f1pred.sim.championship` works."""

from f1pred.sim.championship_sim import main

if __name__ == "__main__":
    raise SystemExit(main())
