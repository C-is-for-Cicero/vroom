#!/usr/bin/env bash
# One-shot local setup for Vroom (WSL/Linux).
#
#   bash scripts/setup_local.sh [--data-dir DIR] [--seasons "2018 ... 2026"]
#
# Creates .venv, installs the package, pulls all Jolpica + FastF1 data,
# builds the feature tables (the FastF1 part takes 1-2 hours on first run;
# everything is cached, so re-runs are fast), and snapshots bookmaker odds
# when ODDS_API_KEY is set.
#
# The data directory defaults to ~/vroom-data (NOT the repo folder): on WSL
# the repo often lives under /mnt/c, where the FastF1 cache's thousands of
# small files are painfully slow. The export is added to ~/.bashrc so every
# later shell agrees on where the data lives.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${F1PRED_DATA_DIR:-$HOME/vroom-data}"
SEASONS="2018 2019 2020 2021 2022 2023 2024 2025 2026"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --seasons)  SEASONS="$2";  shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# --- python ----------------------------------------------------------------
PY=python3
for candidate in python3.12 python3.11; do
  command -v "$candidate" >/dev/null && PY="$candidate" && break
done
"$PY" - <<'EOF' || { echo "need Python 3.11+ (sudo apt install python3.11-venv)"; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 11) else 1)
EOF

echo "==> venv + install ($PY)"
[[ -d .venv ]] || "$PY" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev]"

# --- data location ----------------------------------------------------------
export F1PRED_DATA_DIR="$DATA_DIR"
mkdir -p "$DATA_DIR"
if ! grep -qs "F1PRED_DATA_DIR" "$HOME/.bashrc"; then
  echo "export F1PRED_DATA_DIR=\"$DATA_DIR\"" >> "$HOME/.bashrc"
  echo "==> added F1PRED_DATA_DIR to ~/.bashrc"
fi
echo "==> data directory: $DATA_DIR"

# --- tests first: fail fast if the environment is broken --------------------
echo "==> running test suite"
python -m pytest -q

# --- data pulls --------------------------------------------------------------
echo "==> Jolpica ingest (cached; quick after the first run)"
for season in $SEASONS; do
  python -m f1pred.ingest --season "$season"
done

echo "==> feature tables + FastF1 race laps & practice long runs"
echo "    (first run downloads ~200 sessions: 1-2 hours; safe to re-run if interrupted)"
FIRST=${SEASONS%% *}; LAST=${SEASONS##* }
python -m f1pred.features --seasons "$FIRST-$LAST" --race-pace

# --- odds (optional) ---------------------------------------------------------
if [[ -n "${ODDS_API_KEY:-}" ]]; then
  echo "==> bookmaker odds snapshot"
  python -m f1pred.ingest.odds --snapshot --build
else
  echo "==> ODDS_API_KEY not set - skipping odds snapshot (export it and re-run later)"
fi

cat <<'EOF'

Done. Try it out:

  source .venv/bin/activate
  python -m f1pred.predict --season 2026 --round 20 --mode pre_quali
  python -m f1pred.sim.championship --season 2026
  python -m f1pred.eval.sim_eval --mode both --min-train-races 30
  uvicorn vroom.app:app --reload      # web app on http://localhost:8000
  python -m vroom.manage create-admin <you>   # before opening the web app
EOF
