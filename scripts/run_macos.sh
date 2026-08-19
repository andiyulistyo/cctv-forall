#!/usr/bin/env bash
# Start the dashboard natively on macOS (Apple Silicon accelerated).
#
#   ./scripts/run_macos.sh              # http://localhost:8000
#   PORT=9000 ./scripts/run_macos.sh
#   ./scripts/run_macos.sh --reload     # extra args go to uvicorn
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/backend/.venv"

if [[ ! -x "$VENV/bin/uvicorn" ]]; then
  echo "backend/.venv is missing. Run ./scripts/setup_macos.sh first." >&2
  exit 1
fi

# Settings come from .env at the repo root (see .env.macos.example).
# Only one uvicorn worker: the detection manager owns the child processes and
# the shared frame state, which must live in a single parent process.
cd "$ROOT/backend"
exec "$VENV/bin/uvicorn" app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --workers 1 \
  "$@"
