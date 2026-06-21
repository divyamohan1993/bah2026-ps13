#!/usr/bin/env bash
# NETRA end-to-end demo launcher.
#
# Runs the full offline pipeline over the four validation scenarios and prints an
# operator-style Q1/Q2/Q3 + lead-time report per scenario plus a summary table.
# Fully offline, CPU-only, template-fallback copilot (no GPU / model / internet).
#
# Usage:
#   scripts/run_demo.sh                      # all four scenarios
#   scripts/run_demo.sh --duration 900       # shorter/faster run
#   scripts/run_demo.sh --scenario A         # one scenario
#   scripts/run_demo.sh --json /tmp/out.json # also dump a JSON summary
#
# Any arguments are passed straight through to scripts/demo.py.
set -euo pipefail

# Resolve the repo root (this script lives in scripts/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"

# Offline-by-design env: never reach out to a model hub even if the heavy tier is
# present (the demo path uses the template fallback regardless).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Pick a Python: prefer an explicit $PYTHON, else python3, else python.
PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PY="python3"; else PY="python"; fi
fi

cd "${REPO}"
exec env PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}" "${PY}" scripts/demo.py "$@"
