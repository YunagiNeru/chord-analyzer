#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/neural_chord_engine"

cd "$ROOT_DIR"

python -m py_compile \
  "$EXPERIMENT_DIR/models.py" \
  "$EXPERIMENT_DIR/postprocess.py" \
  "$EXPERIMENT_DIR/engines.py" \
  "$EXPERIMENT_DIR/evaluate.py" \
  "$EXPERIMENT_DIR/tests/test_postprocess.py"

python -m unittest discover \
  -s "$EXPERIMENT_DIR/tests" \
  -p 'test_*.py' \
  -v

bash -n \
  "$EXPERIMENT_DIR/setup_cloud_shell.sh" \
  "$EXPERIMENT_DIR/check.sh"

printf '\nNEURAL_CHORD_EXPERIMENT_CHECK=OK\n'
