#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/neural_chord_engine"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/chord-neural-experiment}"
BEAT_MODEL="${BEAT_MODEL:-small0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  ffmpeg \
  fluidsynth \
  libsndfile1-dev \
  pkg-config

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "numpy>=1.26,<2.1" Cython
python -m pip install -r "$EXPERIMENT_DIR/requirements.txt"

if [[ "${SKIP_OMNIZART_CHECKPOINTS:-0}" != "1" ]]; then
  omnizart download-checkpoints
fi

python - <<PY
from beat_this.inference import load_model

model = load_model("$BEAT_MODEL", device="cpu")
print(f"Beat This! model loaded: {model.__class__.__name__}")
PY

python - <<'PY'
import importlib.metadata

for package in ("omnizart", "beat-this", "mir_eval", "librosa"):
    print(f"{package}={importlib.metadata.version(package)}")
PY

printf '\nEnvironment ready: %s\n' "$VENV_DIR"
printf 'Activate with: source %q/bin/activate\n' "$VENV_DIR"
