#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$ROOT_DIR/experiments/neural_chord_engine"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/chord-neural-experiment}"
BEAT_MODEL="${BEAT_MODEL:-small0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
OMNIZART_CHORD_CHECKPOINT_URL="${OMNIZART_CHORD_CHECKPOINT_URL:-https://github.com/Music-and-Culture-Technology-Lab/omnizart/releases/download/checkpoints-20211001/chord_v1@variables.data-00000-of-00001}"

export PIP_NO_CACHE_DIR=1
export TMPDIR="${TMPDIR:-/tmp}"

if [[ "${RESET_VENV:-0}" == "1" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ "${CLEAN_PACKAGE_CACHE:-1}" == "1" ]]; then
  rm -rf "$HOME/.cache/pip" "$HOME/.cache/uv"
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  ffmpeg \
  fluidsynth \
  libsndfile1-dev \
  pkg-config \
  portaudio19-dev

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade --no-cache-dir \
  pip \
  "setuptools<82" \
  wheel

python -m pip install --no-cache-dir \
  "numpy>=1.26,<2.1" \
  Cython

printf '\nInstalling CPU-only PyTorch from %s\n' "$PYTORCH_INDEX_URL"
python -m pip install --no-cache-dir \
  --index-url "$PYTORCH_INDEX_URL" \
  torch \
  torchaudio

python -m pip install --no-cache-dir \
  -r "$EXPERIMENT_DIR/requirements.txt"

if [[ "${SKIP_OMNIZART_CHORD_CHECKPOINT:-0}" != "1" ]]; then
  OMNIZART_MODULE_PATH="$(
    python - <<'PY'
from omnizart import MODULE_PATH

print(MODULE_PATH)
PY
  )"

  CHECKPOINT_DIR="$OMNIZART_MODULE_PATH/checkpoints/chord/chord_v1/variables"
  CHECKPOINT_FILE="$CHECKPOINT_DIR/variables.data-00000-of-00001"

  mkdir -p "$CHECKPOINT_DIR"

  if [[ ! -s "$CHECKPOINT_FILE" ]]; then
    printf '\nDownloading Omnizart chord checkpoint only\n'
    curl \
      --fail \
      --location \
      --retry 5 \
      --retry-delay 2 \
      --output "$CHECKPOINT_FILE.tmp" \
      "$OMNIZART_CHORD_CHECKPOINT_URL"

    test -s "$CHECKPOINT_FILE.tmp"
    mv "$CHECKPOINT_FILE.tmp" "$CHECKPOINT_FILE"
  fi

  printf 'Omnizart chord checkpoint: %s\n' "$CHECKPOINT_FILE"
fi

python - <<PY
from beat_this.inference import load_model

model = load_model("$BEAT_MODEL", device="cpu")
print(f"Beat This! model loaded: {model.__class__.__name__}")
PY

python - <<'PY'
import importlib.metadata

import torch

for package in (
    "omnizart",
    "beat-this",
    "mir_eval",
    "librosa",
    "torch",
    "torchaudio",
):
    print(f"{package}={importlib.metadata.version(package)}")

print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")

if torch.version.cuda is not None:
    raise RuntimeError("CUDA版PyTorchが導入されています。CPU版で再構築してください。")
PY

printf '\nEnvironment size:\n'
du -sh "$VENV_DIR"

printf '\nCloud Shell home storage:\n'
df -h "$HOME"

printf '\nEnvironment ready: %s\n' "$VENV_DIR"
printf 'Activate with: source %q/bin/activate\n' "$VENV_DIR"
