#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-ai-dojo-july}"
APP_PORT="${APP_PORT:-8080}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/chord-analyzer}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="global"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.5-flash}"
export GEMINI_RESOLVER_MODEL="${GEMINI_RESOLVER_MODEL:-$GEMINI_MODEL}"
export ANALYSIS_PIPELINE="${ANALYSIS_PIPELINE:-v2}"
export MODEL_MAX_PARALLEL_CALLS="${MODEL_MAX_PARALLEL_CALLS:-4}"
export MAX_RESOLVER_CALLS="${MAX_RESOLVER_CALLS:-12}"
export ENABLE_REFERENCE_RESEARCH="${ENABLE_REFERENCE_RESEARCH:-1}"

if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
  echo "Application Default Credentialsがありません。認証を開始します。"
  gcloud auth application-default login
fi

exec uvicorn main:app --host 0.0.0.0 --port "$APP_PORT" --proxy-headers
