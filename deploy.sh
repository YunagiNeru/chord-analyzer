#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-ai-dojo-july}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-music-chord-analyzer}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-music-chord-analyzer-sa}"
MODEL_NAME="${MODEL_NAME:-gemini-3.5-flash}"
RESOLVER_MODEL_NAME="${RESOLVER_MODEL_NAME:-$MODEL_NAME}"
ANALYSIS_PIPELINE="${ANALYSIS_PIPELINE:-v2}"
MODEL_MAX_PARALLEL_CALLS="${MODEL_MAX_PARALLEL_CALLS:-2}"
MAX_RESOLVER_CALLS="${MAX_RESOLVER_CALLS:-12}"
RESOLVER_BATCH_SIZE="${RESOLVER_BATCH_SIZE:-8}"
MAX_RESOLVER_RECOVERY_CALLS="${MAX_RESOLVER_RECOVERY_CALLS:-16}"
ENABLE_REFERENCE_RESEARCH="${ENABLE_REFERENCE_RESEARCH:-0}"
STRICT_QUALITY_GATE="${STRICT_QUALITY_GATE:-0}"
MAX_DEGRADED_UNRESOLVED_RATIO="${MAX_DEGRADED_UNRESOLVED_RATIO:-0.40}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"
RUN_LOCAL_TESTS="${RUN_LOCAL_TESTS:-1}"
YOUTUBE_SECRET_NAME="${YOUTUBE_SECRET_NAME:-chord-analyzer-youtube-api-key}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

test -f Dockerfile || { echo "ERROR: $SCRIPT_DIR に Dockerfile がありません。" >&2; exit 1; }
test -f main.py || { echo "ERROR: $SCRIPT_DIR に main.py がありません。" >&2; exit 1; }
test -f static/index.html || { echo "ERROR: $SCRIPT_DIR に static/index.html がありません。" >&2; exit 1; }
test -f music_agent/accuracy_v2_release_mode.py || { echo "ERROR: release mode module is missing." >&2; exit 1; }
test -f music_agent/accuracy_v2_resolver_recovery.py || { echo "ERROR: resolver recovery module is missing." >&2; exit 1; }
command -v gcloud >/dev/null || { echo "ERROR: gcloud CLI がありません。" >&2; exit 1; }
command -v curl >/dev/null || { echo "ERROR: curl がありません。" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null || { echo "ERROR: Python executable not found: $PYTHON_BIN" >&2; exit 1; }

COMMIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
COMMIT_SHORT="${COMMIT_SHA:0:12}"

echo "[preflight] commit=$COMMIT_SHA"
"$PYTHON_BIN" -m compileall -q main.py music_agent scripts tests
if [[ "$RUN_LOCAL_TESTS" == "1" ]]; then
  echo "[preflight] unit tests (strict quality gate)"
  STRICT_QUALITY_GATE=1 "$PYTHON_BIN" -m unittest discover -s tests -v
fi

echo "[1/9] Google Cloudプロジェクトを設定します: $PROJECT_ID"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "[2/9] 必要なAPIを有効化します。"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  secretmanager.googleapis.com \
  --project="$PROJECT_ID"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "[3/9] Cloud Run用サービスアカウントを確認します: $RUNTIME_SA"
if ! gcloud iam service-accounts describe "$RUNTIME_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUNTIME_SA_NAME" \
    --project="$PROJECT_ID" \
    --display-name="Music Chord Analyzer Cloud Run"
fi

echo "[4/9] Cloud Run実行サービスアカウントへVertex AI権限を付与します。"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  >/dev/null

echo "[5/9] YouTube Data APIキーSecretを任意設定します。"
SECRET_ARGS=()
if gcloud secrets describe "$YOUTUBE_SECRET_NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud secrets add-iam-policy-binding "$YOUTUBE_SECRET_NAME" \
    --project="$PROJECT_ID" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    >/dev/null
  SECRET_ARGS+=("--set-secrets=YOUTUBE_API_KEY=${YOUTUBE_SECRET_NAME}:latest")
  echo "YouTube metadata secret enabled: $YOUTUBE_SECRET_NAME"
else
  echo "INFO: $YOUTUBE_SECRET_NAME は未作成です。oEmbedとモデル推定へフォールバックします。"
fi

echo "[6/9] デプロイ実行者へサービスアカウント使用権限を付与します。"
DEPLOYER_ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [[ -z "$DEPLOYER_ACCOUNT" || "$DEPLOYER_ACCOUNT" == "(unset)" ]]; then
  echo "ERROR: gcloud authenticated account is not configured." >&2
  exit 1
fi
if [[ "$DEPLOYER_ACCOUNT" == *.gserviceaccount.com ]]; then
  DEPLOYER_MEMBER="serviceAccount:${DEPLOYER_ACCOUNT}"
else
  DEPLOYER_MEMBER="user:${DEPLOYER_ACCOUNT}"
fi
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --project="$PROJECT_ID" \
  --member="$DEPLOYER_MEMBER" \
  --role="roles/iam.serviceAccountUser" \
  >/dev/null

echo "[7/9] Cloud Build用サービスアカウントを確認します。"
if gcloud iam service-accounts describe "$BUILD_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${BUILD_SA}" \
    --role="roles/run.builder" \
    --condition=None \
    >/dev/null
else
  echo "WARNING: $BUILD_SA が未作成のため roles/run.builder の明示付与を省略します。"
fi

echo "[8/9] Cloud Runへ本番プロファイルをデプロイします。"
gcloud run deploy "$SERVICE_NAME" \
  --source=. \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --platform=managed \
  --execution-environment=gen2 \
  --port=8080 \
  --allow-unauthenticated \
  --service-account="$RUNTIME_SA" \
  --cpu=2 \
  --memory=4Gi \
  --cpu-boost \
  --concurrency=1 \
  --timeout=900s \
  --min-instances=0 \
  --max-instances="$MAX_INSTANCES" \
  --labels="app=music-chord-analyzer,pipeline=accuracy-v2" \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GEMINI_MODEL=${MODEL_NAME},GEMINI_RESOLVER_MODEL=${RESOLVER_MODEL_NAME},ANALYSIS_PIPELINE=${ANALYSIS_PIPELINE},MODEL_MAX_PARALLEL_CALLS=${MODEL_MAX_PARALLEL_CALLS},MAX_RESOLVER_CALLS=${MAX_RESOLVER_CALLS},RESOLVER_BATCH_SIZE=${RESOLVER_BATCH_SIZE},MAX_RESOLVER_RECOVERY_CALLS=${MAX_RESOLVER_RECOVERY_CALLS},ENABLE_REFERENCE_RESEARCH=${ENABLE_REFERENCE_RESEARCH},STRICT_QUALITY_GATE=${STRICT_QUALITY_GATE},MAX_DEGRADED_UNRESOLVED_RATIO=${MAX_DEGRADED_UNRESOLVED_RATIO},MODEL_RETRY_BASE_SECONDS=1.5,MODEL_RETRY_MAX_SECONDS=12,APP_RELEASE_SHA=${COMMIT_SHORT},LOG_LEVEL=INFO" \
  "${SECRET_ARGS[@]}" \
  --quiet

echo "[9/9] デプロイ結果を確認します。"
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --format='value(status.url)')"

curl \
  --fail \
  --silent \
  --show-error \
  --retry 10 \
  --retry-delay 3 \
  --retry-all-errors \
  --max-time 30 \
  "${SERVICE_URL}/api/health"

printf '\n\nDEPLOYED_URL=%s\nRELEASE_SHA=%s\n' "$SERVICE_URL" "$COMMIT_SHA"
