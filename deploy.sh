#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ID="${PROJECT_ID:-ai-dojo-two26hnd-5011}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-music-chord-analyzer}"
RUNTIME_SA_NAME="${RUNTIME_SA_NAME:-music-chord-analyzer-sa}"
MODEL_NAME="${MODEL_NAME:-gemini-2.5-flash}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

test -f Dockerfile || { echo "ERROR: $SCRIPT_DIR に Dockerfile がありません。" >&2; exit 1; }
test -f main.py || { echo "ERROR: $SCRIPT_DIR に main.py がありません。" >&2; exit 1; }
test -f static/index.html || { echo "ERROR: $SCRIPT_DIR に static/index.html がありません。" >&2; exit 1; }

echo "[1/8] Google Cloudプロジェクトを設定します: $PROJECT_ID"
gcloud config set project "$PROJECT_ID" >/dev/null

echo "[2/8] 必要なAPIを有効化します。"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  --project="$PROJECT_ID"

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "[3/8] Cloud Run用サービスアカウントを確認します: $RUNTIME_SA"
if ! gcloud iam service-accounts describe "$RUNTIME_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUNTIME_SA_NAME" \
    --project="$PROJECT_ID" \
    --display-name="Music Chord Analyzer Cloud Run"
fi

echo "[4/8] Cloud Run実行サービスアカウントへVertex AI権限を付与します。"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  >/dev/null

echo "[5/8] デプロイ実行者へサービスアカウント使用権限を付与します。"
DEPLOYER_ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [[ "$DEPLOYER_ACCOUNT" == *.gserviceaccount.com ]]; then
  DEPLOYER_MEMBER="serviceAccount:${DEPLOYER_ACCOUNT}"
else
  DEPLOYER_MEMBER="user:${DEPLOYER_ACCOUNT}"
fi
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA"   --project="$PROJECT_ID"   --member="$DEPLOYER_MEMBER"   --role="roles/iam.serviceAccountUser"   >/dev/null

echo "[6/8] Cloud Build用サービスアカウントを確認します。"
if gcloud iam service-accounts describe "$BUILD_SA" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${BUILD_SA}" \
    --role="roles/run.builder" \
    --condition=None \
    >/dev/null
else
  echo "WARNING: $BUILD_SA がまだ作成されていないため、roles/run.builder の明示付与を省略します。"
  echo "         source deployで権限エラーになった場合は、CLOUD_SHELL_DEPLOY.mdの対処を実行してください。"
fi

echo "[7/8] Cloud Runへ独立サービスとしてデプロイします。"
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
  --concurrency=2 \
  --timeout=900s \
  --min-instances=0 \
  --max-instances=3 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GEMINI_MODEL=${MODEL_NAME},LOG_LEVEL=INFO" \
  --quiet

echo "[8/8] デプロイ結果を確認します。"
SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --format='value(status.url)')"

curl --fail --silent --show-error "${SERVICE_URL}/api/health"
printf '\n\nDEPLOYED_URL=%s\n' "$SERVICE_URL"
