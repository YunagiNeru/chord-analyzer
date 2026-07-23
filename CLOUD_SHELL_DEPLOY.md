# Cloud Shell導入・デプロイ手順

対象:

- Google Cloudプロジェクト: `ai-dojo-two26hnd-5011`
- 既存ルート: `~/html-site`
- 新規アプリ配置先: `~/html-site/chord-analyzer`
- 新規Cloud Runサービス: `music-chord-analyzer`
- リージョン: `us-central1`

既存の `~/html-site` 直下ファイルや既存Cloud Runサービス `html-site` は変更しません。

---

## 1. ソースZIPをCloud Shellへアップロード

ChatGPTから `chord-analyzer.zip` をPCへ保存し、Cloud Shell右上の「その他」→「アップロード」からアップロードします。アップロード先は通常 `~` です。

### 作業ディレクトリ

```text
~
```

### コマンド

```bash
cd ~

pwd
ls -lh ~/chord-analyzer.zip
```

### 期待される結果

```text
/home/devstar5011
-rw-r--r-- ... chord-analyzer.zip
```

---

## 2. 既存環境をバックアップして展開

### 作業ディレクトリ

```text
~/html-site
```

### コマンド

```bash
cd ~

test -d ~/html-site || {
  echo "ERROR: ~/html-site が存在しません。"
  exit 1
}

cp -a ~/html-site \
  "$HOME/html-site-before-chord-analyzer-$(date +%Y%m%d-%H%M%S)"

cd ~/html-site

if [ -d chord-analyzer ]; then
  mv chord-analyzer \
    "$HOME/chord-analyzer-before-update-$(date +%Y%m%d-%H%M%S)"
fi

unzip -q ~/chord-analyzer.zip -d ~/html-site

find ~/html-site/chord-analyzer \
  -maxdepth 3 \
  -type f \
  -print \
  | sort
```

### 期待される結果

次のファイルが列挙されます。

```text
/home/devstar5011/html-site/chord-analyzer/.dockerignore
/home/devstar5011/html-site/chord-analyzer/.env.example
/home/devstar5011/html-site/chord-analyzer/.gcloudignore
/home/devstar5011/html-site/chord-analyzer/CLOUD_SHELL_DEPLOY.md
/home/devstar5011/html-site/chord-analyzer/Dockerfile
/home/devstar5011/html-site/chord-analyzer/README.md
/home/devstar5011/html-site/chord-analyzer/deploy.sh
/home/devstar5011/html-site/chord-analyzer/local_run.sh
/home/devstar5011/html-site/chord-analyzer/main.py
/home/devstar5011/html-site/chord-analyzer/music_agent/__init__.py
/home/devstar5011/html-site/chord-analyzer/music_agent/agents.py
/home/devstar5011/html-site/chord-analyzer/music_agent/dsp.py
/home/devstar5011/html-site/chord-analyzer/music_agent/prompts.py
/home/devstar5011/html-site/chord-analyzer/music_agent/schemas.py
/home/devstar5011/html-site/chord-analyzer/requirements.txt
/home/devstar5011/html-site/chord-analyzer/static/index.html
```

---

## 3. 配置前の構文検査

### 作業ディレクトリ

```text
~/html-site/chord-analyzer
```

### コマンド

```bash
cd ~/html-site/chord-analyzer

python3 -m compileall -q .

python3 - <<'PY'
from pathlib import Path
import re

html = Path("static/index.html").read_text(encoding="utf-8")
match = re.search(r"<script>([\s\S]*)</script>\s*</body>", html)
if not match:
    raise RuntimeError("インラインJavaScriptを取得できませんでした。")

Path("/tmp/music-chord-analyzer.js").write_text(
    match.group(1),
    encoding="utf-8",
)
print("/tmp/music-chord-analyzer.js を作成しました。")
PY

node --check /tmp/music-chord-analyzer.js
bash -n deploy.sh local_run.sh

echo "SOURCE_SYNTAX_CHECK=OK"
```

### 期待される結果

```text
/tmp/music-chord-analyzer.js を作成しました。
SOURCE_SYNTAX_CHECK=OK
```

`node --check` と `bash -n` は正常時に余分な出力を返しません。

---

## 4. Google Cloud側の状態確認

### 作業ディレクトリ

```text
~/html-site/chord-analyzer
```

### コマンド

```bash
cd ~/html-site/chord-analyzer

PROJECT_ID="ai-dojo-two26hnd-5011"
REGION="us-central1"
SERVICE_NAME="music-chord-analyzer"

gcloud config set project "$PROJECT_ID"

gcloud auth list \
  --filter=status:ACTIVE \
  --format='table(account,status)'

gcloud projects describe "$PROJECT_ID" \
  --format='table(projectId,projectNumber,lifecycleState)'

gcloud run services list \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --format='table(metadata.name,status.url,status.conditions[0].status)'
```

### 期待される結果

- アクティブアカウントに `devstar5011@gcplab.me` が表示されます。
- プロジェクトに `ai-dojo-two26hnd-5011` が表示されます。
- 既存サービスがある場合も一覧へ残ります。新規サービスは別名 `music-chord-analyzer` で作成します。

---

## 5. 推奨: デプロイスクリプトで実行

### 作業ディレクトリ

```text
~/html-site/chord-analyzer
```

### コマンド

```bash
cd ~/html-site/chord-analyzer

chmod +x deploy.sh local_run.sh

PROJECT_ID="ai-dojo-two26hnd-5011" \
REGION="us-central1" \
SERVICE_NAME="music-chord-analyzer" \
./deploy.sh
```

### 実行内容

`deploy.sh` は次を順番に行います。

1. プロジェクト設定
2. Cloud Run、Cloud Build、Artifact Registry、Vertex AI等のAPI有効化
3. `music-chord-analyzer-sa` の作成
4. 実行サービスアカウントへの `roles/aiplatform.user` 付与
5. デプロイ実行者への `roles/iam.serviceAccountUser` 付与
6. Cloud Buildサービスアカウントへの `roles/run.builder` 付与
7. Cloud Runへのソースデプロイ
8. `/api/health` の確認

### 期待される最終結果

```text
{"status":"ok","service":"music-chord-analyzer"}

DEPLOYED_URL=https://music-chord-analyzer-....run.app
```

表示された `DEPLOYED_URL` をブラウザで開きます。

---

## 6. デプロイスクリプトを使わない場合の全コマンド

### 作業ディレクトリ

```text
~/html-site/chord-analyzer
```

### コマンド

```bash
cd ~/html-site/chord-analyzer

PROJECT_ID="ai-dojo-two26hnd-5011"
REGION="us-central1"
SERVICE_NAME="music-chord-analyzer"
RUNTIME_SA_NAME="music-chord-analyzer-sa"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  --project="$PROJECT_ID"

PROJECT_NUMBER="$(
  gcloud projects describe "$PROJECT_ID" \
    --format='value(projectNumber)'
)"

BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

if ! gcloud iam service-accounts describe "$RUNTIME_SA" \
  --project="$PROJECT_ID" \
  >/dev/null 2>&1; then
  gcloud iam service-accounts create "$RUNTIME_SA_NAME" \
    --project="$PROJECT_ID" \
    --display-name="Music Chord Analyzer Cloud Run"
fi

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None

DEPLOYER_ACCOUNT="$(gcloud config get-value account 2>/dev/null)"
if [[ "$DEPLOYER_ACCOUNT" == *.gserviceaccount.com ]]; then
  DEPLOYER_MEMBER="serviceAccount:${DEPLOYER_ACCOUNT}"
else
  DEPLOYER_MEMBER="user:${DEPLOYER_ACCOUNT}"
fi

gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --project="$PROJECT_ID" \
  --member="$DEPLOYER_MEMBER" \
  --role="roles/iam.serviceAccountUser"

if gcloud iam service-accounts describe "$BUILD_SA" \
  --project="$PROJECT_ID" \
  >/dev/null 2>&1; then
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${BUILD_SA}" \
    --role="roles/run.builder" \
    --condition=None
else
  echo "WARNING: Compute Engine default service accountがまだ存在しません。"
fi

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
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=global,GEMINI_MODEL=gemini-2.5-flash,LOG_LEVEL=INFO"

SERVICE_URL="$(
  gcloud run services describe "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format='value(status.url)'
)"

printf 'SERVICE_URL=%s\n' "$SERVICE_URL"

curl --fail --silent --show-error \
  "${SERVICE_URL}/api/health"

echo
```

### 期待される結果

```text
SERVICE_URL=https://music-chord-analyzer-....run.app
{"status":"ok","service":"music-chord-analyzer"}
```

---

## 7. Cloud Shell上でローカル起動する場合

Cloud Runへのデプロイ前に、Cloud Shell Web PreviewでUIを確認する手順です。

### 作業ディレクトリ

```text
~/html-site/chord-analyzer
```

### コマンド

```bash
cd ~/html-site/chord-analyzer

PROJECT_ID="ai-dojo-two26hnd-5011" \
APP_PORT="8080" \
./local_run.sh
```

### 期待される結果

```text
INFO:     Uvicorn running on http://0.0.0.0:8080
```

Cloud Shellの「ウェブでプレビュー」→「ポート8080でプレビュー」を開きます。

停止は、起動したターミナルで `Ctrl+C` です。

---

## 8. デプロイ後の確認

### 作業ディレクトリ

```text
~/html-site/chord-analyzer
```

### コマンド

```bash
cd ~/html-site/chord-analyzer

PROJECT_ID="ai-dojo-two26hnd-5011"
REGION="us-central1"
SERVICE_NAME="music-chord-analyzer"

SERVICE_URL="$(
  gcloud run services describe "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format='value(status.url)'
)"

curl -i "${SERVICE_URL}/api/health"

gcloud run services describe "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --format='yaml(status.url,spec.template.spec.serviceAccountName,spec.template.spec.timeoutSeconds,spec.template.spec.containerConcurrency,spec.template.spec.containers[0].resources)'
```

### 期待される結果

- HTTPステータス: `200`
- service: `music-chord-analyzer`
- service account: `music-chord-analyzer-sa@ai-dojo-two26hnd-5011.iam.gserviceaccount.com`
- timeout: `900`
- concurrency: `2`
- CPU: `2`
- memory: `4Gi`

---

## 9. ログ確認

### 作業ディレクトリ

任意です。次の例では `~/html-site/chord-analyzer` を使います。

### コマンド

```bash
cd ~/html-site/chord-analyzer

PROJECT_ID="ai-dojo-two26hnd-5011"
REGION="us-central1"
SERVICE_NAME="music-chord-analyzer"

gcloud run services logs read "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --limit=100
```

リアルタイムで追う場合:

```bash
gcloud beta run services logs tail "$SERVICE_NAME" \
  --project="$PROJECT_ID" \
  --region="$REGION"
```

---

## 10. 主なエラーと対処

### `PERMISSION_DENIED: serviceusage.services.enable`

API有効化権限がありません。プロジェクト管理者に、実行アカウントへ次の権限を付与してもらいます。

```text
roles/serviceusage.serviceUsageAdmin
```

### Cloud Buildで `Permission denied` または `run.builder` 不足

### コマンド

```bash
PROJECT_ID="ai-dojo-two26hnd-5011"
PROJECT_NUMBER="$(
  gcloud projects describe "$PROJECT_ID" \
    --format='value(projectNumber)'
)"
BUILD_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${BUILD_SA}" \
  --role="roles/run.builder" \
  --condition=None
```

権限反映後、`gcloud run deploy` を再実行します。

### Cloud RunからVertex AIへ `403 PERMISSION_DENIED`

### コマンド

```bash
PROJECT_ID="ai-dojo-two26hnd-5011"
RUNTIME_SA="music-chord-analyzer-sa@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/aiplatform.user" \
  --condition=None
```

その後、新しいリビジョンをデプロイします。

### `504 Gateway Timeout`

現在の設定は900秒です。まずログで処理状況を確認します。必要な場合のみ、最大3600秒以内で延長します。

```bash
gcloud run services update music-chord-analyzer \
  --project=ai-dojo-two26hnd-5011 \
  --region=us-central1 \
  --timeout=1200s
```

### YouTubeだけ失敗する

次を確認します。

- 動画が公開状態か
- 年齢制限、地域制限、ログイン必須ではないか
- URLが通常のYouTube動画URLか
- Vertex AIのモデル利用権限とクォータがあるか

### Spotify／Apple Music URLが422になる

仕様どおりです。公式URLから解析用音声を取得せず、YouTube URLまたは利用権利のある音声ファイルを要求します。

---

## 11. 新規サービスだけを削除する場合

この操作は `music-chord-analyzer` だけを削除し、既存サービス `html-site` には触れません。

### コマンド

```bash
gcloud run services delete music-chord-analyzer \
  --project=ai-dojo-two26hnd-5011 \
  --region=us-central1
```

ローカルソースも削除する場合は、削除前にバックアップを作成します。

```bash
cd ~/html-site

cp -a chord-analyzer \
  "$HOME/chord-analyzer-backup-$(date +%Y%m%d-%H%M%S)"

rm -rf ~/html-site/chord-analyzer
```
