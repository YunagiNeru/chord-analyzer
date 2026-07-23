# Music Chord Analyzer

YouTube URL、音声ファイル、直接音声URLから楽曲構造・BPM・キー・コード進行を解析するCloud Runアプリです。

## Accuracy V2

既定の `ANALYSIS_PIPELINE=v2` では、単一の全曲コード生成を廃止し、次の処理を行います。

1. YouTube公式メタデータまたはoEmbedによるソース確定
2. 全体構造分析とリズム／ダウンビート分析を独立実行
3. BPM半分・標準・2倍とダウンビート位置の決定的なグリッド選択
4. セクションごとのクリップ解析
5. ルート／品質、ベース／テンション、変更拍／反復の3専門分析
6. 候補合議と時系列最適化
7. 低一致度範囲だけの候補制約付き再解析
8. Pythonによる範囲、小節、拍、重複、終了時刻の完全性検査
9. 確定コードを変更しない説明生成

YouTube動画はGeminiへ公式URLを直接渡します。yt-dlp、Cookie、YouTube音声のサーバーダウンロードは使用しません。

## 入力

| 入力 | 処理 |
|---|---|
| YouTube URL | GeminiのYouTube URL入力とVideoMetadataによる区間解析 |
| 音声ファイル | FFmpeg変換、改善DSP、Gemini区間解析、合議 |
| 直接音声URL | SSRF対策後に一時取得し、音声ファイルと同じ処理 |
| Spotify / Apple Music | YouTube URLまたは権利を持つ音声ファイルを案内 |

## API

- `GET /`
- `GET /api/health`
- `POST /api/analyze` — `source_url` または `audio_file`
- `POST /api/question` — 質問と解析JSON

既存の `AnalysisResult` フィールドは維持し、V2では以下を追加します。

- `tempoSegments`
- `downbeatOffsetSeconds`
- `referenceSources`
- `uncertainRanges`
- `diagnostics`
- 各コード／セクションの `agreement`

## 環境変数

| 変数 | 既定値 | 用途 |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | 必須 | Google CloudプロジェクトID |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AIロケーション |
| `GEMINI_MODEL` | `gemini-3.5-flash` | 基本分析モデル |
| `GEMINI_RESOLVER_MODEL` | 基本モデルと同じ | 不一致解決モデル |
| `ANALYSIS_PIPELINE` | `v2` | `v2` または明示的ロールバック用 `legacy` |
| `MODEL_MAX_PARALLEL_CALLS` | `4` | 1リクエスト内の最大並列モデル呼び出し |
| `MAX_RESOLVER_CALLS` | `4` | 低一致度区間の最大再解析数 |
| `ENABLE_REFERENCE_RESEARCH` | `1` | Google Searchによる補助情報調査 |
| `YOUTUBE_API_KEY` | 任意 | YouTube Data APIによる正式タイトル・再生時間取得 |

## ローカル起動

```bash
PROJECT_ID=ai-dojo-july bash local_run.sh
```

## テスト

```bash
python -m compileall -q main.py music_agent tests
python -m unittest discover -s tests -v
python -m pip check
```

## デプロイ

```bash
PROJECT_ID=ai-dojo-july bash deploy.sh
```

ステージングへ出す場合はサービス名だけ変更します。

```bash
PROJECT_ID=ai-dojo-july \
SERVICE_NAME=music-chord-analyzer-v2-staging \
bash deploy.sh
```

## 制限

- 入力ファイル／直接音声URL: 30MiB以下
- 音源長: 12分以下
- Cloud Runタイムアウト: 900秒
- 自動採譜のため、密集ボイシング、ライブアレンジ、意図的な曖昧和音では低信頼範囲が残る場合があります。その場合は `uncertainRanges` と `warnings` に明示します。
