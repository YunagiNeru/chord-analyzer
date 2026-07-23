# Music Chord Analyzer

`ai-dojo-two26hnd-5011` の Cloud Run へ独立サービスとしてデプロイする、楽曲構成・コード進行分析アプリです。

## 主な機能

- 単一の `static/index.html` によるレスポンシブWeb UI
- YouTube URLのGemini直接解析
- MP3 / WAV / FLAC / AAC / M4A / AIFF / OGG / OPUS / WebMアップロード
- 公開された直接音声URLの取得と解析（SSRF対策あり）
- `librosa` DSPとGeminiを統合したBPM・キー・コード・セクション推定
- 構成分析、和声分析、理論監査を分担するサブエージェント構成
- 範囲選択タイムライン、選択範囲再生、セクション境界スナップ
- 小節・拍・コード・度数を示すコード譜
- ライト／ダーク／システムテーマ
- 分析結果だけで答える質問エージェントと、必要時だけGoogle Searchを使う調査エージェント
- プレーンテキスト、Markdown、選択範囲JSON、解析全体JSONのコピー

## 入力対応

| 入力 | 動作 |
|---|---|
| YouTube URL | GeminiへURLを直接渡して解析 |
| 音声ファイル | FFmpeg変換 → DSP解析 → Gemini解析 |
| 直接音声URL | 安全性検査後に一時取得し、アップロードと同じ方式で解析 |
| Spotify URL | 音声を取得できないため、YouTube URLまたは音声ファイルを案内 |
| Apple Music URL | 音声を取得できないため、YouTube URLまたは音声ファイルを案内 |

## エージェント構成

```text
MusicCoordinatorAgent
├── SourceInspectorAgent
├── StructureAnalysisAgent
├── HarmonyAnalysisAgent
├── TheoryAuditAgent
├── QuestionRouterAgent
├── AnalysisQuestionAgent
└── MusicResearchAgent
```

## API

- `GET /` — Web UI
- `GET /api/health` — ヘルスチェック
- `POST /api/analyze` — `multipart/form-data` で `source_url` または `audio_file`
- `POST /api/question` — 質問と解析JSON

## 環境変数

| 変数 | 既定値 | 用途 |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | 必須 | Google CloudプロジェクトID |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AIロケーション |
| `GEMINI_MODEL` | `gemini-2.5-flash` | 使用モデル |
| `LOG_LEVEL` | `INFO` | ログレベル |

## 現在の上限

- 入力ファイルまたは直接音声URL: 30 MiB以下
- 音源長: 12分以下
- Geminiインライン音声: 14 MiB以下へFFmpegで変換
- サーバー側へ音源・分析結果・質問履歴を永続保存しない

## 注意

コード、キー、BPM、セクション境界は自動推定です。複雑なテンション、転調、転回形、ミックスの密度が高い部分では誤認する場合があります。
