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

## Accuracy V2

`ANALYSIS_PIPELINE=v2` では、YouTube音声を外部ダウンロードせず、Vertex AIのYouTube URL入力を使用します。

- 全体構造とリズムを独立解析
- 全体構造が不十分な場合は重複する時間窓で再解析
- セクションごとにルート／基本品質、ベース／拡張音、変更拍／反復を独立解析
- 候補列を時系列最適化してコード譜へ統合
- 信頼度は代替候補を独立票として数えず、各専門分析の一次回答から算出
- `C`、`Cmaj7`、`C/E` のようにルートと基本品質が一致する差は重大な未解決範囲へ誤分類しない
- ルートまたはmajor/minor等の基本品質が不一致の場合だけ局所再解析
- セクション数、コード密度、既知コードの時間カバー率、未解決範囲率を品質ゲートで検査
- 品質ゲート未達の解析を正常結果として返さない

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
| `GEMINI_MODEL` | `gemini-3.5-flash` | 使用モデル |
| `GEMINI_RESOLVER_MODEL` | `GEMINI_MODEL`と同じ | 不一致解決モデル |
| `ANALYSIS_PIPELINE` | `v2` | `v2` または明示的ロールバック用の `legacy` |
| `MODEL_MAX_PARALLEL_CALLS` | `4` | 同時モデル呼び出し上限 |
| `MAX_RESOLVER_CALLS` | `4` | 局所不一致解決の最大回数 |
| `ENABLE_REFERENCE_RESEARCH` | `1` | Google Search補助調査 |
| `YOUTUBE_API_KEY` | 未設定 | YouTube Data APIによる正式メタデータ取得 |
| `LOG_LEVEL` | `INFO` | ログレベル |

## 現在の上限

- 入力ファイルまたは直接音声URL: 30 MiB以下
- 音源長: 12分以下
- Geminiインライン音声: 14 MiB以下へFFmpegで変換
- サーバー側へ音源・分析結果・質問履歴を永続保存しない

## 注意

コード、キー、BPM、セクション境界は自動推定です。複雑なテンション、転調、転回形、ミックスの密度が高い部分では誤認する場合があります。
