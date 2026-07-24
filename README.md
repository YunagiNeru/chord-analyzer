# Music Chord Analyzer

`ai-dojo-july` の Cloud Runへ独立サービスとしてデプロイする、楽曲構成・コード進行分析アプリです。

## 主な機能

- 単一の `static/index.html` によるレスポンシブWeb UI
- YouTube URLのVertex AI直接解析
- MP3 / WAV / FLAC / AAC / M4A / AIFF / OGG / OPUS / WebMアップロード
- 公開された直接音声URLの取得と解析（SSRF対策あり）
- `librosa` DSPとGeminiを統合したBPM・キー・コード・セクション推定
- 範囲選択タイムライン、選択範囲再生、セクション境界スナップ
- 小節・拍・コード・度数、信頼度、代替候補を示すコード譜
- 分析結果だけで答える質問エージェントと、必要時だけGoogle Searchを使う調査エージェント
- プレーンテキスト、Markdown、選択範囲JSON、解析全体JSONのコピー

## Accuracy V2

`ANALYSIS_PIPELINE=v2` では、YouTube音声を外部ダウンロードせず、Vertex AIのYouTube URL入力を使用します。

- 全体構造とリズムを独立解析
- 全体構造が不十分な場合は重複する時間窓で再解析
- BPM確定後にセクション境界を拍・小節グリッドへ整列
- 長すぎる主要セクションは意味的再解析し、失敗時は小節境界で決定的に分割
- Specialistへ渡す分析区間を最大16小節に制限
- セクションごとにルート／基本品質、ベース／拡張音、変更拍／反復を独立解析
- Specialist出力はコード、時刻、信頼度、代替候補だけの小型JSON
- 候補列を時系列最適化してコード譜へ統合
- 信頼度は代替候補を独立票として数えず、各専門分析の一次回答から算出
- 反復パターンで解決できない範囲だけResolverで局所再解析
- Resolverは複数状態を個別のallow-list付きで判定し、JSON途中終了時は小さいバッチへ分割して回復
- Vertex AIの429、503、タイムアウト、構造化JSON途中終了には上限付き指数バックオフを適用
- 冒頭、末尾、セクション内の未判定時間を消さず、`X`または低信頼候補として保持
- セクション数、コード密度、既知コードの時間カバー率、未解決率、境界整列、長すぎる区間を品質ゲートで検査
- グローバルキー、セクションキー、異名同音表記、Roman numeralを最終結果上で再整合

## 本番リリースプロファイル

`deploy.sh` は次の安全側設定でCloud Runへデプロイします。

- Cloud Run concurrency: `1`
- Vertex AI同時呼び出し: `2`
- 通常Resolverバッチ: 最大`12`呼び出し
- 失敗したResolverだけを小分け再試行: 最大`16`呼び出し
- リクエストタイムアウト: `900s`
- 最大インスタンス数: `2`
- Google Search補助調査: 既定で無効
- 構造・時間軸・既知コードカバー率が破綻した結果は拒否
- Resolverの一時障害だけが残る場合は、未確定率と候補を明示した下書き結果を返却

本番モードでも、コード譜を確定譜として断定しません。UIは信頼度、候補、警告を表示し、利用者が音源と照合できる設計です。

## 入力対応

| 入力 | 動作 |
|---|---|
| YouTube URL | GeminiへURLを直接渡して解析 |
| 音声ファイル | FFmpeg変換 → DSP解析 → Gemini解析 |
| 直接音声URL | 安全性検査後に一時取得し、アップロードと同じ方式で解析 |
| Spotify URL | 音声を取得できないため、YouTube URLまたは音声ファイルを案内 |
| Apple Music URL | 音声を取得できないため、YouTube URLまたは音声ファイルを案内 |

## API

- `GET /` — Web UI
- `GET /api/health` — ヘルスチェック
- `POST /api/analyze` — `multipart/form-data`で`source_url`または`audio_file`
- `POST /api/question` — 質問と解析JSON

## 環境変数

| 変数 | 開発時の既定値 | 用途 |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | 必須 | Google CloudプロジェクトID |
| `GOOGLE_CLOUD_LOCATION` | `global` | Vertex AIロケーション |
| `GEMINI_MODEL` | `gemini-3.5-flash` | 使用モデル |
| `GEMINI_RESOLVER_MODEL` | `GEMINI_MODEL`と同じ | 不一致解決モデル |
| `ANALYSIS_PIPELINE` | `v2` | `v2`または明示的ロールバック用の`legacy` |
| `MODEL_MAX_PARALLEL_CALLS` | `4` | 同時モデル呼び出し上限。デプロイ時は`2` |
| `MAX_RESOLVER_CALLS` | `12` | 通常Resolverバッチの最大回数 |
| `RESOLVER_BATCH_SIZE` | `8` | 1バッチで個別判定する最大状態数 |
| `MAX_RESOLVER_RECOVERY_CALLS` | `12` | 失敗バッチの小分け回復上限。デプロイ時は`16` |
| `ENABLE_REFERENCE_RESEARCH` | `1` | Google Search補助調査。デプロイ時は`0` |
| `STRICT_QUALITY_GATE` | `1` | `0`では安全な部分結果を警告付きで返す |
| `YOUTUBE_API_KEY` | 未設定 | YouTube Data APIによる正式メタデータ取得 |
| `LOG_LEVEL` | `INFO` | ログレベル |

## テスト

```bash
python -m compileall -q main.py music_agent scripts tests
python -m unittest discover -s tests -v
bash -n deploy.sh
```

`deploy.sh` はデプロイ前に同じコンパイルと全テストを実行します。再デプロイ時にテストを省略する場合だけ、明示的に `RUN_LOCAL_TESTS=0` を指定します。

## Cloud Runデプロイ

```bash
cd ~/chord-analyzer
git pull --ff-only origin feature/accuracy-v2
source "$HOME/.venvs/chord-analyzer/bin/activate"
PROJECT_ID=ai-dojo-july REGION=us-central1 ./deploy.sh
```

成功時は最後に次を表示します。

```text
DEPLOYED_URL=https://...
RELEASE_SHA=<deployed commit sha>
```

## 現在の上限

- 入力ファイルまたは直接音声URL: 30 MiB以下
- 音源長: 12分以下
- Geminiインライン音声: 14 MiB以下へFFmpegで変換
- サーバー側へ音源・分析結果・質問履歴を永続保存しない

## 注意

コード、キー、BPM、セクション境界は自動推定です。複雑なテンション、転調、転回形、密度の高いミックスでは誤認する場合があります。低信頼区間はUI上の候補、信頼度、警告と音源再生を使って確認してください。
