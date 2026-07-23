# Neural Chord Engine Phase 1

このディレクトリは、本番アプリへ組み込む前に、現行DSPと `Omnizart + Beat This!` を同じ音源で比較するための隔離実験です。

本番の `main.py`、`music_agent/agents.py`、`static/index.html` はこの段階では変更しません。精度改善を数値または実聴で確認できた場合だけ、次の段階で非公開推論サービスへ統合します。

## 使用技術

- Omnizart 0.6.3: コード名とコード境界
- Beat This! 1.1.0: 拍とダウンビート
- mir_eval 0.8.2: 正解ラベルがある場合の評価
- 現行 `music_agent.dsp`: 比較対象

## 1. ブランチを取得

```bash
cd ~/chord-analyzer

git fetch origin

git switch feature/neural-chord-engine

git pull --ff-only origin feature/neural-chord-engine
```

## 2. 実験環境を構築

```bash
cd ~/chord-analyzer

chmod +x \
  experiments/neural_chord_engine/setup_cloud_shell.sh \
  experiments/neural_chord_engine/check.sh

experiments/neural_chord_engine/setup_cloud_shell.sh
```

Omnizartの公式手順に従い、必要なシステムパッケージ、Python依存関係、学習済みチェックポイントを導入します。Beat This! の `small0` モデルも事前ロードします。

Omnizartチェックポイントを既に配置済みの場合だけ、次のように再ダウンロードを省略できます。

```bash
SKIP_OMNIZART_CHECKPOINTS=1 \
experiments/neural_chord_engine/setup_cloud_shell.sh
```

## 3. 構文検査と単体テスト

```bash
cd ~/chord-analyzer

source "$HOME/.venvs/chord-neural-experiment/bin/activate"

experiments/neural_chord_engine/check.sh
```

期待結果:

```text
NEURAL_CHORD_EXPERIMENT_CHECK=OK
```

## 4. 音声をWAVへ変換

Omnizartのコード転記入力はWAVを使用します。

```bash
cd ~/chord-analyzer

mkdir -p test-audio

ffmpeg \
  -hide_banner \
  -loglevel error \
  -y \
  -i "/path/to/source.mp3" \
  -vn \
  -ac 1 \
  -ar 44100 \
  -c:a pcm_s16le \
  "test-audio/target.wav"
```

## 5. A/B比較を実行

```bash
cd ~/chord-analyzer

source "$HOME/.venvs/chord-neural-experiment/bin/activate"

python experiments/neural_chord_engine/evaluate.py \
  --audio "test-audio/target.wav" \
  --output "ab-test-output/target" \
  --beat-model small0
```

生成物:

```text
ab-test-output/target/
├── legacy.json
├── omnizart-raw.json
├── omnizart-processed.json
├── timing.json
├── report.json
└── omnizart/
    └── Omnizartが生成したCSVとMIDI
```

## 6. 正解ラベルがある場合

Harte形式または一般的なコード表記で、1行を `開始秒 終了秒 コード` とします。

```text
0.000 1.500 N
1.500 3.000 C:min
3.000 4.500 G#:maj
4.500 6.000 D#:maj
```

評価コマンド:

```bash
python experiments/neural_chord_engine/evaluate.py \
  --audio "test-audio/target.wav" \
  --reference "test-audio/target.lab" \
  --output "ab-test-output/target-with-reference" \
  --beat-model small0
```

`report.json` に `Root`、`MajMin`、`Sevenths`、`MIREX`、`OverSeg`、`UnderSeg`、`Seg` 等が出力されます。

## 採用判定

本番統合へ進む最低条件:

- 現行方式よりRoot一致率が15ポイント以上改善
- Root加重一致率80%以上
- MajMin加重一致率75%以上
- Seg 0.70以上
- 曲終了後のコードイベント0件
- 無音区間の誤コード0件
- 一拍だけの不要なコード振動を現行比70%以上削減
- 同じ音源を3回解析したJSONが一致

正解ラベルが用意できない場合は、少なくとも「フォニイ」の既知コード譜と原音を使い、`legacy.json` と `omnizart-processed.json` を同じ時間位置で比較します。

## 現時点の制約

- この段階では拡張コード判定器をまだ追加していません。
- Omnizartのメジャー・マイナー・N・Xを正本候補として評価します。
- Beat This! の拍スナップは、元境界から0.15秒以内の場合だけ適用します。
- 0.35秒未満の孤立コードは決定的規則で統合します。
- 無音区間はモデル結果より優先し、Nへ置換します。
- 精度改善を確認するまでCloud Run本番サービスには組み込みません。
