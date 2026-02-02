# MoE (Mixture of Experts) 動的ゲーティング統合

LASSOディレクトリではロジスティック回帰による**固定重み**で複数予測子を統合しているのに対し、ここでは**動的な重み**でソフトゲーティングするMoE的な統合を実装しています。

## 条件の揃え方（LASSOとの一致）

- **データ**: LASSOと同じ `load_json_data`, `extract_labels`, `load_qpp_scores`, `load_base_scores`, `merge_features` を使用（LASSOの `module` を参照）
- **正規化**: LASSOと同じ `normalize_features`（z-score または Min-Max、訓練の統計量でテストも変換）
- **特徴量タイプ**: `--feature-types post pre nsp` 等、LASSOと同じオプション
- **ラベル調整**: `--balance-label-distribution` でLASSOと同様に調整可能

## ゲーティング方式（複数パターン）

| 方式 | 説明 |
|------|------|
| **linear** | 入力特徴量の線形結合 + softmax。`g = softmax(Wx + b)`。入力に応じてどのExpertを重視するかを線形で決定。 |
| **mlp** | 2層MLP + softmax。`g = softmax(W2 @ ReLU(W1 x + b1) + b2)`。非線形なゲーティング。`--gate-hidden-size` で隠れ層サイズを指定。 |
| **temperature** | 線形 + 温度付きsoftmax。`g = softmax((Wx + b) / T)`。Tを小さくすると重みがよりスパイキー（1つに集中）に。`--gate-temperature` でTを指定。 |
| **learned_fixed** | 入力に依存しない学習可能な固定重み。`g = softmax(learned_logits)`。LASSOの固定重みに近いが、softmaxで正規化。比較用。 |

## 使い方

### 一括実行して比較（推奨）

全ゲーティング方式を順に実行し、最後に比較表と CSV を出力します。

```bash
cd LogReg/MoE
./script/run_moe.sh
# または
./script/run_all_moe.sh
# 環境変数で指定
DATASET=INSCIT RETRIEVAL=bm25 ./script/run_all_moe.sh
```

結果は `outputs/<dataset>/run_all_<timestamp>/` に保存され、`comparison.csv` と各方式の `results.txt` が作成されます。

### 個別実行

```bash
# 線形ゲートで train/dev 分離評価（CVなし）
python main.py --dataset AmbigNQ --gating-type linear --no-cv

# MLPゲート（隠れ層32）
python main.py --dataset AmbigNQ --gating-type mlp --gate-hidden-size 32 --no-cv

# 温度付きsoftmax（T=0.5）
python main.py --dataset AmbigNQ --gating-type temperature --gate-temperature 0.5 --no-cv

# 学習固定重み（比較用）
python main.py --dataset AmbigNQ --gating-type learned_fixed --no-cv

# クロスバリデーション（デフォルト5-fold）
python main.py --dataset AmbigNQ --gating-type linear

# LASSOと同じ正規化・特徴量オプション
python main.py --dataset AmbigNQ --gating-type linear --no-cv \
  --use-minmax-normalization \
  --feature-types post pre nsp
```

結果は `LogReg/MoE/outputs/<dataset>/<特徴量名>_<gating>/results.txt` に保存されます。

## ディレクトリ構成

- `module/gating.py` - ゲートの実装（linear, mlp, temperature, learned_fixed）
- `module/experts.py` - 各予測子を1変数ロジスティック回帰Expertとして学習
- `module/moe.py` - MoE統合モデル（Experts + Gate）
- `main.py` - データ読み込み（LASSO module 利用）・正規化・学習・評価
