# QPP4SIP 実験システム

QPP特徴量を使用したSIP（System Initiative Prediction）実験のための統合システムです。

## 🚀 クイックスタート

### データ前処理

```bash
# データセットの準備と検証
uv run python preprocess.py --prepare_dataset --validate_dataset --input_pkl data.pkl

# 全ての前処理を実行
uv run uv run python preprocess.py --all --input_pkl data.pkl --dataset_type train
```

### 学習・推論・評価

```bash
# 学習
uv run python run.py --mode train --model qpp4sip --qpp4sip_pattern feature_fusion --qpp_feature_indices 0 1 2 --input_path data.pkl

# 推論
uv run python run.py --mode inference --model qpp4sip --qpp4sip_pattern feature_fusion --qpp_feature_indices 0 1 2 --input_path data.pkl --saved_model_path ./checkpoints/qpp4sip_feature_fusion_012

# 評価
uv run python run.py --mode evaluation --model qpp4sip --qpp4sip_pattern feature_fusion --qpp_feature_indices 0 1 2 --input_path data.pkl --output_path ./output/qpp4sip_feature_fusion_012/result

# MUSICモデル（ベースライン）
uv run python run.py --mode train --model music --input_path data.pkl
```

### バッチ実験の実行

```bash
# 全特徴量セット×全パターンの実験を一括実行
./run_all_experiments.sh
```

## 📁 ディレクトリ構造

```
SIP/
├── preprocess.py              # データ前処理エントリポイント
├── run.py                     # 学習・推論・評価エントリポイント
├── run_all_experiments.sh     # バッチ実験スクリプト
├── config/                    # 設定ファイル
│   ├── qpp_config.py         # QPP特徴量実験の設定クラス
│   └── experiment_configs.json # 実験設定のテンプレート
├── model/                     # モデル定義
│   ├── qpp4sip_model.py      # QPP4SIPモデル
│   ├── music_model.py        # MUSICモデル（ベースライン）
│   └── ...
├── utils/                     # ユーティリティ
│   └── experiment_utils.py   # 実験管理用ユーティリティ
├── dataset/                   # データセット
├── evaluation/                # 評価モジュール
└── experiments/               # 実験結果（自動生成）
```

## 🔧 実験の流れ

1. **環境セットアップ**: 実験ディレクトリの作成、設定の保存
2. **学習の実行**: モデルの学習
3. **推論の実行**: 学習済みモデルでの推論
4. **評価の実行**: 推論結果の評価
5. **サマリー出力**: 実験結果のサマリー生成

## ⚙️ 利用可能な設定

### 基本引数

| 引数 | 説明 | デフォルト | 必須 |
|------|------|------------|------|
| `--mode` | 実行モード | `train` | はい |
| `--model` | モデルタイプ | `music` | はい |
| `--input_path` | 入力PKLファイルのパス | - | はい |
| `--output_path` | 出力パス | `./output` | いいえ |
| `--saved_model_path` | チェックポイントのパス | `./checkpoints` | いいえ |

### モデル

- `music`: MUSICモデル（ベースライン）
- `qpp4sip`: QPP4SIPモデル

### QPP4SIPパターン

- `feature_fusion`: 特徴融合（BERT表現にQPP特徴量を結合）
- `auxiliary_head`: 補助ヘッド（QPP予測タスクを追加）
- `policy_gating`: ポリシー連動（QPPスコアに基づいてゲート値を計算）

### QPP特徴量の指定

#### 特徴量インデックス（推奨）

```bash
# 特定の特徴量のみ使用
--qpp_feature_indices 0 1 2  # ndcg@1, ndcg@3, ndcg@5
--qpp_feature_indices 3 4 5  # precision@1, precision@5, precision@10
--qpp_feature_indices 6 7 8  # recall@1, recall@5, recall@10
--qpp_feature_indices 0 1 2 3 4 5 6 7 8  # 全特徴量
```

#### 特徴量マッピング

| インデックス | 特徴量名 | 説明 |
|-------------|----------|------|
| 0 | ndcg@1 | NDCG@1 |
| 1 | ndcg@3 | NDCG@3 |
| 2 | ndcg@5 | NDCG@5 |
| 3 | precision@1 | Precision@1 |
| 4 | precision@5 | Precision@5 |
| 5 | precision@10 | Precision@10 |
| 6 | recall@1 | Recall@1 |
| 7 | recall@5 | Recall@5 |
| 8 | recall@10 | Recall@10 |

### 学習パラメータ

| 引数 | 説明 | デフォルト |
|------|------|------------|
| `--epoch_num` | エポック数 | `20` |
| `--learning_rate` | 学習率 | `2e-5` |
| `--lr_distance_crf` | CRF学習率 | `1e-3` |
| `--max_utterance_len` | 最大発話長 | `128` |
| `--max_context_len` | 最大コンテキスト長 | `384` |

### 出力パスの命名規則

QPP4SIPモデルの場合、出力パスは以下の形式で自動生成されます：

```
./output/qpp4sip_{pattern}_{feature_id}/
```

例：
- `qpp4sip_feature_fusion_012` (ndcg系のみ)
- `qpp4sip_feature_fusion_345` (precision系のみ)
- `qpp4sip_feature_fusion_012345678` (全特徴量)

## 📊 実験結果の管理

各実験は以下の構造で保存されます：

```
experiments/
└── model_pattern_features_YYYYMMDD_HHMMSS/
    ├── models/              # 学習済みモデル
    ├── logs/                # ログファイル
    ├── results/             # 推論結果
    ├── evaluation/          # 評価結果
    ├── configs/             # 設定ファイル
    ├── experiment_info.txt  # 実験情報
    └── experiment_summary.txt # 実験サマリー
```

## 🛠️ 高度な使用方法

### 実験例

#### 1. NDCG系特徴量のみで実験

```bash
# 学習
python run.py --mode train --model qpp4sip --qpp4sip_pattern feature_fusion --qpp_feature_indices 0 1 2 --input_path dataset/pkl/bm25_train.pkl

# 推論
python run.py --mode inference --model qpp4sip --qpp4sip_pattern feature_fusion --qpp_feature_indices 0 1 2 --input_path dataset/pkl/bm25_dev.pkl --saved_model_path ./checkpoints/qpp4sip_feature_fusion_012

# 評価
python run.py --mode evaluation --model qpp4sip --qpp4sip_pattern feature_fusion --qpp_feature_indices 0 1 2 --input_path dataset/pkl/bm25_dev.pkl --output_path ./output/qpp4sip_feature_fusion_012/result
```

#### 2. 全特徴量で実験

```bash
# 学習（特徴量インデックスを指定しない場合は全特徴量を使用）
python run.py --mode train --model qpp4sip --qpp4sip_pattern feature_fusion --input_path dataset/pkl/bm25_train.pkl

# 推論
python run.py --mode inference --model qpp4sip --qpp4sip_pattern feature_fusion --input_path dataset/pkl/bm25_dev.pkl --saved_model_path ./checkpoints/qpp4sip_feature_fusion_012345678

# 評価
python run.py --mode evaluation --model qpp4sip --qpp4sip_pattern feature_fusion --input_path dataset/pkl/bm25_dev.pkl --output_path ./output/qpp4sip_feature_fusion_012345678/result
```

#### 3. MUSICモデル（ベースライン）

```bash
# 学習
python run.py --mode train --model music --input_path dataset/pkl/bm25_train.pkl

# 推論
python run.py --mode inference --model music --input_path dataset/pkl/bm25_dev.pkl --saved_model_path ./checkpoints/music

# 評価
python run.py --mode evaluation --model music --input_path dataset/pkl/bm25_dev.pkl --output_path ./output/music/result
```

### バッチサイズについて

**注意**: このシステムはCRFの制約により、バッチサイズは1に固定されています。`--batch_size`引数は使用できません。

## 🔍 トラブルシューティング

### よくある問題

1. **インポートエラー**: パスが正しく設定されているか確認してください
2. **QPP特徴量の次元エラー**: 実際のPKLデータに存在する特徴量のみを使用してください
3. **メモリ不足**: 使用する特徴量を減らしてください（バッチサイズは1に固定）
4. **出力パスエラー**: 推論・評価時は学習時に生成されたパスを使用してください

### ログの確認

実験の詳細なログは `output/model_pattern_features/logs/` ディレクトリに保存されます。

### 特徴量の確認

使用可能なQPP特徴量を確認するには：

```python
from config.qpp_config import QPPExperimentConfig
print(QPPExperimentConfig.QPP_FEATURE_NAMES)
```

## 📈 実験結果の管理

### 出力ディレクトリ構造

```
output/
├── music/                          # MUSICモデル
├── qpp4sip_feature_fusion_012/     # NDCG系特徴量のみ
├── qpp4sip_feature_fusion_345/     # Precision系特徴量のみ
├── qpp4sip_feature_fusion_678/     # Recall系特徴量のみ
└── qpp4sip_feature_fusion_012345678/ # 全特徴量
```

### 実験結果の比較

異なる特徴量セットの結果を比較するには、各ディレクトリの評価結果を確認してください。

## 🧪 開発者向け情報

### 新しいQPP特徴量の追加

1. `config/qpp_config.py`の`QPP_FEATURE_NAMES`に新しい特徴量を追加
2. `model/dataset.py`の`qpp_feature_names`リストを更新
3. `config/qpp_config.py`の`validate_qpp_features`関数を更新

### 新しいQPP4SIPパターンの追加

`model/qpp4sip_model.py`に新しいパターンを実装し、`run.py`の引数処理を更新してください。