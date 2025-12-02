# Neural QPP: 学習が必要なQPP手法の実装

BERT-QPPなど、学習が必要なQPP手法の実装です。

## ディレクトリ構造

```
neural_qpp/
├── __init__.py
├── data_loader.py          # データローダーとQPPスコア計算
├── main.py                 # 学習と検証のメインスクリプト
├── methods/
│   ├── __init__.py
│   └── bertqpp.py         # BERT-QPPの学習と評価実装
└── outputs/               # モデルと結果の保存先
    └── {dataset}/
        └── {model_type}_{metric}/
            ├── best_model.pth
            └── training_history.json
```

## 機能

### 1. データローダー (`data_loader.py`)
- DPRの結果ファイルを読み込み
- QPPスコア（MAP@20など）を計算
- 学習用データを準備

### 2. BERT-QPP (`methods/bertqpp.py`)
- **Bi-encoder形式**: クエリとドキュメントを別々にエンコード
- **Cross-encoder形式**: クエリとドキュメントを結合してエンコード
- 学習と検証の実装
- モデルの保存と読み込み

## 使用方法

### 学習と検証の実行

```bash
# Bi-encoder形式で学習（デフォルト）
python main.py --dataset INSCIT --model_type bi --metric map@20 --num_epochs 10

# Cross-encoder形式で学習
python main.py --dataset INSCIT --model_type cross --metric map@20 --num_epochs 10

# パラメータをカスタマイズ
python main.py \
    --dataset INSCIT \
    --model_type bi \
    --metric map@20 \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --max_length 512

# wandbを使用して学習を記録
python main.py \
    --dataset INSCIT \
    --model_type bi \
    --metric map@20 \
    --use_wandb \
    --wandb_project neural-qpp \
    --wandb_mode online
```

### 引数

- `--dataset`: データセット名（`INSCIT` または `AmbigNQ`、必須）
- `--model_type`: モデルタイプ（`bi` または `cross`、デフォルト: `bi`）
- `--metric`: QPPスコアの計算メトリクス（`map@20` または `map`、デフォルト: `map@20`）
- `--num_epochs`: エポック数（デフォルト: 10）
- `--batch_size`: バッチサイズ（デフォルト: 16）
- `--learning_rate`: 学習率（デフォルト: 2e-5）
- `--max_length`: 最大シーケンス長（デフォルト: 512）
- `--device`: 使用するデバイス（`cuda` または `cpu`、デフォルト: 自動選択）
- `--model_name`: 事前学習済みBERTモデル名（デフォルト: `bert-base-uncased`）
- `--use_wandb`: wandbを使用して学習を記録する（デフォルト: False）
- `--wandb_project`: wandbプロジェクト名（デフォルト: `neural-qpp`）
- `--wandb_mode`: wandbのモード（`online`, `offline`, `disabled`、デフォルト: `online`）。環境変数`WANDB_MODE`で上書き可能

## 出力

学習が完了すると、以下のファイルが保存されます：

- `outputs/{dataset}/{model_type}_{metric}/best_model.pth`: 最良モデル
- `outputs/{dataset}/{model_type}_{metric}/training_history.json`: 学習履歴

## データ分割について

- **trainデータ**: 学習に使用
- **devデータ**: 検証に使用（各エポック後）
- **testデータ**: 現在は使用していません（devデータで検証と最終評価を実施）

固定分割方式で、train/devの分割を使用しています。

## wandbの使用

wandbを使用して学習過程を記録できます：

```bash
# wandbを使用
python main.py --dataset INSCIT --model_type bi --use_wandb

# wandbのモードを指定（offlineモードで実行）
python main.py --dataset INSCIT --model_type bi --use_wandb --wandb_mode offline

# 環境変数でwandbモードを指定
export WANDB_MODE=offline
python main.py --dataset INSCIT --model_type bi --use_wandb
```

wandbがインストールされていない場合、`--use_wandb`を指定しても警告が表示されるだけで、学習は正常に実行されます。

## 注意事項

- 学習にはGPUを推奨します
- モデルの保存には十分なディスク容量が必要です
- 学習時間はデータセットサイズとエポック数に依存します
- wandbを使用する場合は、事前に`pip install wandb`を実行し、`wandb login`でログインしてください

