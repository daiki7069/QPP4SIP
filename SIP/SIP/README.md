# QPP4SIP: Query Performance Prediction for System Initiative Prediction

このプロジェクトは、QPP（Query Performance Prediction）特徴量を活用したSIP（System Initiative Prediction）タスクの実装です。

## 概要

QPP4SIPは、会話システムにおけるシステムのイニシアチブ発動を予測するタスクです。従来のSIPタスクに加えて、QPP特徴量（ndcg@3、precision@1など）を活用することで、より精度の高い予測を実現します。

## 実装パターン

3つの異なるアプローチを実装しています：

1. **Feature Fusion**: BERTの[CLS]表現にQPP特徴量を結合してMLPで埋め込み
2. **Auxiliary Head**: QPP回帰ヘッド（ndcg@3予測）を追加したマルチタスク学習
3. **Policy Gating**: QPPスコアに基づいてイニシアチブ発動の閾値を調整するポリシー連動

## ファイル構成

```
SIP/
├── model/
│   ├── run.py                 # メイン実行スクリプト
│   ├── music_model.py         # 従来のmusicモデル
│   ├── qpp4sip_model.py       # QPP4SIPモデル（3パターン対応）
│   ├── dataset.py             # データセット処理
│   ├── trainer.py             # 学習・推論クラス
│   ├── utils.py               # ユーティリティ関数
│   └── load_pkl.py            # データ読み込み
├── scripts/                   # シェルスクリプト
│   └── *.sh                   # 実行スクリプト
├── evaluation/                # 評価関連
│   └── evaluation.py          # 評価スクリプト
├── dataset/                   # データセットファイル
├── checkpoints/               # 学習済みモデル
├── output/                    # 推論・評価結果
└── logs/                      # ログファイル
```

## 使用方法

### 1. 環境設定

```bash
# conda環境のアクティベート
conda activate sip

# 依存関係のインストール
uv sync
```

### 2. モデル選択

#### 従来のmusicモデル
```bash
# 学習
uv run qpp4sip-train --model music --mode train --input_path dataset/train_resolved_retrieved.pkl

# 推論
uv run qpp4sip-inference --model music --mode inference --input_path dataset/dev_resolved_retrieved.pkl
```

#### QPP4SIPモデル（3パターン）
```bash
# Feature Fusion
uv run qpp4sip-train --model qpp4sip --qpp4sip_pattern feature_fusion --mode train --input_path dataset/train_resolved_retrieved.pkl

# Auxiliary Head
uv run qpp4sip-train --model qpp4sip --qpp4sip_pattern auxiliary_head --mode train --input_path dataset/train_resolved_retrieved.pkl

# Policy Gating
uv run qpp4sip-train --model qpp4sip --qpp4sip_pattern policy_gating --mode train --input_path dataset/train_resolved_retrieved.pkl
```

### 3. 評価

```bash
# 評価の実行
uv run qpp4sip-evaluate --prediction_path output/qpp4sip_feature_fusion --label_path dataset/dev_resolved_retrieved.pkl --task SIP --dataset_type dev
```

### 4. 一括実行（推奨）

#### GPU並列実行
```bash
# 3つのパターンを並列でGPU実行
nohup bash scripts/run_qpp4sip_gpu_experiments.sh > experiment_gpu_main.log 2>&1 &
```

#### 進行状況の監視
```bash
bash scripts/monitor_gpu_experiments.sh
```

#### 評価の実行
```bash
bash scripts/evaluate_qpp4sip_experiments.sh
```

## スクリプト詳細

### メインスクリプト

- `run_qpp4sip_gpu_experiments.sh`: GPU並列実行メインスクリプト
- `evaluate_qpp4sip_experiments.sh`: 評価実行スクリプト
- `monitor_gpu_experiments.sh`: GPU実験監視スクリプト

詳細は [README_scripts.md](README_scripts.md) を参照してください。

### 評価スクリプト

評価スクリプトは `evaluation/evaluation.py` にあります。

## 引数説明

### 基本設定
- `--model`: モデルタイプ（"music" または "qpp4sip"）
- `--qpp4sip_pattern`: QPP4SIP実装パターン（"feature_fusion", "auxiliary_head", "policy_gating"）
- `--mode`: 実行モード（"train" または "inference"）
- `--input_path`: 入力データのパス（必須）
- `--output_path`: 出力パス（デフォルト: "./output"）
- `--saved_model_path`: チェックポイントのパス（デフォルト: "./checkpoints"）

### モデルパラメータ
- `--hidden_size`: 隠れ層のサイズ（デフォルト: 768）
- `--dropout`: ドロップアウト率（デフォルト: 0.1）
- `--BiLSTM_layer`: BiLSTM層数（デフォルト: 1）

### 学習・推論パラメータ
- `--epoch_num`: エポック数（デフォルト: 20）
- `--batch_size`: バッチサイズ（デフォルト: 1）
- `--learning_rate`: 学習率（デフォルト: 1e-5）

## 出力ディレクトリ構造

```
output/
├── qpp4sip_feature_fusion/     # feature_fusionパターンの結果
├── qpp4sip_auxiliary_head/     # auxiliary_headパターンの結果
└── qpp4sip_policy_gating/      # policy_gatingパターンの結果

checkpoints/
├── qpp4sip_feature_fusion/     # feature_fusionパターンのチェックポイント
├── qpp4sip_auxiliary_head/     # auxiliary_headパターンのチェックポイント
└── qpp4sip_policy_gating/      # policy_gatingパターンのチェックポイント
```

## システム要件

- Python 3.12+
- PyTorch
- Transformers
- CUDA対応GPU（推奨: RTX 3090以上）
- メモリ: 16GB以上

## 注意事項

1. GPU実験は6つのRTX 3090 GPUを使用して並列実行されます
2. 各パターンは異なるGPUで実行されます
3. 実験完了まで数時間かかる場合があります
4. ログファイルで進行状況を確認できます

## ライセンス

このプロジェクトは研究目的で作成されています。
