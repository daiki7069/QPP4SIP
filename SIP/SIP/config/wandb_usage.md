# wandb使用ガイド

このドキュメントは、wandbを使用して学習曲線の可視化とevaluation結果の表示を行う方法を説明します。

## 概要

`config/wandb.py`には、wandbによる学習曲線の可視化とevaluation結果の表示を行うための設定とユーティリティ関数が含まれています。

## 初回セットアップ

### 1. wandbのインストール

```bash
pip install wandb
```

### 2. wandbアカウントの作成

1. [https://wandb.ai/](https://wandb.ai/) にアクセス
2. アカウントを作成（GitHub/Googleアカウントでログイン可能）

### 3. APIキーの取得と設定

#### 方法A：コマンドラインでログイン（推奨）

```bash
wandb login
```

上記コマンドを実行すると、APIキーの入力が求められます。
APIキーは [https://wandb.ai/settings](https://wandb.ai/settings) の「API keys」セクションで取得できます。

#### 方法B：環境変数で設定

```bash
export WANDB_API_KEY=your_api_key_here
```

または、`~/.netrc` ファイルに設定：

```bash
# ~/.netrcファイルを作成または編集
machine api.wandb.ai
login your_username
password your_api_key
```

#### 方法C：設定ファイルに記載（非推奨）

`config/wandb.py`に直接記載することもできますが、セキュリティ上の理由で推奨されません。

### 4. ログイン状態の確認

```bash
wandb status
```

## 機能

1. **学習中のメトリクス記録**
   - 各エポックの学習損失
   - 学習率
   - タスク別の損失コンポーネント（SIPタスクの場合、distance_crfとmle_e）

2. **評価結果の記録**
   - F1スコア、精度、再現率
   - ラベル別の評価指標
   - 対話単位の統計情報
   - Initiative/Non-initiativeの詳細評価指標

## 使用方法

### 1. 学習時にwandbを使用する

学習時に`--use_wandb`フラグを追加します：

```bash
python run.py --mode train \
    --model music \
    --input_path ./dataset/dev_score.pkl \
    --output_path ./output/music \
    --saved_model_path ./checkpoints/music \
    --log_path ./logs/music \
    --epoch_num 20 \
    --use_wandb \
    --wandb_mode online
```

### 2. 評価時にwandbを使用する

評価時に`--use_wandb`フラグを追加します：

```bash
python run.py --mode evaluation \
    --input_path ./dataset/dev_score.pkl \
    --output_path ./output/music \
    --epoch_num 20 \
    --use_wandb \
    --wandb_mode online
```

### 3. wandbモードの選択

`--wandb_mode`オプションでモードを選択できます：

- `online`: オンラインモード（デフォルト）。wandbサーバーにデータを送信
- `offline`: オフラインモード。データをローカルに保存し、後でアップロード
- `disabled`: wandbを無効化

```bash
# オフラインモードでの実行
python run.py --mode train --use_wandb --wandb_mode offline \
    --input_path ./dataset/dev_score.pkl \
    --output_path ./output/music
```

## 記録されるメトリクス

### 学習中のメトリクス

SIPタスクの場合、以下のメトリクスが記録されます：

- `train/loss_overall`: 全体の損失
- `train/loss_distance_crf`: CRF損失
- `train/loss_mle_e`: MLE損失
- `learning_rate`: 学習率
- `epoch`: エポック番号

### 評価時のメトリクス

以下のメトリクスが記録されます：

#### 基本的な評価指標

- `eval/f1`: F1スコア
- `eval/precision`: 精度（平均）
- `eval/recall`: 再現率（平均）
- `eval/accuracy`: 正解率（平均）

#### ラベル別の評価指標

- `eval/accuracy_non_initiative`: Non-initiativeの精度
- `eval/accuracy_initiative`: Initiativeの精度
- `eval/precision_non_initiative`: Non-initiativeの精度
- `eval/precision_initiative`: Initiativeの精度
- `eval/recall_non_initiative`: Non-initiativeの再現率
- `eval/recall_initiative`: Initiativeの再現率

#### 対話単位の統計

- `eval/avg_conversation_accuracy`: 対話単位の平均精度
- `eval/total_conversations`: 総対話数
- `eval/total_turns`: 総ターン数
- `eval/avg_turns_per_conversation`: 対話あたりの平均ターン数
- `eval/conversations_with_initiative`: Initiativeを含む対話数

#### Initiative詳細指標

- `eval/initiative_acc`: Initiativeの精度
- `eval/initiative_precision`: Initiativeの精度（Precision）
- `eval/initiative_recall`: Initiativeの再現率
- `eval/initiative_f1`: InitiativeのF1スコア
- `eval/initiative_hit_num`: Initiative正解数
- `eval/initiative_total_num`: Initiative総数

#### Non-initiative詳細指標

- `eval/non_initiative_acc`: Non-initiativeの精度
- `eval/non_initiative_precision`: Non-initiativeの精度（Precision）
- `eval/non_initiative_recall`: Non-initiativeの再現率
- `eval/non_initiative_f1`: Non-initiativeのF1スコア
- `eval/non_initiative_hit_num`: Non-initiative正解数
- `eval/non_initiative_total_num`: Non-initiative総数

## 例：完全な実行フロー

```bash
# 1. 学習
python run.py --mode train \
    --model music \
    --input_path ./dataset/dev_score.pkl \
    --output_path ./output/music \
    --saved_model_path ./checkpoints/music \
    --log_path ./logs/music \
    --epoch_num 20 \
    --use_wandb \
    --wandb_mode online

# 2. 推論
python run.py --mode inference \
    --model music \
    --input_path ./dataset/dev_score.pkl \
    --output_path ./output/music \
    --saved_model_path ./checkpoints/music \
    --epoch_num 20

# 3. 評価（wandbに記録）
python run.py --mode evaluation \
    --input_path ./dataset/dev_score.pkl \
    --output_path ./output/music \
    --epoch_num 20 \
    --use_wandb \
    --wandb_mode online
```

## 設定のカスタマイズ

### プロジェクト名の変更

`config/wandb.py`の`init_wandb`関数呼び出しでプロジェクト名を変更できます：

```python
init_wandb(project_name="MyProject", experiment_name=experiment_name, 
          config_dict=config_dict, mode=wandb_mode)
```

### 実験名のカスタマイズ

`config/wandb.py`の`get_experiment_name`関数を変更することで、実験名の生成方法をカスタマイズできます。

## トラブルシューティング

### wandbが見つからないエラー

以下のコマンドでwandbをインストールしてください：

```bash
pip install wandb
```

### ログインエラー

APIキーが正しく設定されていない場合、以下のコマンドで再ログインしてください：

```bash
wandb login --relogin
```

### ログイン状態の確認

```bash
wandb status
```

現在のログイン状態、プロジェクト、エンティティ（ユーザー名）などが表示されます。

### ログアウト

別のアカウントに切り替えたい場合：

```bash
wandb logout
```

### オフラインモードで後からアップロード

オフラインモードで実行したデータを後からアップロードするには：

```bash
wandb sync wandb/offline-run-*
```

## 参考

- [wandb公式ドキュメント](https://docs.wandb.ai/)
- `config/wandb.py`: wandb設定とユーティリティ関数
- `model/trainer.py`: 学習中のメトリクス記録
- `run.py`: メインの実行スクリプト

