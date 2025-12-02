# Clarification予測モデルのFine Tuning

queryを入力として、response_typeがclarifyかどうかを予測するモデルをFine Tuningします。

## セットアップ

### 仮想環境の作成と有効化

```bash
# 仮想環境の作成
python3 -m venv .venv

# 仮想環境の有効化
source .venv/bin/activate
```

### パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 訓練

```bash
python main.py --mode train \
    --train_path /home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/train.json \
    --dev_path /home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/dev.json \
    --output_dir ./output \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --use_wandb \
    --use_class_weights
```

デフォルト値を使用する場合は、パスだけ指定すれば実行できます：

```bash
python main.py --mode train --early_stopping
```

### K-fold交差検証による訓練（Out-of-Fold予測）

訓練データ全体に対してリークなしで予測確率を付与するには、K-fold交差検証を使用します：

```bash
python main.py --mode train \
    --k_fold 5 \
    --num_epochs5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --use_class_weights
```

`--k_fold`を指定すると、以下の処理が実行されます：
1. 訓練データをK分割（StratifiedKFoldを使用）
2. 各foldでK-1個の分割でモデルを訓練
3. 残り1個の分割に対して予測確率を計算
4. 全foldの予測結果を結合して、訓練データ全体に対する予測確率を`{output_dir}/oof_predictions.txt`に保存

出力ファイル形式（TSV）：
```
query	true_label	prob_not_clarification	prob_clarification	true_response_type
...
```

クラス重みを使用する場合（不均衡データに対応）：

```bash
python main.py --mode train \
    --early_stopping \
    --use_class_weights \
    --use_wandb
```

### RoBERTaを使用する場合

RoBERTaで実験する場合は、`--model_name`でRoBERTaモデルを指定します：

```bash
python main.py --mode train \
    --model_name roberta-base \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --use_class_weights \
    --use_wandb
```

または、HuggingFace Hubの形式でも指定可能です：

```bash
python main.py --mode train \
    --model_name facebook/roberta-base \
    --early_stopping \
    --use_wandb
```

### wandbを使用した可視化

学習と評価の結果をwandbで可視化するには、`--use_wandb`フラグを追加します：

```bash
python main.py --mode train --early_stopping --use_wandb
```

評価時にもwandbを使用する場合：

```bash
python main.py --mode evaluate --use_wandb --save_predictions
```

wandbのモードは環境変数`WANDB_MODE`で制御できます：
- `online`: オンラインモード（デフォルト）
- `offline`: オフラインモード
- `disabled`: wandbを無効化

例：
```bash
WANDB_MODE=offline python main.py --mode train --use_wandb
```

### 評価

```bash
python main.py --mode evaluate \
    --dev_path /home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/dev.json \
    --save_predictions \
    --output_dir ./output
```

`--model_path`を指定しない場合、`--output_dir/best_model`が使用されます。

特定のモデルを指定して評価する場合：

```bash
python main.py --mode evaluate \
    --dev_path /home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/dev.json \
    --model_path ./output/bert-base_lr2e-05_bs16/best_model \
    --output_dir ./output \
    --save_predictions
```

#### K-fold交差検証モデルの評価

K-fold交差検証で訓練したモデルを評価する場合、または`--model_path`にK-foldディレクトリを指定すると、自動的に全foldのモデルでアンサンブル評価が実行されます：

```bash
python main.py --mode evaluate \
    --dev_path /home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/dev.json \
    --output_dir ./output/bert-base_lr2e-05_bs16_kfold5
```

この場合、各foldの`best_model`が自動的に検出され、全foldの予測確率の平均を取ってアンサンブル予測が行われます。

`--save_predictions`を使用すると、各ターンの予測結果と確率がTSV形式で`{output_dir}/predictions.txt`に保存されます。ファイルには以下の列が含まれます：
- `query`: クエリ文
- `true_label`: 正解ラベル（0: Not Clarification, 1: Clarification）
- `predicted_label`: 予測ラベル（0: Not Clarification, 1: Clarification）
- `prob_not_clarification`: Not Clarificationの予測確率
- `prob_clarification`: Clarificationの予測確率
- `true_response_type`: 正解のresponse_type

### 推論（単一クエリ）

```bash
python main.py --mode predict \
    --query "What are the different colors of glass used for beer bottles?" \
    --output_dir ./output
```

## オプション

- `--model_name`: 事前学習済みモデル名（デフォルト: `bert-base-uncased`）
  - BERT: `bert-base-uncased`, `bert-large-uncased`など
  - RoBERTa: `roberta-base`, `roberta-large`, `facebook/roberta-base`など
  - その他のTransformersモデルも使用可能
- `--max_length`: 最大シーケンス長（デフォルト: 512）
- `--num_epochs`: 訓練エポック数（デフォルト: 5）
- `--batch_size`: バッチサイズ（デフォルト: 16）
- `--k_fold`: K-fold交差検証のK値（指定するとK-fold交差検証モードで実行）
- `--learning_rate`: 学習率（デフォルト: 2e-5）
- `--early_stopping`: Early stoppingを有効化
- `--use_class_weights`: クラス重みを使用（不均衡データに対応）
- `--fp16`: FP16を使用（GPU使用時）
- `--use_wandb`: wandbを使用してログを記録
- `--wandb_project`: wandbプロジェクト名（デフォルト: `FT-PLM-Clarification`）

## データ形式

入力データは以下の形式のJSONファイルです：

```json
[
  [
    {
      "query": "How does ultraviolet spoil beer?",
      "response_type": "directAnswer [SEP] directAnswer"
    },
    {
      "query": "What are the different colors of glass used for beer bottles?",
      "response_type": "directAnswer [SEP] clarification"
    }
  ]
]
```

`response_type`が`[SEP]`で分けられている場合は、前方のタイプを使用します。

