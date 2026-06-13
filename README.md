# QPP4SIP

DEIM 2026 論文の実験コードです。

> 柴田大暉, 酒井哲也. **意味的特徴およびクエリ性能予測の統合に基づく対話型検索における明確化必要性予測**. DEIM 2026.  
> 論文 PDF: https://pub-files.atlas.jp/fs/public/deim2026/ver_29/abstract/ja/4F-02.pdf

本リポジトリは、対話型検索における明確化必要性予測（Clarification Need Prediction; CNP）の実験用コードをまとめたものです。Query Performance Prediction（QPP）特徴量の算出、BERT / RoBERTa による予測スコアの出力、およびロジスティック回帰による特徴量統合を行います。

## リポジトリ構成

| パス | 内容 | 論文中の対応 |
| --- | --- | --- |
| `dataset/` | 入力データと検索結果を配置するディレクトリ | 4 節の AmbigNQ / DPR top-100 データ |
| `QPP/pre_retrieval/` | Pre-retrieval QPP 特徴量の算出 | 表 1: AvgICTF, AvgIDF, MaxIDF, MaxSCQ, SCS |
| `QPP/post_retrieval/` | Post-retrieval QPP 特徴量の算出 | 表 1: Clarity, NQC, SMV, WIG, n(σ%) |
| `SIP/FT-PLM/` | PLM ベースの明確化必要性予測モデルの fine-tuning / evaluation | 表 1: BERT, RoBERTa |
| `LogReg/LASSO/` | ロジスティック回帰による特徴量統合と統計的評価 | 表 2: Pre, Post, Pre/Post, Pre/Post/BERT, Pre/Post/RoBERTa |
| `LogReg/LASSO/config.py` | 統合実験で使う特徴量リストの設定 | DEIM 実験で用いた特徴量セット |

## セットアップ

実験は Python スクリプト群として実行します。最小構成は以下です。

```bash
git clone https://github.com/d-shibata7069/QPP4SIP.git
cd QPP4SIP

python -m venv .venv
source .venv/bin/activate

pip install -r QPP/requirements.txt
pip install -r SIP/FT-PLM/requirements.txt
```

一部のスクリプトは、プロジェクトルートを `/home/daiki_shibata/pj/QPP4SIP` と仮定しています。別の環境で実行する場合は、この場所にリポジトリを配置するか、各スクリプトの `BASE_DIR` / default path を実行環境に合わせて変更してください。

## データ配置

AmbigNQ と検索結果を `dataset/AmbigNQ/` 以下に配置します。

```text
dataset/AmbigNQ/
├── train.json
├── dev.json
├── dpr_train.json
└── dpr_dev.json
```

BM25 実験を行う場合、Post-retrieval QPP のスクリプトは同じ形式で `bm25_train.json` と `bm25_dev.json` を参照します。DEIM 論文の主結果は DPR top-100 に基づきます。

PLM スコアは、`SIP/FT-PLM/` の出力を `LogReg/LASSO/main.py` が読み込みます。デフォルトで参照する実験名は `LogReg/LASSO/config.py` で定義されています。

```text
SIP/FT-PLM/output/AmbigNQ/
├── AmbigNQ_bert-base_lr2e-05_bs16_kfold5/
│   ├── train_with_predictions.json
│   └── dev_with_predictions.json
└── AmbigNQ_roberta-base_lr2e-05_bs16_earlystop_kfold5/
    ├── train_with_predictions.json
    └── dev_with_predictions.json
```

## DEIM 実験の再現

特に記載がない限り、コマンドはリポジトリルートから実行します。

### 1. Pre-retrieval QPP 特徴量の算出

論文の表 1 における Pre-retrieval QPP、および表 2 の `Pre` 条件に対応します。

```bash
python QPP/pre_retrieval/main.py \
  --dataset AmbigNQ \
  --metric all \
  --split all
```

出力先:

```text
QPP/pre_retrieval/outputs/AmbigNQ/train_*.csv
QPP/pre_retrieval/outputs/AmbigNQ/dev_*.csv
```

### 2. Post-retrieval QPP 特徴量の算出

論文の表 1 における Post-retrieval QPP、および表 2 の `Post` 条件に対応します。

DEIM 論文で使用した Post-retrieval QPP 特徴量は、`clarity`, `nqc`, `smv`, `wig`, `n_sigma_50` です。

```bash
for metric in clarity nqc smv wig n_sigma_50; do
  python QPP/post_retrieval/main.py \
    --dataset AmbigNQ \
    --retrieval_method dpr \
    --metric "$metric" \
    --split all \
    --top_k 100
done
```

出力先:

```text
QPP/post_retrieval/outputs/AmbigNQ/dpr/train_*.csv
QPP/post_retrieval/outputs/AmbigNQ/dpr/dev_*.csv
```

### 3. PLM ベースラインの fine-tuning

論文の表 1 における BERT / RoBERTa 条件に対応します。ここで出力される logits は、表 2 の BERT / RoBERTa 付き統合条件でも特徴量として使用されます。

BERT:

```bash
python SIP/FT-PLM/main.py \
  --dataset AmbigNQ \
  --mode train \
  --model_name bert-base-uncased \
  --learning_rate 2e-5 \
  --batch_size 16 \
  --k_fold 5 \
  --output_dir SIP/FT-PLM/output
```

RoBERTa:

```bash
python SIP/FT-PLM/main.py \
  --dataset AmbigNQ \
  --mode train \
  --model_name roberta-base \
  --learning_rate 2e-5 \
  --batch_size 16 \
  --k_fold 5 \
  --early_stopping \
  --output_dir SIP/FT-PLM/output
```

その後、`*_with_predictions.json` を生成します。

```bash
python SIP/FT-PLM/main.py \
  --dataset AmbigNQ \
  --mode evaluate \
  --model_path SIP/FT-PLM/output/AmbigNQ/AmbigNQ_bert-base_lr2e-05_bs16_kfold5 \
  --output_dir SIP/FT-PLM/output

python SIP/FT-PLM/main.py \
  --dataset AmbigNQ \
  --mode evaluate \
  --model_path SIP/FT-PLM/output/AmbigNQ/AmbigNQ_roberta-base_lr2e-05_bs16_earlystop_kfold5 \
  --output_dir SIP/FT-PLM/output
```

### 4. ロジスティック回帰による特徴量統合

論文の表 2 に対応します。`--feature-types` で実験条件を指定します。

| 論文中の条件 | コマンドオプション |
| --- | --- |
| `(1) Pre` | `--feature-types pre` |
| `(2) Post` | `--feature-types post` |
| `(3) Pre/Post` | `--feature-types pre post` |
| `(4) Pre/Post/BERT` | `--feature-types pre post bert` |
| `(5) Pre/Post/RoBERTa` | `--feature-types pre post roberta` |

例:

```bash
python LogReg/LASSO/main.py \
  --dataset AmbigNQ \
  --retrieval-method dpr \
  --use-minmax-normalization \
  --no-cv \
  --delong-test \
  --feature-types pre post roberta
```

主要な特徴量組み合わせをまとめて実行する場合:

```bash
cd LogReg/LASSO
bash script/run_all_feature_combinations.sh
```

出力先:

```text
LogReg/LASSO/outputs/AmbigNQ/
```

## 主な結果

| 条件 | AUC-ROC |
| --- | ---: |
| Best single QPP: WIG | 0.5535 |
| Pre | 0.5964 |
| Post | 0.5150 |
| Pre/Post | 0.5853 |
| BERT | 0.6987 |
| RoBERTa | 0.7179 |
| Pre/Post/BERT | 0.6909 |
| Pre/Post/RoBERTa | 0.7158 |

## 引用

```bibtex
@inproceedings{shibata2026qpp4sip,
  title = {意味的特徴およびクエリ性能予測の統合に基づく対話型検索における明確化必要性予測},
  author = {柴田, 大暉 and 酒井, 哲也},
  booktitle = {DEIM Forum 2026},
  year = {2026}
}
```
