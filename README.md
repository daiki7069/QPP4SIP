# QPP4SIP

Code for the DEIM 2026 paper:

> 柴田大暉, 酒井哲也. **意味的特徴およびクエリ性能予測の統合に基づく対話型検索における明確化必要性予測**. DEIM 2026.  
> Paper: https://pub-files.atlas.jp/fs/public/deim2026/ver_29/abstract/ja/4F-02.pdf

This repository contains the experimental code for clarification need prediction in conversational search. The main pipeline computes Query Performance Prediction (QPP) features, obtains BERT/RoBERTa prediction scores, and combines them with logistic regression.

## Repository layout

| Path | Description | Paper mapping |
| --- | --- | --- |
| `dataset/` | Input data and retrieval results. | AmbigNQ / DPR top-100 data used in Section 4 |
| `QPP/pre_retrieval/` | Pre-retrieval QPP feature extraction. | Table 1: AvgICTF, AvgIDF, MaxIDF, MaxSCQ, SCS |
| `QPP/post_retrieval/` | Post-retrieval QPP feature extraction. | Table 1: Clarity, NQC, SMV, WIG, n(σ%) |
| `SIP/FT-PLM/` | Fine-tuning and evaluation of PLM-based clarification predictors. | Table 1: BERT, RoBERTa |
| `LogReg/LASSO/` | Logistic-regression integration and statistical evaluation. | Table 2: Pre, Post, Pre/Post, Pre/Post/BERT, Pre/Post/RoBERTa |
| `LogReg/LASSO/config.py` | Feature list used by the integration script. | Defines the DEIM feature set |

## Setup

The experiments were run as a collection of Python scripts. A minimal setup is:

```bash
git clone https://github.com/d-shibata7069/QPP4SIP.git
cd QPP4SIP

python -m venv .venv
source .venv/bin/activate

pip install -r QPP/requirements.txt
pip install -r SIP/FT-PLM/requirements.txt
```

Several scripts currently assume the project root is `/home/daiki_shibata/pj/QPP4SIP`. When running in another environment, either place the repository there or update the `BASE_DIR` / default path definitions in the corresponding scripts.

## Data layout

Place AmbigNQ and retrieval outputs under `dataset/AmbigNQ/`:

```text
dataset/AmbigNQ/
├── train.json
├── dev.json
├── dpr_train.json
└── dpr_dev.json
```

For BM25 experiments, the post-retrieval scripts expect the same pattern with `bm25_train.json` and `bm25_dev.json`. The DEIM paper results use DPR top-100 retrieval.

PLM scores are consumed by `LogReg/LASSO/main.py` from the output directory of `SIP/FT-PLM/`. The default experiment names are defined in `LogReg/LASSO/config.py`:

```text
SIP/FT-PLM/output/AmbigNQ/
├── AmbigNQ_bert-base_lr2e-05_bs16_kfold5/
│   ├── train_with_predictions.json
│   └── dev_with_predictions.json
└── AmbigNQ_roberta-base_lr2e-05_bs16_earlystop_kfold5/
    ├── train_with_predictions.json
    └── dev_with_predictions.json
```

## Reproducing the DEIM experiments

Run commands from the repository root unless otherwise noted.

### 1. Compute Pre-retrieval QPP features

This corresponds to the Pre-retrieval QPP rows in Table 1 and to the `Pre` component in Table 2.

```bash
python QPP/pre_retrieval/main.py \
  --dataset AmbigNQ \
  --metric all \
  --split all
```

Expected outputs:

```text
QPP/pre_retrieval/outputs/AmbigNQ/train_*.csv
QPP/pre_retrieval/outputs/AmbigNQ/dev_*.csv
```

### 2. Compute Post-retrieval QPP features

This corresponds to the Post-retrieval QPP rows in Table 1 and to the `Post` component in Table 2.

The paper uses the following post-retrieval features: `clarity`, `nqc`, `smv`, `wig`, and `n_sigma_50`.

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

Expected outputs:

```text
QPP/post_retrieval/outputs/AmbigNQ/dpr/train_*.csv
QPP/post_retrieval/outputs/AmbigNQ/dpr/dev_*.csv
```

### 3. Fine-tune PLM baselines

This corresponds to the BERT and RoBERTa rows in Table 1. The generated logits are also used as features in Table 2.

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

Then evaluate to write `*_with_predictions.json` files:

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

### 4. Run logistic-regression integration

This corresponds to Table 2. The `--feature-types` argument selects the experimental condition.

| Paper condition | Command option |
| --- | --- |
| `(1) Pre` | `--feature-types pre` |
| `(2) Post` | `--feature-types post` |
| `(3) Pre/Post` | `--feature-types pre post` |
| `(4) Pre/Post/BERT` | `--feature-types pre post bert` |
| `(5) Pre/Post/RoBERTa` | `--feature-types pre post roberta` |

Example:

```bash
python LogReg/LASSO/main.py \
  --dataset AmbigNQ \
  --retrieval-method dpr \
  --use-minmax-normalization \
  --no-cv \
  --delong-test \
  --feature-types pre post roberta
```

To run the major combinations used during the experiments:

```bash
cd LogReg/LASSO
bash script/run_all_feature_combinations.sh
```

Outputs are written under:

```text
LogReg/LASSO/outputs/AmbigNQ/
```

## Main results

| Condition | AUC-ROC |
| --- | ---: |
| Best single QPP: WIG | 0.5535 |
| Pre | 0.5964 |
| Post | 0.5150 |
| Pre/Post | 0.5853 |
| BERT | 0.6987 |
| RoBERTa | 0.7179 |
| Pre/Post/BERT | 0.6909 |
| Pre/Post/RoBERTa | 0.7158 |

## Citation

```bibtex
@inproceedings{shibata2026qpp4sip,
  title = {意味的特徴およびクエリ性能予測の統合に基づく対話型検索における明確化必要性予測},
  author = {柴田, 大暉 and 酒井, 哲也},
  booktitle = {DEIM Forum 2026},
  year = {2026}
}
```
