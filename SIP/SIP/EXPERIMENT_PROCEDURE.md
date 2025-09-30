# TSVファイルをPKLに変換してSIP処理を行う実験手順

## 📋 実験概要
TSVファイル形式のINSCITデータセット（QPP特徴量付き）をPKL形式に変換し、既存のSIP処理パイプラインで学習・推論・評価を行う実験。

## 🎯 実験目的
1. TSVファイルの豊富なQPP特徴量を活用
2. 既存のSIP処理パイプラインとの互換性を保つ
3. QPP4SIPモデルとMusicモデルの性能比較

## 📁 ファイル構成
```
SIP/SIP/
├── model/
│   ├── tsv_to_pkl_converter.py    # TSV→PKL変換スクリプト
│   ├── load_pkl.py                # データ読み込み（修正済み）
│   ├── run.py                     # メイン実行スクリプト
│   └── ...
├── scripts/
│   ├── convert_tsv_to_pkl.sh      # 変換実行スクリプト
│   └── run_with_tsv_conversion.sh # 統合実行スクリプト
├── evaluation/
│   └── evaluation.py              # 評価スクリプト（修正済み）
├── show_results_summary.py        # サマリー表示・保存スクリプト
└── dataset/pkl/                   # 変換されたPKLファイル
```

## 🔧 実験手順

### Step 1: 環境準備
```bash
cd SIP/SIP
# 必要な依存関係をインストール（uvが自動で処理）
```

### Step 2: TSVファイルをPKLに変換
```bash
# 基本変換
./scripts/convert_tsv_to_pkl.sh

# カスタムパスで変換
./scripts/convert_tsv_to_pkl.sh \
  /path/to/train.tsv \
  /path/to/dev.tsv \
  /path/to/test.tsv \
  ./dataset/pkl
```

**変換されるファイル:**
- `./dataset/pkl/bm25_train.pkl` (訓練データ)
- `./dataset/pkl/bm25_dev.pkl` (開発データ)

### Step 3: 学習の実行

#### QPP4SIPモデル（feature_fusionパターン）
```bash
uv run python model/run.py \
  --mode train \
  --task SIP \
  --name QPP4SIP \
  --dataset QPP4SIP \
  --model qpp4sip \
  --qpp4sip_pattern feature_fusion \
  --input_path ./dataset/pkl/bm25_train.pkl \
  --output_path ./output/qpp4sip_feature_fusion \
  --saved_model_path ./checkpoints/qpp4sip_feature_fusion \
  --log_path ./logs/qpp4sip_feature_fusion \
  --hidden_size 768 \
  --dropout 0.1 \
  --BiLSTM_layers 1 \
  --epoch_num 20 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --lr_crf 1e-3 \
  --accumulation_steps 1 \
  --clip 1.0 \
  --max_utterance_len 128 \
  --max_context_len 384 \
  --random_seed 42 \
  --class_imbalance_ratio 6.5 \
  --focal_gamma 2.0
```

#### Musicモデル（比較用）
```bash
uv run python model/run.py \
  --mode train \
  --task SIP \
  --name Music \
  --dataset QPP4SIP \
  --model music \
  --input_path ./dataset/pkl/bm25_train.pkl \
  --output_path ./output/music \
  --saved_model_path ./checkpoints/music \
  --log_path ./logs/music \
  --hidden_size 768 \
  --dropout 0.1 \
  --BiLSTM_layers 1 \
  --epoch_num 20 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --lr_crf 1e-3 \
  --accumulation_steps 1 \
  --clip 1.0 \
  --max_utterance_len 128 \
  --max_context_len 384 \
  --random_seed 42
```

### Step 4: 推論の実行

#### QPP4SIPモデル
```bash
uv run python model/run.py \
  --mode inference \
  --task SIP \
  --name QPP4SIP \
  --dataset QPP4SIP \
  --model qpp4sip \
  --qpp4sip_pattern feature_fusion \
  --input_path ./dataset/pkl/bm25_dev.pkl \
  --output_path ./output/qpp4sip_feature_fusion \
  --saved_model_path ./checkpoints/qpp4sip_feature_fusion \
  --log_path ./logs/qpp4sip_feature_fusion \
  --hidden_size 768 \
  --dropout 0.1 \
  --BiLSTM_layers 1 \
  --epoch_num 20 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --lr_crf 1e-3 \
  --accumulation_steps 1 \
  --clip 1.0 \
  --max_utterance_len 128 \
  --max_context_len 384 \
  --random_seed 42 \
  --class_imbalance_ratio 6.5 \
  --focal_gamma 2.0
```

#### Musicモデル
```bash
uv run python model/run.py \
  --mode inference \
  --task SIP \
  --name Music \
  --dataset QPP4SIP \
  --model music \
  --input_path ./dataset/pkl/bm25_dev.pkl \
  --output_path ./output/music \
  --saved_model_path ./checkpoints/music \
  --log_path ./logs/music \
  --hidden_size 768 \
  --dropout 0.1 \
  --BiLSTM_layers 1 \
  --epoch_num 20 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --lr_crf 1e-3 \
  --accumulation_steps 1 \
  --clip 1.0 \
  --max_utterance_len 128 \
  --max_context_len 384 \
  --random_seed 42
```

### Step 5: 評価の実行

#### QPP4SIPモデル
```bash
uv run python evaluation/evaluation.py \
  --prediction_path ./output/qpp4sip_feature_fusion \
  --label_path ./dataset/pkl/bm25_dev.pkl \
  --task SIP \
  --dataset_type dev \
  --epoch_num 20 \
  --dataset QPP4SIP
```

#### Musicモデル
```bash
uv run python evaluation/evaluation.py \
  --prediction_path ./output/music \
  --label_path ./dataset/pkl/bm25_dev.pkl \
  --task SIP \
  --dataset_type dev \
  --epoch_num 20 \
  --dataset QPP4SIP
```

### Step 6: 結果サマリーの生成

#### QPP4SIPモデル
```bash
uv run python show_results_summary.py \
  --result_file ./output/qpp4sip_feature_fusion/result.dev.txt \
  --dataset_type dev \
  --output_file ./output/qpp4sip_feature_fusion/summary_dev.txt
```

#### Musicモデル
```bash
uv run python show_results_summary.py \
  --result_file ./output/music/result.dev.txt \
  --dataset_type dev \
  --output_file ./output/music/summary_dev.txt
```

## 📊 結果ファイルの場所

### QPP4SIPモデル
```
./output/qpp4sip_feature_fusion/
├── result.dev.txt          # 生の評価結果データ
├── summary_dev.txt         # フォーマットされたサマリー
├── dev.1.txt ～ dev.20.txt  # 各エポックの推論結果
└── test.1.txt ～ test.20.txt # 各エポックのテストデータ推論結果
```

### Musicモデル
```
./output/music/
├── result.dev.txt          # 生の評価結果データ
├── summary_dev.txt         # フォーマットされたサマリー
├── dev.1.txt ～ dev.20.txt  # 各エポックの推論結果
└── test.1.txt ～ test.20.txt # 各エポックのテストデータ推論結果
```

## 🎯 実験結果（参考）

### 性能比較
| モデル | 最高F1スコア | 最高正解率 | Initiative予測率 | 最良エポック |
|--------|-------------|-----------|-----------------|-------------|
| Music | 53.85 | 86.85 | 18.18% | 16 |
| QPP4SIP | **60.21** | 81.65 | **27.17%** | 12 |

### 主要な成果
- ✅ TSVファイルのQPP特徴量を活用
- ✅ 既存パイプラインとの完全互換性
- ✅ QPP4SIPモデルがMusicモデルを上回る性能
- ✅ Initiative予測能力の大幅改善

## 🔧 トラブルシューティング

### よくある問題
1. **pandasが見つからない**: `uv run --with pandas` を使用
2. **パスエラー**: 絶対パスを使用
3. **メモリ不足**: バッチサイズを調整

### デバッグ用コマンド
```bash
# データ変換の確認
uv run python -c "
from model.load_pkl import convert_dialogue_to_conversations
import pickle
with open('./dataset/pkl/bm25_train.pkl', 'rb') as f:
    data = pickle.load(f)
conversations = convert_dialogue_to_conversations(data)
print(f'会話数: {len(conversations)}')
"
```

## 📝 注意事項
- 学習には時間がかかります（20エポックで約1時間）
- GPUメモリを確認してください
- 結果ファイルは自動的に上書きされます
- 異なるモデル間でパラメータを統一してください
