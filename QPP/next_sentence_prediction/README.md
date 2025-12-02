# Next Sentence Prediction (NSP) ツール

BERT NSP を使って以下を行います。

1. **Demoモード**: NSP がちゃんと働いているかの最小構成
2. **Graphモード**: top-k passage から coherency グラフを作り、NC/ANC を計算

## セットアップ

```bash
pip install transformers torch networkx pandas tqdm
```

## 使い方

### 1. Demoモード（既定）

```bash
python main.py
python main.py --mode demo --model_name bert-base-uncased --device cuda
```

### 2. Graphモード（top-k → グラフ → NC/ANC）

```bash
python main.py \
  --mode graph \
  --dataset INSCIT \
  --split dev \
  --top_k 20 \
  --threshold 0.55 \
  --device cuda \
  --batch_size 64
```

**パフォーマンス最適化オプション:**
- `--batch_size`: バッチサイズ（デフォルト: 64）。GPUメモリに応じて調整（32-128推奨）
- `--no_multi_gpu`: マルチGPUを無効化（デフォルト: 自動で複数GPUを使用）
- 複数GPU（4枚以上）がある場合、自動的にDataParallelで並列処理されます

出力: `outputs/<dataset>/<split>_nsp_graph.csv`

| conv_id | turn_id | node_connectivity | average_node_connectivity | density |
| --- | --- | --- | --- | --- |
| ... | ... | 2.0 | 1.3 | 0.21 |

## ディレクトリ構成

```
next_sentence_prediction/
├── main.py              # demo & graph mode
├── methods/
│   ├── __init__.py
│   ├── nsp.py           # NSPクラス
│   └── nsp_graph.py     # グラフ構築 & NC/ANC
├── outputs/             # graphモードの結果
└── README.md
```

## ハイライト

- top-k 文書ペアに BERT NSP を適用し、IsNext 確率がしきい値を超えたペアだけに有向エッジを張る
- networkx で coherency graph を構築し、`node_connectivity` / `average_node_connectivity` を計測
- Clarification 判定や QPP 指標のベースラインとして利用可能

