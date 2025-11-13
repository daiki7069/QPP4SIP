# Post-retrieval QPP Analysis

DPRの結果ファイルを使用してPost-retrieval QPPの分析を行うためのシンプルなツールセットです。

## 機能

### 1. データローダー (`data_loader.py`)
- DPRの結果ファイル（JSON形式）を読み込み
- 各ターンの文書セットとスコアを扱いやすい形で提供

### 2. スコア分布分析 (`score_analysis.py`)
- 各ターンにおけるスコアの基本統計（平均、標準偏差、最小値、最大値）

### 3. QPP手法の実装

各手法は独立したクラスとして実装されています：

- **`base.py`** - `BaseQPPAnalyzer`: 共通機能（BERT埋め込み、キャッシュ、タイトル抽出など）を提供するベースクラス
- **`nqc.py`** - `NQC`: Normalized Query Clarity の計算
- **`lci.py`** - `LCI`: Local Concentration Index の計算
- **`entropy.py`** - `Entropy`: タイトル分布の正規化エントロピーの計算
- **`unique_titles.py`** - `UniqueTitles`: ユニークタイトル数の計算
- **`similarity.py`** - `Similarity`: 文書間の類似度統計の計算

## 使用方法

### コマンドラインから実行

```bash
# 特定の手法を実行
python main.py --metric nqc --split dev --top_k 100

# 全ての手法を実行
python main.py --metric all --split dev --top_k 100
```

### Pythonコードから使用

```python
from data_loader import DPRResultLoader
from nqc import NQC
from lci import LCI
from entropy import Entropy
from unique_titles import UniqueTitles
from similarity import Similarity

# データの読み込み
loader = DPRResultLoader("path/to/dpr_dev.json")
turn_data_list = loader.load_data()

# NQCを計算
nqc_analyzer = NQC(turn_data_list)
nqc_results = nqc_analyzer.compute(top_k=100)

# LCIを計算
lci_analyzer = LCI(turn_data_list)
lci_results = lci_analyzer.compute(top_k=100, window=3)

# エントロピーを計算
entropy_analyzer = Entropy(turn_data_list)
entropy_results = entropy_analyzer.compute(top_k=100)

# ユニークタイトル数を計算
unique_titles_analyzer = UniqueTitles(turn_data_list)
unique_titles_results = unique_titles_analyzer.compute(top_k=100)

# 類似度統計を計算（BERT埋め込みを使用）
similarity_analyzer = Similarity(turn_data_list, device='cuda', cache_dir='.embedding_cache')
similarity_results = similarity_analyzer.compute(top_k=100)
```

### ファイルから直接計算

```python
from nqc import NQC

# ファイルから直接計算してJSONに追記
NQC.compute_from_files(
    dpr_json_path="path/to/dpr_dev.json",
    base_json_path="path/to/dev.json",
    output_json_path="path/to/dev_nqc.json",
    output_csv_path="path/to/outputs/dev_nqc.csv",
    top_k=100
)
```

## 依存関係

```
pandas>=1.3.0
numpy>=1.21.0
torch>=1.9.0
transformers>=4.0.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```
