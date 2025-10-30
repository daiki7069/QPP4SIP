# Post-retrieval QPP Analysis

DPRの結果ファイルを使用してPost-retrieval QPPの分析を行うためのシンプルなツールセットです。

## 機能

### 1. データローダー (`data_loader.py`)
- DPRの結果ファイル（JSON形式）を読み込み
- 各ターンの文書セットとスコアを扱いやすい形で提供

### 2. スコア分布分析 (`score_analysis.py`)
- 各ターンにおけるスコアの基本統計（平均、標準偏差、最小値、最大値）

### 3. 文書内容分析 (`content_analysis.py`)
- 質問と文書のペアデータを取得（BERT等での分析用）

## 使用方法

```python
from post_retrieval.data_loader import DPRResultLoader, ScoreAnalyzer, ContentAnalyzer

# データの読み込み
loader = DPRResultLoader("/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results")
dev_data = loader.load_data("dev")

# スコア分析
score_analyzer = ScoreAnalyzer(dev_data)
score_stats = score_analyzer.get_score_statistics()

# 内容分析
content_analyzer = ContentAnalyzer(dev_data)
question_doc_pairs = content_analyzer.get_question_document_pairs()
```

## 依存関係

```
pandas>=1.3.0
numpy>=1.21.0
```
