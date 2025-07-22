#!/bin/bash
mkdir -p /mnt/disk6/daiki/Datasets/INSCIT/models/DPR/retrieval_data/wikipedia/index/

# pyseriniを使用してインデックスを作成
uv run python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input  /mnt/disk6/daiki/Datasets/INSCIT/models/DPR/retrieval_data/wikipedia/wiki/ \
  --index /mnt/disk6/daiki/Datasets/INSCIT/models/DPR/retrieval_data/wikipedia/index/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw