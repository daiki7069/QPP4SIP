#!/bin/bash
# pyseriniを使用してインデックスを作成
uv run python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input  /mnt/disk6/daiki/Datasets/iKAT/ikat_demo/collection \
  --index /mnt/disk6/daiki/Datasets/iKAT/ikat_demo/index/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw