#!/bin/bash
# 全MoEゲーティング方式を一括実行して比較（推奨）
# 使い方: ./script/run_moe.sh

BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
exec ./script/run_all_moe.sh "$@"
