"""
LASSOで使用する特徴量（指標）の設定
このファイルで使用する指標を一括管理
コメントアウトで簡単に有効化/無効化できます
"""
from typing import Dict, Tuple, List

# ============================================================================
# Post-retrieval QPPスコアの設定
# ============================================================================
# 使用したいメトリクスをコメントアウト解除して有効化

POST_RETRIEVAL_CONFIGS: Dict[str, Tuple[str, str]] = {
    'nqc': ('nqc.csv', 'nqc'),
    'clarity': ('clarity.csv', 'clarity'),
    'wig': ('wig.csv', 'wig'),
    'smv': ('smv.csv', 'smv'),
    'n_sigma_50': ('n_sigma_50.csv', 'n_sigma_50'),
}

# ============================================================================
# Pre-retrieval QPPスコアの設定
# ============================================================================
# 使用したいメトリクスをコメントアウト解除して有効化

PRE_RETRIEVAL_CONFIGS: Dict[str, Tuple[str, str]] = {
    'avgidf': ('avgidf.csv', 'avgidf'),
    'avgictf': ('avgictf.csv', 'avgictf'),
    'maxidf': ('maxidf.csv', 'maxidf'),
    'maxscq': ('maxscq.csv', 'maxscq'),
    'simplified_clarity': ('simplified_clarity.csv', 'simplified_clarity'),
}

# ============================================================================
# Baseモデル（FT-PLM）の設定
# ============================================================================
# 使用したい実験名をコメントアウト解除して有効化
# {dataset}は自動的にデータセット名に置換されます

BASE_EXPERIMENT_NAMES: List[str] = [
    "{dataset}_bert-base_lr2e-05_bs16_kfold5",
    "{dataset}_roberta-base_lr2e-05_bs16_earlystop_kfold5",
]

# ============================================================================
# NSPメトリクスの設定
# ============================================================================
# 使用したいメトリクスをコメントアウト解除して有効化

NSP_METRICS: Dict[str, str] = {
    # 'node_connectivity': 'node_connectivity',
    # 'average_node_connectivity': 'average_node_connectivity',
    # 'density': 'density',
}

# NSPメトリクスのカラム名マッピング（top_kは動的に決定される）
def get_nsp_metric_name(column_name: str, top_k: int) -> str:
    """NSPメトリクス名を生成"""
    return f'nsp_{column_name}_topk{top_k}'

