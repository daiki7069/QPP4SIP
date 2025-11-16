"""
QPP評価スクリプト
"""
import argparse
import os
import pandas as pd
from evaluate_post_retrieval import visualize_all_metrics as visualize_post_retrieval_metrics
from evaluate_retrieval_data import visualize_all_metrics as visualize_retrieval_data_metrics
from auc import plot_roc_curves, plot_pr_curves


def evaluate_post_retrieval(split: str = 'train', dataset: str = 'INSCIT'):
    """Post-retrieval QPP指標の評価"""
    # ベースパスの設定
    base_dir = '/home/daiki_shibata/pj/QPP4SIP'
    
    # 入力ファイルの設定
    csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, f'{split}.csv')
    output_dir = os.path.join(base_dir, 'QPP', 'evaluate', 'outputs', dataset)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # メトリクス設定（メトリクス名、CSVパス、カラム名）
    post_retrieval_outputs_dir = os.path.join(base_dir, 'QPP', 'post_retrieval', 'outputs', dataset)
    metric_configs = {
        # 'entropy': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_entropy.csv'),
        #     'column': 'entropy'
        # },
        # 'unique_titles': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_unique_titles.csv'),
        #     'column': 'num_unique_titles'
        # },
        # 'nqc': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_nqc.csv'),
        #     'column': 'nqc'
        # },
        # 'lci': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_lci.csv'),
        #     'column': 'lci'
        # },
        'similarity': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_similarity.csv'),
            'column': 'mean_similarity'
        },
        # 'wig': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_wig.csv'),
        #     'column': 'wig'
        # },
    }
    
    # マージ設定
    merge_config = {
        'left_on': ['dialogue_id', 'turn_id'],
        'right_on': ['conv_id', 'turn_id'],
        'how': 'inner'
    }
    
    # 全指標の可視化
    print("=== Post-retrieval QPP指標の可視化 ===")
    visualize_post_retrieval_metrics(
        csv_path=csv_path,
        metric_configs=metric_configs,
        merge_config=merge_config,
        output_dir=output_dir
    )
    
    # AUC算出とROC曲線の可視化
    print("\n=== Post-retrieval AUC算出とROC曲線の可視化 ===")
    # 表示したいメトリクスを選択（Noneの場合は全て表示）
    # 例: selected_metrics = ['entropy', 'nqc', 'lci']  # 特定のメトリクスのみ表示
    # 例: selected_metrics = None  # 全てのメトリクスを表示
    selected_metrics = None  # ここで表示したいメトリクスを指定
    
    if selected_metrics is None:
        # 全てのメトリクスを表示
        metric_csv_paths = {
            name: config['csv_path'] 
            for name, config in metric_configs.items()
        }
    else:
        # 選択されたメトリクスのみ表示
        metric_csv_paths = {
            name: config['csv_path'] 
            for name, config in metric_configs.items()
            if name in selected_metrics
        }
        if not metric_csv_paths:
            print("警告: 選択されたメトリクスが見つかりません。全てのメトリクスを表示します。")
            metric_csv_paths = {
                name: config['csv_path'] 
                for name, config in metric_configs.items()
            }
    
    roc_output_path = f'{output_dir}/roc_curves_post_retrieval_{split}.png'
    auc_scores = plot_roc_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=roc_output_path,
        positive_class='clarification',
        merge_config=merge_config
    )
    
    # PR曲線の可視化
    print("\n=== Post-retrieval Average Precision算出とPR曲線の可視化 ===")
    pr_output_path = f'{output_dir}/pr_curves_post_retrieval_{split}.png'
    ap_scores = plot_pr_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=pr_output_path,
        positive_class='clarification',
        merge_config=merge_config
    )


def evaluate_retrieval_data(split: str = 'train', dataset: str = 'INSCIT'):
    """Retrieval data QPP指標の評価"""
    # ベースパスの設定
    base_dir = '/home/daiki_shibata/pj/QPP4SIP'
    
    # 入力ファイルの設定
    csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, f'{split}.csv')
    retrieval_csv_path = os.path.join(base_dir, 'QPP', 'retrieval_data', 'outputs', dataset, f'dpr_{split}_only_evidence.csv')
    output_dir = os.path.join(base_dir, 'QPP', 'evaluate', 'outputs', dataset)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # メトリクス設定（num_evidence_docsから始まる全ての評価指標）
    metric_configs = {
        'num_evidence_docs': {
            'csv_path': retrieval_csv_path,
            'column': 'num_evidence_docs'
        },
        # 'num_prev_evidence_docs': {
        #     'csv_path': retrieval_csv_path,
        #     'column': 'num_prev_evidence_docs'
        # },
        # 'found_ratio': {
        #     'csv_path': retrieval_csv_path,
        #     'column': 'found_ratio'
        # },
        'mrr': {
            'csv_path': retrieval_csv_path,
            'column': 'mrr'
        },
        'ndcg@1': {
            'csv_path': retrieval_csv_path,
            'column': 'ndcg@1'
        },
        'ndcg@5': {
            'csv_path': retrieval_csv_path,
            'column': 'ndcg@5'
        },
        'ndcg@10': {
            'csv_path': retrieval_csv_path,
            'column': 'ndcg@10'
        },
        'ndcg@20': {
            'csv_path': retrieval_csv_path,
            'column': 'ndcg@20'
        },
        'ndcg@50': {
            'csv_path': retrieval_csv_path,
            'column': 'ndcg@50'
        },
        'ndcg@100': {
            'csv_path': retrieval_csv_path,
            'column': 'ndcg@100'
        }
    }
    
    # 不足している評価指標があれば追加（CSVに存在する場合のみ）
    retrieval_df = pd.read_csv(retrieval_csv_path)
    additional_metrics = {
        'map': 'map',
        'precision@1': 'precision@1',
        'precision@5': 'precision@5',
        'precision@10': 'precision@10',
        'precision@20': 'precision@20',
        'precision@50': 'precision@50',
        'precision@100': 'precision@100',
        'recall@1': 'recall@1',
        'recall@5': 'recall@5',
        'recall@10': 'recall@10',
        'recall@20': 'recall@20',
        'recall@50': 'recall@50',
        'recall@100': 'recall@100',
        'hit_rate@1': 'hit_rate@1',
        'hit_rate@5': 'hit_rate@5',
        'hit_rate@10': 'hit_rate@10',
        'hit_rate@20': 'hit_rate@20',
        'hit_rate@50': 'hit_rate@50',
        'hit_rate@100': 'hit_rate@100'
    }
    
    for metric_name, column_name in additional_metrics.items():
        if column_name in retrieval_df.columns:
            metric_configs[metric_name] = {
                'csv_path': retrieval_csv_path,
                'column': column_name
            }
    
    # マージ設定
    merge_config = {
        'left_on': ['dialogue_id', 'turn_id'],
        'right_on': ['conv_id', 'turn_id'],
        'how': 'inner'
    }
    
    # 全指標の可視化
    print("=== Retrieval data QPP指標の可視化 ===")
    visualize_retrieval_data_metrics(
        csv_path=csv_path,
        metric_configs=metric_configs,
        merge_config=merge_config,
        output_dir=output_dir
    )
    
    # AUC算出とROC曲線の可視化
    print("\n=== Retrieval data AUC算出とROC曲線の可視化 ===")
    # 表示したいメトリクスを選択（Noneの場合は全て表示）
    # 例: selected_metrics = ['mrr', 'ndcg@5', 'ndcg@10']  # 特定のメトリクスのみ表示
    # 例: selected_metrics = None  # 全てのメトリクスを表示
    selected_metrics = ['num_evidence_docs', 'num_prev_evidence_docs', 'precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100']  # ここで表示したいメトリクスを指定
    
    if selected_metrics is None:
        # 全てのメトリクスを表示
        metric_csv_paths = {
            name: retrieval_csv_path  # 全て同じCSVファイル
            for name in metric_configs.keys()
        }
    else:
        # 選択されたメトリクスのみ表示
        metric_csv_paths = {
            name: retrieval_csv_path
            for name in metric_configs.keys()
            if name in selected_metrics
        }
        if not metric_csv_paths:
            print("警告: 選択されたメトリクスが見つかりません。全てのメトリクスを表示します。")
            metric_csv_paths = {
                name: retrieval_csv_path
                for name in metric_configs.keys()
            }
    
    roc_output_path = f'{output_dir}/roc_curves_retrieval_data_{split}.png'
    auc_scores = plot_roc_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=roc_output_path,
        positive_class='clarification',
        merge_config=merge_config
    )
    
    # PR曲線の可視化
    print("\n=== Retrieval data Average Precision算出とPR曲線の可視化 ===")
    pr_output_path = f'{output_dir}/pr_curves_retrieval_data_{split}.png'
    ap_scores = plot_pr_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=pr_output_path,
        positive_class='clarification',
        merge_config=merge_config
    )


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description="QPP評価スクリプト")
    
    # データセット名（必須）
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['INSCIT', 'AmbigNQ'],
        help='データセット名（INSCIT または AmbigNQ）'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['post_retrieval', 'retrieval_data', 'all'],
        default='post_retrieval',
        help='評価モード: post_retrieval (Post-retrieval指標), retrieval_data (Retrieval data指標), all (両方)'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'dev'],
        default='train',
        help='データセットの種類 (train または dev, デフォルト: train)'
    )
    
    args = parser.parse_args()
    
    print(f"データセット: {args.dataset}")
    print(f"スプリット: {args.split}")
    print(f"モード: {args.mode}")
    
    if args.mode == 'post_retrieval' or args.mode == 'all':
        evaluate_post_retrieval(split=args.split, dataset=args.dataset)
    
    if args.mode == 'retrieval_data' or args.mode == 'all':
        evaluate_retrieval_data(split=args.split, dataset=args.dataset)


if __name__ == '__main__':
    main()
