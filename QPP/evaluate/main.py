"""
QPP評価スクリプト
"""
import argparse
import os
import pandas as pd
from evaluate_post_retrieval import visualize_all_metrics as visualize_post_retrieval_metrics
from evaluate_retrieval_data import visualize_all_metrics as visualize_retrieval_data_metrics
from evaluate_neural_qpp import visualize_all_metrics as visualize_neural_qpp_metrics
from evaluate_nsp_graph import visualize_all_metrics as visualize_nsp_graph_metrics
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
        'nqc': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_nqc.csv'),
            'column': 'nqc'
        },
        'smv': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_smv.csv'),
            'column': 'smv'
        },
        'nsv': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_nsv.csv'),
            'column': 'nsv'
        },
        # 'lci': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_lci.csv'),
        #     'column': 'lci'
        # },
        'similarity': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_similarity.csv'),
            'column': 'mean_similarity'
        },
        'wig': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_wig.csv'),
            'column': 'wig'
        },
        'acc': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_coherency.csv'),
            'column': 'acc'
        },
        # 'wacc': {
        #     'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_coherency.csv'),
        #     'column': 'wacc'
        # },
        'clarity': {
            'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_clarity.csv'),
            'column': 'clarity'
        },
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


def evaluate_nsp_graph(split: str = 'train', dataset: str = 'INSCIT'):
    """NSP Graph QPP指標の評価（top_k=10,20,50,100）"""
    # ベースパスの設定
    base_dir = '/home/daiki_shibata/pj/QPP4SIP'
    
    # 入力ファイルの設定
    csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, f'{split}.csv')
    output_dir = os.path.join(base_dir, 'QPP', 'evaluate', 'outputs', dataset)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # NSP Graph出力ディレクトリ
    nsp_graph_outputs_dir = os.path.join(base_dir, 'QPP', 'next_sentence_prediction', 'outputs', dataset)
    
    # 存在するtop_k値を自動検出
    TOP_K_VALUES = []
    if os.path.exists(nsp_graph_outputs_dir):
        for filename in os.listdir(nsp_graph_outputs_dir):
            # {split}_nsp_graph_topk{top_k}.csv の形式を検出
            if filename.startswith(f'{split}_nsp_graph_topk') and filename.endswith('.csv'):
                try:
                    # topk{数字}の部分を抽出
                    top_k_str = filename.replace(f'{split}_nsp_graph_topk', '').replace('.csv', '')
                    top_k = int(top_k_str)
                    TOP_K_VALUES.append(top_k)
                except ValueError:
                    continue
    
    # top_k値をソート
    TOP_K_VALUES = sorted(TOP_K_VALUES)
    
    if not TOP_K_VALUES:
        print(f"警告: NSP Graphの出力ファイルが見つかりません: {nsp_graph_outputs_dir}")
        print(f"  先にNSP Graphの計算を実行してください。")
        return
    
    print(f"検出されたtop_k値: {TOP_K_VALUES}")
    
    # 全てのtop_k値をまとめて処理
    # メトリクス設定（全top_k値を含む）
    metric_configs = {}
    metric_csv_paths = {}
    metric_column_map = {}
    
    for top_k in TOP_K_VALUES:
        csv_file = os.path.join(nsp_graph_outputs_dir, f'{split}_nsp_graph_topk{top_k}.csv')
        
        # NC (Node Connectivity)
        metric_name_nc = f'nsp_nc_topk{top_k}'
        metric_configs[metric_name_nc] = {
            'csv_path': csv_file,
            'column': 'node_connectivity',
            'top_k': top_k
        }
        metric_csv_paths[metric_name_nc] = csv_file
        metric_column_map[metric_name_nc] = 'node_connectivity'
        
        # ANC (Average Node Connectivity)
        metric_name_anc = f'nsp_anc_topk{top_k}'
        metric_configs[metric_name_anc] = {
            'csv_path': csv_file,
            'column': 'average_node_connectivity',
            'top_k': top_k
        }
        metric_csv_paths[metric_name_anc] = csv_file
        metric_column_map[metric_name_anc] = 'average_node_connectivity'
        
        # Density
        metric_name_density = f'nsp_density_topk{top_k}'
        metric_configs[metric_name_density] = {
            'csv_path': csv_file,
            'column': 'density',
            'top_k': top_k
        }
        metric_csv_paths[metric_name_density] = csv_file
        metric_column_map[metric_name_density] = 'density'
    
    # マージ設定
    merge_config = {
        'left_on': ['dialogue_id', 'turn_id'],
        'right_on': ['conv_id', 'turn_id'],
        'how': 'inner'
    }
    
    # 全指標の可視化（全top_k値を1つのグラフに）
    print(f"\n=== NSP Graph QPP指標の可視化 (全top_k値) ===")
    visualize_nsp_graph_metrics(
        csv_path=csv_path,
        metric_configs=metric_configs,
        merge_config=merge_config,
        output_dir=output_dir,
        top_k_values=TOP_K_VALUES
    )
    
    # AUC算出とROC曲線の可視化（全top_k値を1つのグラフに）
    print(f"\n=== NSP Graph AUC算出とROC曲線の可視化 (全top_k値) ===")
    roc_output_path = f'{output_dir}/roc_curves_nsp_graph_{split}.png'
    auc_scores = plot_roc_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=roc_output_path,
        positive_class='clarification',
        merge_config=merge_config,
        metric_column_map=metric_column_map
    )
    
    # PR曲線の可視化（全top_k値を1つのグラフに）
    print(f"\n=== NSP Graph Average Precision算出とPR曲線の可視化 (全top_k値) ===")
    pr_output_path = f'{output_dir}/pr_curves_nsp_graph_{split}.png'
    ap_scores = plot_pr_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=pr_output_path,
        positive_class='clarification',
        merge_config=merge_config,
        metric_column_map=metric_column_map
    )


def evaluate_neural_qpp(split: str = 'train', dataset: str = 'INSCIT'):
    """Neural QPP指標の評価"""
    # ベースパスの設定
    base_dir = '/home/daiki_shibata/pj/QPP4SIP'
    
    # 入力ファイルの設定
    csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, f'{split}.csv')
    output_dir = os.path.join(base_dir, 'QPP', 'evaluate', 'outputs', dataset)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # メトリクス設定（Neural QPPの出力ファイル）
    # CSVファイルはモデルディレクトリ（{model_type}_{metric}）内に保存される
    neural_qpp_outputs_dir = os.path.join(base_dir, 'QPP', 'neural_qpp', 'outputs', dataset)
    metric_configs = {}
    
    # bi-encoderとcross-encoderの両方をチェック
    for model_type in ['bi', 'cross']:
        for metric in ['map@20', 'map']:
            model_dir = os.path.join(neural_qpp_outputs_dir, f'{model_type}_{metric}')
            csv_file = os.path.join(model_dir, f'{split}_bertqpp_{model_type}_{metric}.csv')
            if os.path.exists(csv_file):
                metric_name = f'bertqpp_{model_type}_{metric}'
                metric_configs[metric_name] = {
                    'csv_path': csv_file,
                    'column': f'bertqpp_{model_type}'
                }
    
    if not metric_configs:
        print(f"警告: Neural QPPの出力ファイルが見つかりません。推論を実行してください。")
        print(f"  推論コマンド例: python QPP/neural_qpp/inference.py --dataset {dataset} --split {split} --model_type bi --metric map@20")
        return
    
    # マージ設定
    merge_config = {
        'left_on': ['dialogue_id', 'turn_id'],
        'right_on': ['conv_id', 'turn_id'],
        'how': 'inner'
    }
    
    # 全指標の可視化
    print("=== Neural QPP指標の可視化 ===")
    visualize_neural_qpp_metrics(
        csv_path=csv_path,
        metric_configs=metric_configs,
        merge_config=merge_config,
        output_dir=output_dir
    )
    
    # AUC算出とROC曲線の可視化
    print("\n=== Neural QPP AUC算出とROC曲線の可視化 ===")
    selected_metrics = None  # 全てのメトリクスを表示
    
    if selected_metrics is None:
        metric_csv_paths = {
            name: config['csv_path'] 
            for name, config in metric_configs.items()
        }
    else:
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
    
    # metric_csv_pathsとmetric_column_mapを作成
    metric_column_map = {
        name: config['column']
        for name, config in metric_configs.items()
    }
    
    roc_output_path = f'{output_dir}/roc_curves_neural_qpp_{split}.png'
    auc_scores = plot_roc_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=roc_output_path,
        positive_class='clarification',
        merge_config=merge_config,
        metric_column_map=metric_column_map
    )
    
    # PR曲線の可視化
    print("\n=== Neural QPP Average Precision算出とPR曲線の可視化 ===")
    pr_output_path = f'{output_dir}/pr_curves_neural_qpp_{split}.png'
    ap_scores = plot_pr_curves(
        csv_path=csv_path,
        metric_csv_paths=metric_csv_paths,
        output_path=pr_output_path,
        positive_class='clarification',
        merge_config=merge_config,
        metric_column_map=metric_column_map
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
        choices=['post_retrieval', 'retrieval_data', 'neural_qpp', 'nsp_graph', 'all'],
        default='post_retrieval',
        help='評価モード: post_retrieval (Post-retrieval指標), retrieval_data (Retrieval data指標), neural_qpp (Neural QPP指標), nsp_graph (NSP Graph指標), all (全て)'
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
    
    if args.mode == 'neural_qpp' or args.mode == 'all':
        evaluate_neural_qpp(split=args.split, dataset=args.dataset)
    
    if args.mode == 'nsp_graph' or args.mode == 'all':
        evaluate_nsp_graph(split=args.split, dataset=args.dataset)


if __name__ == '__main__':
    main()
