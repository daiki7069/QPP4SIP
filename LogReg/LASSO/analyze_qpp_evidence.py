"""
QPPの証拠を分析するスクリプト
1. 3つのグループ（A/B/C）でのQPPスコア比較
2. BERTの確信度分析
3. 検索結果の矛盾度チェック
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# scipy.special.expit (sigmoid関数) を使用するため
try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_json_data(json_path: Path) -> List[List[Dict]]:
    """JSONデータを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def classify_queries(results_df: pd.DataFrame, base_model: str) -> Dict[str, pd.DataFrame]:
    """
    クエリを3つのグループに分類
    
    A群: BERTもLogRegも正解したクエリ（簡単）
    B群: BERTは間違えたが、LogRegは正解したクエリ（今回のターゲット）
    C群: 両方とも間違えたクエリ（難問）
    """
    # 予測を計算
    results_df['LogReg_pred'] = (results_df['LogReg'] >= 0.5).astype(int)
    results_df[f'{base_model}_pred'] = (results_df[base_model] > 0.0).astype(int)
    
    # 正解/不正解を判定
    logreg_correct = results_df['LogReg_pred'] == results_df['label']
    base_correct = results_df[f'{base_model}_pred'] == results_df['label']
    
    # グループ分類
    group_a = results_df[logreg_correct & base_correct].copy()  # 両方正解
    group_b = results_df[logreg_correct & ~base_correct].copy()  # LogReg正解、ベースモデル不正解
    group_c = results_df[~logreg_correct & ~base_correct].copy()  # 両方不正解
    
    # グループD: LogReg不正解、ベースモデル正解（参考用）
    group_d = results_df[~logreg_correct & base_correct].copy()
    
    return {
        'A (Both correct)': group_a,
        'B (LogReg wins)': group_b,
        'C (Both wrong)': group_c,
        'D (Base wins)': group_d
    }


def create_detailed_case_studies(
    conflict_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    top_n: int = 10
):
    """詳細なケーススタディを作成（論文用の具体例）"""
    # スコア差が大きいケースを選ぶ
    conflict_df = conflict_df.copy()
    conflict_df['score_diff'] = conflict_df['LogReg_score'] - conflict_df[f'{base_model}_confidence']
    
    # 上位N件を選ぶ（スコア差が大きい順）
    top_cases = conflict_df.nlargest(top_n, 'score_diff')
    
    # テキストファイルに保存
    case_study_path = output_dir / f'detailed_case_studies_{base_model.lower()}.txt'
    with open(case_study_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Detailed Case Studies: LogReg Wins (Top {top_n} by Score Difference)\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, (_, case) in enumerate(top_cases.iterrows(), 1):
            f.write(f"Case {idx}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Query: {case['query']}\n")
            f.write(f"Answer: {case['answer']}\n")
            f.write(f"\nPredictions:\n")
            f.write(f"  {base_model} confidence: {case[f'{base_model}_confidence']:.4f} (predicted: {'clarification' if case[f'{base_model}_confidence'] > 0.5 else 'answer'})\n")
            f.write(f"  LogReg score: {case['LogReg_score']:.4f} (predicted: {'clarification' if case['LogReg_score'] >= 0.5 else 'answer'})\n")
            f.write(f"  Score difference: {case['score_diff']:.4f}\n")
            f.write(f"\nQPP Features:\n")
            for feat in ['WIG', 'NQC', 'SMV', 'Clarity', 'nSigma']:
                if feat in case and pd.notna(case[feat]):
                    f.write(f"  {feat}: {case[feat]:.4f}\n")
            f.write(f"\nRetrieval Results (Top 5):\n")
            if case['retrieval_titles']:
                titles = case['retrieval_titles'].split(' | ')
                for i, title in enumerate(titles[:5], 1):
                    f.write(f"  {i}. {title}\n")
            else:
                f.write("  (No retrieval titles available)\n")
            f.write(f"\nRetrieval Statistics:\n")
            f.write(f"  Diversity: {case['diversity']:.3f}\n")
            f.write(f"  Score std: {case['score_std']:.3f}\n")
            f.write(f"  Score range: {case['score_range']:.3f}\n")
            f.write(f"  Has answer in top 10: {case['has_answer_in_topk']}\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"詳細ケーススタディを保存: {case_study_path}")


def plot_qpp_comparison(
    groups: Dict[str, pd.DataFrame],
    output_dir: Path,
    base_model: str
):
    """QPPスコアの比較を可視化"""
    # QPP特徴量を取得
    qpp_features = ['WIG', 'NQC', 'SMV', 'Clarity', 'nSigma', 'AvgIDF', 'MaxIDF', 'AvgICTF', 'MaxSCQ', 'SClarity']
    
    # データを準備
    plot_data = []
    for group_name, group_df in groups.items():
        for feature in qpp_features:
            if feature in group_df.columns:
                for val in group_df[feature].dropna():
                    plot_data.append({
                        'Group': group_name,
                        'Feature': feature,
                        'Value': val
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # 各特徴量についてボックスプロットを作成
    n_features = len(qpp_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(qpp_features):
        if feature not in plot_df['Feature'].values:
            continue
            
        ax = axes[idx]
        feature_data = plot_df[plot_df['Feature'] == feature]
        
        # ボックスプロット
        sns.boxplot(data=feature_data, x='Group', y='Value', ax=ax, order=['A (Both correct)', 'B (LogReg wins)', 'C (Both wrong)', 'D (Base wins)'])
        
        # 平均値を表示
        for i, group_name in enumerate(['A (Both correct)', 'B (LogReg wins)', 'C (Both wrong)', 'D (Base wins)']):
            group_data = feature_data[feature_data['Group'] == group_name]['Value']
            if len(group_data) > 0:
                mean_val = group_data.mean()
                ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'qpp_comparison_by_group_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QPP比較チャートを保存: {output_path}")


def analyze_bert_confidence(
    results_df: pd.DataFrame,
    groups: Dict[str, pd.DataFrame],
    output_dir: Path,
    base_model: str
):
    """BERTの確信度を分析"""
    # ベースモデルのlogit値を確率に変換
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 各グループの確信度分布を可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    group_names = ['A (Both correct)', 'B (LogReg wins)', 'C (Both wrong)', 'D (Base wins)']
    
    for idx, group_name in enumerate(group_names):
        ax = axes[idx]
        if group_name in groups and len(groups[group_name]) > 0:
            group_df = groups[group_name]
            group_df = group_df.copy()
            group_df[f'{base_model}_proba'] = expit(group_df[base_model])
            
            # ヒストグラム
            ax.hist(group_df[f'{base_model}_proba'], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
            ax.axvline(group_df[f'{base_model}_proba'].mean(), color='blue', linestyle='--', linewidth=2,
                      label=f'Mean: {group_df[f"{base_model}_proba"].mean():.3f}')
            
            # 統計情報を表示
            mean_conf = group_df[f'{base_model}_proba'].mean()
            std_conf = group_df[f'{base_model}_proba'].std()
            median_conf = group_df[f'{base_model}_proba'].median()
            
            # 迷っているケース（0.4-0.6）の割合
            uncertain = ((group_df[f'{base_model}_proba'] >= 0.4) & (group_df[f'{base_model}_proba'] <= 0.6)).sum()
            uncertain_pct = uncertain / len(group_df) * 100
            
            ax.set_title(f'{group_name}\n(n={len(group_df)}, uncertain: {uncertain_pct:.1f}%)', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel(f'{base_model} Confidence (probability)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            
            # テキストで統計を表示
            stats_text = f'Mean: {mean_conf:.3f}\nStd: {std_conf:.3f}\nMedian: {median_conf:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f'bert_confidence_analysis_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BERT確信度分析を保存: {output_path}")
    
    # 統計サマリーをCSVに保存
    summary_data = []
    for group_name in group_names:
        if group_name in groups and len(groups[group_name]) > 0:
            group_df = groups[group_name].copy()
            group_df[f'{base_model}_proba'] = expit(group_df[base_model])
            
            summary_data.append({
                'Group': group_name,
                'Count': len(group_df),
                'Mean_confidence': group_df[f'{base_model}_proba'].mean(),
                'Std_confidence': group_df[f'{base_model}_proba'].std(),
                'Median_confidence': group_df[f'{base_model}_proba'].median(),
                'Uncertain_count': ((group_df[f'{base_model}_proba'] >= 0.4) & (group_df[f'{base_model}_proba'] <= 0.6)).sum(),
                'Uncertain_pct': ((group_df[f'{base_model}_proba'] >= 0.4) & (group_df[f'{base_model}_proba'] <= 0.6)).sum() / len(group_df) * 100,
                'High_confidence_wrong': ((group_df[f'{base_model}_proba'] < 0.3) & (group_df['label'] == 1)).sum() if 'label' in group_df.columns else 0,
                'Low_confidence_correct': ((group_df[f'{base_model}_proba'] > 0.7) & (group_df['label'] == 0)).sum() if 'label' in group_df.columns else 0
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f'bert_confidence_summary_{base_model.lower()}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"BERT確信度サマリーを保存: {summary_path}")


def analyze_retrieval_conflicts(
    results_df: pd.DataFrame,
    dataset_json_path: Path,
    groups: Dict[str, pd.DataFrame],
    output_dir: Path,
    base_model: str,
    dpr_json_path: Path = None,
    top_k: int = 10
):
    """検索結果の矛盾度を分析"""
    # DPR結果ファイルを読み込む（検索結果が含まれている）
    if dpr_json_path is None:
        # デフォルトパスを試す
        dpr_json_path = dataset_json_path.parent / "dpr_dev.json"
    
    if not dpr_json_path.exists():
        print(f"警告: DPR結果ファイルが見つかりません: {dpr_json_path}")
        print("検索結果分析をスキップします。")
        return
    
    dpr_data = load_json_data(dpr_json_path)
    print(f"DPR結果を読み込み: {len(dpr_data)} 会話")
    
    # グループB（LogReg wins）の詳細分析
    if 'B (LogReg wins)' not in groups or len(groups['B (LogReg wins)']) == 0:
        print("グループBが見つかりません。検索結果分析をスキップします。")
        return
    
    group_b = groups['B (LogReg wins)']
    
    # DPRデータをインデックス化（高速検索のため）
    dpr_index = {}
    for conversation in dpr_data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 0))
            key = (conv_id, turn_id)
            dpr_index[key] = turn
    
    # 検索結果の情報を収集
    conflict_analysis = []
    
    # results.csvの行インデックスとデータセットの対応を確認
    # results.csvの行順序がデータセットの順序と一致していることを前提とする
    flat_idx = 0
    
    # results.csvの行インデックスを保存（DataFrameのインデックスを使用）
    results_df_with_index = results_df.reset_index()
    results_df_with_index['csv_row_idx'] = results_df_with_index.index
    
    # グループBの行インデックスを取得
    group_b_indices = group_b.index.tolist()
    
    for csv_row_idx in group_b_indices:
        row = group_b.loc[csv_row_idx]
        
        # データセットから該当するクエリを探す（行順序で対応）
        conv_id = None
        turn_id = None
        key = None
        
        # 行順序で対応を試す
        current_idx = 0
        for conversation in dpr_data:
            for turn in conversation:
                if current_idx == csv_row_idx:
                    conv_id = str(turn.get('conv_id', ''))
                    turn_id = int(turn.get('turn_id', 0))
                    key = (conv_id, turn_id)
                    break
                current_idx += 1
            if key is not None:
                break
        
        # クエリ情報を取得（データセットから）
        query_text = ''
        answer = ''
        if key is not None and key in dpr_index:
            turn = dpr_index[key]
            query_text = turn.get('query', '')
            answer = turn.get('answer', [])
            if isinstance(answer, list):
                answer = ', '.join(answer) if answer else ''
            else:
                answer = str(answer)
        
        base_proba = expit(row[base_model]) if base_model in row else 0.0
        logreg_score = row['LogReg']
        
        # 検索結果を取得
        retrieval_titles = []
        retrieval_texts = []
        retrieval_scores = []
        has_answer_in_topk = False
        
        if key is not None and key in dpr_index:
            turn = dpr_index[key]
            ctxs = turn.get('ctxs', [])[:top_k]
            for ctx in ctxs:
                title = ctx.get('title', '')
                text = ctx.get('text', '')
                score_str = ctx.get('score', '0.0')
                try:
                    score = float(score_str)
                except (ValueError, TypeError):
                    score = 0.0
                
                retrieval_titles.append(title)
                retrieval_texts.append(text[:100] if text else '')  # 最初の100文字
                retrieval_scores.append(score)
                if ctx.get('has_answer', False):
                    has_answer_in_topk = True
        
        # 簡易的な矛盾検出（タイトルから）
        num_titles = len(retrieval_titles)
        unique_titles = len(set(retrieval_titles))
        diversity = unique_titles / num_titles if num_titles > 0 else 0
        
        # スコアの分散（検索結果の質のばらつき）
        score_std = np.std(retrieval_scores) if len(retrieval_scores) > 0 else 0.0
        score_range = max(retrieval_scores) - min(retrieval_scores) if len(retrieval_scores) > 0 else 0.0
        
        # QPP特徴量を取得
        qpp_features = {}
        for feat in ['WIG', 'NQC', 'SMV', 'Clarity', 'nSigma', 'AvgIDF', 'MaxIDF', 'AvgICTF', 'MaxSCQ', 'SClarity']:
            if feat in row:
                qpp_features[feat] = row[feat]
        
        conflict_analysis.append({
            'csv_row_index': int(csv_row_idx),
            'conv_id': conv_id if conv_id else '',
            'turn_id': turn_id if turn_id else 0,
            'query': query_text,
            'answer': str(answer),
            f'{base_model}_confidence': base_proba,
            'LogReg_score': logreg_score,
            'num_retrieval': num_titles,
            'unique_titles': unique_titles,
            'diversity': diversity,
            'has_answer_in_topk': has_answer_in_topk,
            'score_std': score_std,
            'score_range': score_range,
            'retrieval_titles': ' | '.join(retrieval_titles[:5]) if retrieval_titles else '',  # 上位5件
            'retrieval_texts_preview': ' | '.join([t[:50] for t in retrieval_texts[:3]]) if retrieval_texts else '',  # 上位3件のテキスト
            **qpp_features  # QPP特徴量を追加
        })
    
    conflict_df = pd.DataFrame(conflict_analysis)
    
    # CSVに保存
    conflict_path = output_dir / f'retrieval_conflict_analysis_{base_model.lower()}.csv'
    conflict_df.to_csv(conflict_path, index=False, encoding='utf-8')
    print(f"検索結果矛盾分析を保存: {conflict_path}")
    
    # 詳細なケーススタディを作成
    create_detailed_case_studies(conflict_df, output_dir, base_model, top_n=10)
    
    # 統計を表示
    print(f"\n=== 検索結果分析サマリー（グループB: LogReg wins） ===")
    print(f"総クエリ数: {len(conflict_df)}")
    print(f"平均検索結果数: {conflict_df['num_retrieval'].mean():.1f}")
    print(f"平均多様性: {conflict_df['diversity'].mean():.3f}")
    print(f"平均スコア標準偏差: {conflict_df['score_std'].mean():.3f}")
    print(f"平均スコア範囲: {conflict_df['score_range'].mean():.3f}")
    print(f"上位{top_k}件に答えが含まれている割合: {conflict_df['has_answer_in_topk'].sum() / len(conflict_df) * 100:.1f}%")
    
    # QPP特徴量の統計
    qpp_features = ['WIG', 'NQC', 'SMV', 'Clarity', 'nSigma']
    print(f"\n=== QPP特徴量の統計（グループB） ===")
    for feat in qpp_features:
        if feat in conflict_df.columns:
            print(f"{feat:10s}: 平均={conflict_df[feat].mean():8.4f}, 中央値={conflict_df[feat].median():8.4f}, 最小={conflict_df[feat].min():8.4f}, 最大={conflict_df[feat].max():8.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="QPPの証拠を分析"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="results.csvのパス"
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        required=True,
        help="データセットのJSONファイル（dev.jsonなど）のパス"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="ベースモデル名（BERT/RoBERTa/Transfer、Noneの場合は自動検出）"
    )
    parser.add_argument(
        "--analysis-csv",
        type=str,
        default=None,
        help="分析結果CSV（logreg_correct_*_wrong.csv、オプション）"
    )
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    dataset_json_path = Path(args.dataset_json)
    output_dir = Path(args.output_dir)
    
    if not results_csv_path.exists():
        print(f"エラー: results.csvが見つかりません: {results_csv_path}")
        return
    
    if not dataset_json_path.exists():
        print(f"エラー: データセットJSONが見つかりません: {dataset_json_path}")
        return
    
    # ベースモデルを自動検出
    base_model = args.base_model
    if base_model is None:
        results_df = pd.read_csv(results_csv_path, nrows=1)
        if 'RoBERTa' in results_df.columns:
            base_model = 'RoBERTa'
        elif 'BERT' in results_df.columns:
            base_model = 'BERT'
        elif 'Transfer' in results_df.columns:
            base_model = 'Transfer'
        else:
            print("エラー: ベースモデルを自動検出できませんでした。--base-modelを指定してください。")
            return
    
    print(f"=== QPPの証拠分析 ===")
    print(f"ベースモデル: {base_model}")
    print()
    
    # データを読み込む
    results_df = pd.read_csv(results_csv_path)
    print(f"results.csvを読み込み: {len(results_df)} 行")
    
    # 分析結果CSVが指定されている場合、それも読み込む
    analysis_df = None
    group_b_from_analysis = None
    if args.analysis_csv and Path(args.analysis_csv).exists():
        analysis_df = pd.read_csv(args.analysis_csv)
        print(f"分析結果CSVを読み込み: {len(analysis_df)} 行")
        # 分析結果CSVのインデックスを使って、results_dfから該当行を抽出
        if 'index' in analysis_df.columns:
            # 'index'カラムがresults.csvの行インデックス
            analysis_indices = analysis_df['index'].astype(int).tolist()
            # グループBとして、分析結果CSVに含まれる行を使用
            group_b_from_analysis = results_df.iloc[analysis_indices].copy()
            print(f"分析結果CSVからグループBを抽出: {len(group_b_from_analysis)} 行")
        elif 'csv_row_index' in analysis_df.columns:
            analysis_indices = analysis_df['csv_row_index'].astype(int).tolist()
            group_b_from_analysis = results_df.iloc[analysis_indices].copy()
            print(f"分析結果CSVからグループBを抽出: {len(group_b_from_analysis)} 行")
    
    # クエリをグループに分類
    groups = classify_queries(results_df, base_model)
    
    # 分析結果CSVが指定されている場合、グループBを置き換え
    if group_b_from_analysis is not None:
        groups['B (LogReg wins)'] = group_b_from_analysis
        print(f"グループBを分析結果CSVから抽出したデータに置き換えました（{len(group_b_from_analysis)} 行）")
    
    print("=== グループ分類結果 ===")
    for group_name, group_df in groups.items():
        print(f"{group_name}: {len(group_df)} 件")
    print()
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. QPPスコアの比較
    print("1. QPPスコアの比較を生成中...")
    plot_qpp_comparison(groups, output_dir, base_model)
    
    # 2. BERTの確信度分析
    print("2. BERTの確信度分析を生成中...")
    analyze_bert_confidence(results_df, groups, output_dir, base_model)
    
    # 3. 検索結果の矛盾度分析
    print("3. 検索結果の矛盾度分析を生成中...")
    dpr_json_path = dataset_json_path.parent / "dpr_dev.json"
    analyze_retrieval_conflicts(results_df, dataset_json_path, groups, output_dir, base_model, dpr_json_path)
    
    print()
    print("=== 分析完了 ===")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()

