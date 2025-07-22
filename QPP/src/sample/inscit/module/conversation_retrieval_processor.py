import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .lucene_retriever import LuceneRetriever
from .retrieval_evaluator import RetrievalEvaluator


class ConversationRetrievalProcessor:
    """
    会話データに対して検索を適用し、評価指標を計算するクラス
    """
    
    def __init__(self, retriever: LuceneRetriever):
        """
        会話検索プロセッサーを初期化
        
        Args:
            retriever: 検索エンジンのインスタンス
        """
        self.retriever = retriever
        self.evaluator = RetrievalEvaluator()
    
    def _create_new_turn_structure(self, turn: Dict[str, Any], search_results: List[Dict[str, Any]], 
                                 query_key: str) -> Dict[str, Any]:
        """
        新しいturnの構造を作成
        
        Args:
            turn: 元のturnデータ
            search_results: 検索結果
            query_key: クエリキー名
            
        Returns:
            新しいturnの構造
        """
        new_turn = {}
        
        # クエリ関連の情報を最初に配置
        if 'query' in turn:
            new_turn['query'] = turn['query']
        if query_key in turn:
            new_turn['resolvedQuery'] = turn[query_key]
        
        # 検索結果を配置
        new_turn['retrieved_evidence'] = search_results
        
        # labelsを最後に配置
        if 'labels' in turn:
            new_turn['labels'] = turn['labels']
        
        return new_turn
    
    def _calculate_metrics_for_labels(self, labels: List[Dict[str, Any]], 
                                    search_results: List[Dict[str, Any]], 
                                    k_values: List[int]) -> None:
        """
        各labelに対して評価指標を計算して追加
        
        Args:
            labels: ラベルのリスト
            search_results: 検索結果
            k_values: 評価するk値のリスト
        """
        for label in labels:
            if 'evidence' not in label:
                continue
                
            gold_evidence = label['evidence']
            # 全ての評価指標を一度に計算
            metrics = self.evaluator.calculate_all_metrics(
                search_results, gold_evidence, k_values
            )
            
            # 各評価指標をlabelに追加
            for metric_name, score in metrics.items():
                label[metric_name] = score
    
    def _calculate_ndcg_for_labels(self, labels: List[Dict[str, Any]], 
                                 search_results: List[Dict[str, Any]], 
                                 k_values: List[int]) -> None:
        """
        各labelに対してnDCGを計算して追加（後方互換性のため残す）
        
        Args:
            labels: ラベルのリスト
            search_results: 検索結果
            k_values: 評価するk値のリスト
        """
        for label in labels:
            if 'evidence' not in label:
                continue
                
            gold_evidence = label['evidence']
            ndcg_scores = self.evaluator.calculate_ndcg_multiple_k(
                search_results, gold_evidence, k_values
            )
            
            # 各k値のnDCGスコアをlabelに追加
            for k, score in ndcg_scores.items():
                label[k] = score
    
    def _preprocess_query(self, query: str) -> str:
        """
        クエリの前処理
        
        Args:
            query: 元のクエリ
            
        Returns:
            前処理されたクエリ
        """
        # 基本的な前処理
        processed_query = query.strip()
        
        # 質問詞を除去してキーワードを抽出
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'can', 'could', 'would', 'will', 'do', 'does', 'did', 'is', 'are', 'was', 'were']
        
        # 文を単語に分割
        words = processed_query.lower().split()
        
        # 質問詞と一般的な単語を除去
        filtered_words = []
        for word in words:
            # 質問詞を除去
            if word in question_words:
                continue
            # 短すぎる単語を除去（2文字以下）
            if len(word) <= 2:
                continue
            # 句読点を除去
            word = word.strip('.,!?;:')
            if word:
                filtered_words.append(word)
        
        # キーワードのみのクエリを作成
        keyword_query = ' '.join(filtered_words)
        
        # キーワードが少なすぎる場合は元のクエリを使用
        if len(filtered_words) < 2:
            return processed_query
        
        return keyword_query
    
    def _process_single_turn(self, turn: Dict[str, Any], query_key: str, 
                           k_values: List[int], calculate_ndcg: bool) -> Dict[str, Any]:
        """
        単一のturnを処理
        
        Args:
            turn: 処理するturn
            query_key: クエリキー名
            k_values: 評価するk値のリスト
            calculate_ndcg: nDCGを計算するかどうか
            
        Returns:
            処理済みのturn
        """
        if query_key not in turn:
            return turn
        
        # 検索を実行
        query = turn.get(query_key, '')
        processed_query = self._preprocess_query(query)
        
        # 検索結果を取得
        search_results = self.retriever.search(processed_query, top_k=10)
        
        # 新しいturnの構造を作成
        new_turn = self._create_new_turn_structure(turn, search_results, query_key)
        
        # nDCGを計算して追加
        if calculate_ndcg and 'labels' in new_turn:
            self._calculate_metrics_for_labels(new_turn['labels'], search_results, k_values)
        
        return new_turn
    
    def _process_turns(self, target_data: Dict[str, Any], k_values: List[int], 
                      calculate_ndcg: bool) -> None:
        """
        turnsリストを処理
        
        Args:
            target_data: 対象データ
            k_values: 評価するk値のリスト
            calculate_ndcg: nDCGを計算するかどうか
        """
        if 'turns' not in target_data:
            return
        
        # resolvedQueryキーを使用
        query_key = 'resolvedQuery'
        
        for i, turn in enumerate(target_data['turns']):
            processed_turn = self._process_single_turn(turn, query_key, k_values, calculate_ndcg)
            target_data['turns'][i] = processed_turn
    
    def process_json_file(self, input_file_path: str, output_dir_path: Optional[str] = None, 
                         calculate_ndcg: bool = True, k_values: List[int] = [1, 3, 5]) -> str:
        """
        JSONファイルを処理して検索結果を追加
        
        Args:
            input_file_path: 入力JSONファイルのパス
            output_dir_path: 出力ディレクトリのパス（Noneの場合は入力ファイルと同じディレクトリ）
            calculate_ndcg: nDCGを計算するかどうか
            k_values: nDCG計算時のk値のリスト
            
        Returns:
            出力ファイルのパス
        """
        # 出力ディレクトリの設定
        if output_dir_path is None:
            output_dir_path = str(Path(input_file_path).parent)
        
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir_path, exist_ok=True)
        
        # JSONファイルを読み込み
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 全要素を処理
        if not isinstance(data, dict):
            raise ValueError("JSONファイルの構造が不正です")
        
        for top_key, target_data in data.items():
            self._process_turns(target_data, k_values, calculate_ndcg)
        
        # 出力ファイル名を生成
        input_filename = Path(input_file_path).stem
        output_filename = f"{input_filename}_retrieved.json"
        output_file_path = os.path.join(output_dir_path, output_filename)
        
        # 処理結果を保存
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return output_file_path
    
    def _collect_metrics_scores(self, json_data: Dict[str, Any], k_values: List[int]) -> Dict[str, List[float]]:
        """
        JSONデータから評価指標のスコアを収集
        
        Args:
            json_data: 処理済みのJSONデータ
            k_values: 評価するk値のリスト
            
        Returns:
            各評価指標でのスコアのリスト
        """
        # 評価指標の種類を定義
        metric_types = ['ndcg', 'precision', 'recall']
        metrics_by_type = {}
        
        for metric_type in metric_types:
            for k in k_values:
                metrics_by_type[f'{metric_type}@{k}'] = []
        
        for top_key, target_data in json_data.items():
            if 'turns' not in target_data:
                continue
                
            for turn in target_data['turns']:
                if 'labels' not in turn:
                    continue
                    
                for label in turn['labels']:
                    for metric_name in metrics_by_type.keys():
                        if metric_name in label:
                            metrics_by_type[metric_name].append(label[metric_name])
        
        return metrics_by_type
    
    def _collect_ndcg_scores(self, json_data: Dict[str, Any], k_values: List[int]) -> Dict[str, List[float]]:
        """
        JSONデータからnDCGスコアを収集（後方互換性のため残す）
        
        Args:
            json_data: 処理済みのJSONデータ
            k_values: 評価するk値のリスト
            
        Returns:
            各k値でのnDCGスコアのリスト
        """
        ndcg_scores_by_k = {f'ndcg@{k}': [] for k in k_values}
        
        for top_key, target_data in json_data.items():
            if 'turns' not in target_data:
                continue
                
            for turn in target_data['turns']:
                if 'labels' not in turn:
                    continue
                    
                for label in turn['labels']:
                    for k in k_values:
                        k_key = f'ndcg@{k}'
                        if k_key in label:
                            ndcg_scores_by_k[k_key].append(label[k_key])
        
        return ndcg_scores_by_k
    
    def _collect_metrics_by_response_type(self, json_data: Dict[str, Any], k_values: List[int]) -> Dict[str, Dict[str, List[float]]]:
        """
        responseType別に評価指標のスコアを収集
        
        Args:
            json_data: 処理済みのJSONデータ
            k_values: 評価するk値のリスト
            
        Returns:
            responseType別の評価指標スコア
        """
        # 評価指標の種類を定義
        metric_types = ['ndcg', 'precision', 'recall']
        
        # responseType別の辞書を初期化
        metrics_by_response_type = {}
        
        for top_key, target_data in json_data.items():
            if 'turns' not in target_data:
                continue
                
            for turn in target_data['turns']:
                if 'labels' not in turn:
                    continue
                    
                for label in turn['labels']:
                    response_type = label.get('responseType', 'unknown')
                    
                    # responseTypeが初回の場合は初期化
                    if response_type not in metrics_by_response_type:
                        metrics_by_response_type[response_type] = {}
                        for metric_type in metric_types:
                            for k in k_values:
                                metrics_by_response_type[response_type][f'{metric_type}@{k}'] = []
                    
                    # 各評価指標を収集
                    for metric_type in metric_types:
                        for k in k_values:
                            metric_name = f'{metric_type}@{k}'
                            if metric_name in label:
                                metrics_by_response_type[response_type][metric_name].append(label[metric_name])
        
        return metrics_by_response_type
    
    def calculate_metrics_by_response_type(self, json_data: Dict[str, Any], k_values: List[int] = [1, 3, 5]) -> Dict[str, Dict[str, float]]:
        """
        responseType別の評価指標を算出
        
        Args:
            json_data: 処理済みのJSONデータ
            k_values: 評価するk値のリスト
            
        Returns:
            responseType別の評価指標
        """
        if not isinstance(json_data, dict):
            raise ValueError("JSONデータの構造が不正です")
        
        # responseType別にスコアを収集
        metrics_by_response_type = self._collect_metrics_by_response_type(json_data, k_values)
        
        # 各responseTypeでの統計を計算
        results = {}
        for response_type, metrics in metrics_by_response_type.items():
            results[response_type] = {}
            
            # 各評価指標の統計を計算
            for metric_name, scores in metrics.items():
                if scores:
                    results[response_type][f'avg_{metric_name}'] = np.mean(scores)
                    results[response_type][f'max_{metric_name}'] = np.max(scores)
                    results[response_type][f'min_{metric_name}'] = np.min(scores)
                    results[response_type][f'std_{metric_name}'] = np.std(scores)
                    results[response_type][f'count_{metric_name}'] = len(scores)
                else:
                    results[response_type][f'avg_{metric_name}'] = 0.0
                    results[response_type][f'max_{metric_name}'] = 0.0
                    results[response_type][f'min_{metric_name}'] = 0.0
                    results[response_type][f'std_{metric_name}'] = 0.0
                    results[response_type][f'count_{metric_name}'] = 0
        
        return results
    
    def calculate_overall_metrics(self, json_data: Dict[str, Any], k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        JSONデータ全体の評価指標を算出
        
        Args:
            json_data: 処理済みのJSONデータ
            k_values: 評価するk値のリスト
            
        Returns:
            全体の評価指標
        """
        if not isinstance(json_data, dict):
            raise ValueError("JSONデータの構造が不正です")
        
        # 評価指標スコアを収集
        metrics_scores_by_type = self._collect_metrics_scores(json_data, k_values)
        
        # 各k値での平均評価指標と最大値を計算
        overall_metrics = {}
        for metric_name in metrics_scores_by_type.keys():
            scores = metrics_scores_by_type[metric_name]
            if scores:
                overall_metrics[f'avg_{metric_name}'] = np.mean(scores)
                overall_metrics[f'max_{metric_name}'] = np.max(scores)
            else:
                overall_metrics[f'avg_{metric_name}'] = 0.0
                overall_metrics[f'max_{metric_name}'] = 0.0
        
        # 総クエリ数を追加
        total_queries = sum(len(scores) for scores in metrics_scores_by_type.values())
        overall_metrics['num_queries'] = total_queries
        
        return overall_metrics 