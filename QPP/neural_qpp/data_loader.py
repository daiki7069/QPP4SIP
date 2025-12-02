"""
Neural QPP用のデータローダー
DPRの結果ファイルを読み込み、QPPスコア（MAP@20など）を計算
has_answerラベルを使用して関連文書を判断する
"""
from pathlib import Path
from typing import List
from dataclasses import dataclass
import importlib.util

# post_retrievalのdata_loaderを直接インポート（循環インポートを避けるため）
post_retrieval_path = Path(__file__).parent.parent / "post_retrieval" / "data_loader.py"
spec = importlib.util.spec_from_file_location("post_retrieval_data_loader", post_retrieval_path)
post_retrieval_data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(post_retrieval_data_loader)

DPRResultLoader = post_retrieval_data_loader.DPRResultLoader
TurnData = post_retrieval_data_loader.TurnData
RetrievedDocument = post_retrieval_data_loader.RetrievedDocument


@dataclass
class QPPTrainingData:
    """QPP学習用のデータ"""
    query: str
    doc_text: str
    qpp_score: float  # MAP@20など
    conv_id: str
    turn_id: str


def calculate_map_at_k(relevance_scores: List[float], k: int, total_relevant: int) -> float:
    """MAP@kを計算"""
    if not relevance_scores or total_relevant == 0:
        return 0.0
    
    # 最初のk個のみを使用
    scores = relevance_scores[:k]
    
    precision_sum = 0.0
    relevant_found = 0
    
    for i, score in enumerate(scores):
        if score > 0:  # 関連ドキュメントが見つかった
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def calculate_qpp_scores(
    turn_data: TurnData, 
    metric: str = "map@20"
) -> float:
    """
    ターンデータからQPPスコアを計算
    
    Args:
        turn_data: TurnDataオブジェクト
        metric: 計算するメトリクス（"map@20", "map"など）
    
    Returns:
        QPPスコア
    """
    documents = turn_data.documents
    if not documents:
        return 0.0
    
    # 関連性スコアを計算（doc.has_answerを使用）
    relevance_scores = [1.0 if doc.has_answer else 0.0 for doc in documents]
    total_relevant = sum(relevance_scores)  # has_answerがTrueの文書数
    
    if total_relevant == 0:
        return 0.0
    
    # メトリクスに応じて計算
    if metric.lower() == "map@20":
        return calculate_map_at_k(relevance_scores, k=20, total_relevant=total_relevant)
    elif metric.lower() == "map":
        return calculate_map_at_k(relevance_scores, k=len(relevance_scores), total_relevant=total_relevant)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class QPPDataLoader:
    """QPP学習用のデータローダー"""
    
    def __init__(self, dpr_json_path: str, base_json_path: str = None):
        """
        Args:
            dpr_json_path: DPR結果ファイルのパス（has_answerラベルを含む）
            base_json_path: ベースJSONファイルのパス（互換性のため残しているが、使用しない）
        """
        self.dpr_json_path = dpr_json_path
        self.dpr_loader = DPRResultLoader(dpr_json_path)
    
    def load_training_data(
        self, 
        metric: str = "map@20",
        use_first_doc_only: bool = True
    ) -> List[QPPTrainingData]:
        """
        学習用データを読み込む
        
        Args:
            metric: QPPスコアの計算メトリクス（"map@20"など）
            use_first_doc_only: Trueの場合、最初のドキュメントのみを使用
        
        Returns:
            QPPTrainingDataのリスト
        """
        # DPR結果を読み込み（has_answerラベルが含まれている）
        turn_data_list = self.dpr_loader.load_data()
        
        training_data = []
        
        for turn in turn_data_list:
            # QPPスコアを計算（has_answerラベルを使用）
            qpp_score = calculate_qpp_scores(turn, metric=metric)
            
            # 最初のドキュメントを取得
            if turn.documents:
                first_doc = turn.documents[0]
                
                training_data.append(QPPTrainingData(
                    query=turn.question,
                    doc_text=first_doc.text,
                    qpp_score=qpp_score,
                    conv_id=turn.conv_id,
                    turn_id=turn.turn_id
                ))
        
        return training_data

