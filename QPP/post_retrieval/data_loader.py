"""
Post-retrieval QPP用のデータローダー
DPRの結果ファイルを読み込み、分析用のデータを準備する
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    id: str
    title: str
    text: str
    score: float
    has_answer: bool
    rank: int


@dataclass
class TurnData:
    question: str
    answers: List[str]
    conv_id: str
    turn_id: str
    documents: List[RetrievedDocument]
    
    @property
    def scores(self) -> List[float]:
        return [doc.score for doc in self.documents]


class DPRResultLoader:
    def __init__(self, dpr_json_path: str):
        self.dpr_json_path = dpr_json_path
        self.data: Dict[str, List[TurnData]] = {}
    
    def load_data(self) -> List[TurnData]:
        file_path = self.dpr_json_path
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        turn_data_list = []
        
        # データ形式の判定: 会話のリストのリスト（INSCIT形式）か、フラットなリストか
        if isinstance(raw_data, list) and len(raw_data) > 0:
            # 最初の要素がリストかどうかで判定
            if isinstance(raw_data[0], list):
                # 会話のリストのリスト形式（新しい形式）
                for conversation in raw_data:
                    for turn in conversation:
                        documents = []
                        # ctxsが存在する場合のみ処理
                        if 'ctxs' in turn and isinstance(turn['ctxs'], list):
                            for rank, ctx in enumerate(turn['ctxs'], 1):
                                doc = RetrievedDocument(
                                    id=ctx['id'], title=ctx['title'], text=ctx['text'],
                                    score=float(ctx['score']), has_answer=ctx['has_answer'], rank=rank
                                )
                                documents.append(doc)
                        
                        # questionまたはqueryフィールドから質問を取得
                        question = turn.get('query', turn.get('question', ''))
                        # answersまたはanswerフィールドから回答を取得
                        answers = turn.get('answer', turn.get('answers', []))
                        if not isinstance(answers, list):
                            answers = [answers] if answers else []
                        
                        turn_data = TurnData(
                            question=question, answers=answers,
                            conv_id=str(turn['conv_id']), turn_id=str(turn['turn_id']), documents=documents
                        )
                        turn_data_list.append(turn_data)
            else:
                # フラットなリスト形式（古い形式）
                for item in raw_data:
                    documents = []
                    if 'ctxs' in item and isinstance(item['ctxs'], list):
                        for rank, ctx in enumerate(item['ctxs'], 1):
                            doc = RetrievedDocument(
                                id=ctx['id'], title=ctx['title'], text=ctx['text'],
                                score=float(ctx['score']), has_answer=ctx['has_answer'], rank=rank
                            )
                            documents.append(doc)
                    
                    # questionまたはqueryフィールドから質問を取得
                    question = item.get('query', item.get('question', ''))
                    # answersまたはanswerフィールドから回答を取得
                    answers = item.get('answer', item.get('answers', []))
                    if not isinstance(answers, list):
                        answers = [answers] if answers else []
                    
                    turn_data = TurnData(
                        question=question, answers=answers,
                        conv_id=str(item['conv_id']), turn_id=str(item.get('turn_id', '')), documents=documents
                    )
                    turn_data_list.append(turn_data)
        
        return turn_data_list
    
    def load_data_from_file(self, file_path: str) -> List[TurnData]:
        """ファイルパスを直接指定してデータを読み込む"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        turn_data_list = []
        
        # データ形式の判定: 会話のリストのリスト（INSCIT形式）か、フラットなリストか
        if isinstance(raw_data, list) and len(raw_data) > 0:
            # 最初の要素がリストかどうかで判定
            if isinstance(raw_data[0], list):
                # 会話のリストのリスト形式（新しい形式）
                for conversation in raw_data:
                    for turn in conversation:
                        documents = []
                        # ctxsが存在する場合のみ処理
                        if 'ctxs' in turn and isinstance(turn['ctxs'], list):
                            for rank, ctx in enumerate(turn['ctxs'], 1):
                                doc = RetrievedDocument(
                                    id=ctx['id'], title=ctx['title'], text=ctx['text'],
                                    score=float(ctx['score']), has_answer=ctx['has_answer'], rank=rank
                                )
                                documents.append(doc)
                        
                        # questionまたはqueryフィールドから質問を取得
                        question = turn.get('query', turn.get('question', ''))
                        # answersまたはanswerフィールドから回答を取得
                        answers = turn.get('answer', turn.get('answers', []))
                        if not isinstance(answers, list):
                            answers = [answers] if answers else []
                        
                        turn_data = TurnData(
                            question=question, answers=answers,
                            conv_id=str(turn['conv_id']), turn_id=str(turn['turn_id']), documents=documents
                        )
                        turn_data_list.append(turn_data)
            else:
                # フラットなリスト形式（古い形式）
                for item in raw_data:
                    documents = []
                    if 'ctxs' in item and isinstance(item['ctxs'], list):
                        for rank, ctx in enumerate(item['ctxs'], 1):
                            doc = RetrievedDocument(
                                id=ctx['id'], title=ctx['title'], text=ctx['text'],
                                score=float(ctx['score']), has_answer=ctx['has_answer'], rank=rank
                            )
                            documents.append(doc)
                    
                    # questionまたはqueryフィールドから質問を取得
                    question = item.get('query', item.get('question', ''))
                    # answersまたはanswerフィールドから回答を取得
                    answers = item.get('answer', item.get('answers', []))
                    if not isinstance(answers, list):
                        answers = [answers] if answers else []
                    
                    turn_data = TurnData(
                        question=question, answers=answers,
                        conv_id=str(item['conv_id']), turn_id=str(item.get('turn_id', '')), documents=documents
                    )
                    turn_data_list.append(turn_data)
        
        return turn_data_list


