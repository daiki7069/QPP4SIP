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
        for item in raw_data:
            documents = []
            for rank, ctx in enumerate(item['ctxs'], 1):
                doc = RetrievedDocument(
                    id=ctx['id'], title=ctx['title'], text=ctx['text'],
                    score=float(ctx['score']), has_answer=ctx['has_answer'], rank=rank
                )
                documents.append(doc)
            
            turn_data = TurnData(
                question=item['question'], answers=item['answers'],
                conv_id=item['conv_id'], turn_id=item['turn_id'], documents=documents
            )
            turn_data_list.append(turn_data)
        
        return turn_data_list
    
    def load_data_from_file(self, file_path: str) -> List[TurnData]:
        """ファイルパスを直接指定してデータを読み込む"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        turn_data_list = []
        for item in raw_data:
            documents = []
            for rank, ctx in enumerate(item['ctxs'], 1):
                doc = RetrievedDocument(
                    id=ctx['id'], title=ctx['title'], text=ctx['text'],
                    score=float(ctx['score']), has_answer=ctx['has_answer'], rank=rank
                )
                documents.append(doc)
            
            turn_data = TurnData(
                question=item['question'], answers=item['answers'],
                conv_id=item['conv_id'], turn_id=str(item['turn_id']), documents=documents
            )
            turn_data_list.append(turn_data)
        
        return turn_data_list


