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
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data: Dict[str, List[TurnData]] = {}
    
    def load_data(self, split: str) -> List[TurnData]:
        file_path = self.results_dir / f"dpr_{split}.json"
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
        
        self.data[split] = turn_data_list
        return turn_data_list


