"""
Post-retrieval QPP用の文書内容分析
"""
import pandas as pd
import numpy as np
from typing import List
from data_loader import TurnData


class ContentAnalyzer:
    def __init__(self, turn_data_list: List[TurnData]):
        self.turn_data_list = turn_data_list
    
    def get_question_document_pairs(self) -> pd.DataFrame:
        pair_data = []
        for turn in self.turn_data_list:
            for doc in turn.documents:
                pair_data.append({
                    'conv_id': turn.conv_id, 'turn_id': turn.turn_id,
                    'question': turn.question, 'document_text': doc.text,
                    'score': doc.score, 'has_answer': doc.has_answer
                })
        return pd.DataFrame(pair_data)
