"""
Post-retrieval QPP用のスコア分布分析
"""
import pandas as pd
import numpy as np
from typing import List
from data_loader import TurnData


class ScoreAnalyzer:
    def __init__(self, turn_data_list: List[TurnData]):
        self.turn_data_list = turn_data_list
    
    def get_score_statistics(self) -> pd.DataFrame:
        stats_data = []
        for turn in self.turn_data_list:
            if not turn.scores:
                continue
            scores = np.array(turn.scores)
            stats_data.append({
                'conv_id': turn.conv_id, 'turn_id': turn.turn_id,
                'score_mean': np.mean(scores), 'score_std': np.std(scores),
                'score_min': np.min(scores), 'score_max': np.max(scores)
            })
        return pd.DataFrame(stats_data)
