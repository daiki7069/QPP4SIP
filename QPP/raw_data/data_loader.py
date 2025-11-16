"""
INSCIT/AmbigNQデータ用のデータローダー
JSONファイルを読み込み、CSVに変換する
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class DialogueTurn:
    dialogue_id: str
    turn_id: str
    query: str
    response: str
    response_type: str
    dialogue_history: str


class INSCITDataLoader:
    def __init__(self, data_dir: str, dataset: str = "INSCIT"):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
    
    def load_data(self, split: str = "dev") -> List[DialogueTurn]:
        """
        指定されたスプリットのデータを読み込む
        
        Args:
            split: "dev", "train", "test"
            
        Returns:
            DialogueTurnのリスト
        """
        file_path = self.data_dir / f"{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if self.dataset == "AmbigNQ":
            return self._load_ambignq_data(raw_data)
        else:
            return self._load_inscit_data(raw_data)
    
    def _load_inscit_data(self, raw_data: Dict) -> List[DialogueTurn]:
        """INSCIT形式のデータを読み込む"""
        dialogue_turns = []
        
        for dialogue_id, dialogue_data in raw_data.items():
            turns = dialogue_data.get('turns', [])
            dialogue_history = ""
            
            for turn_idx, turn in enumerate(turns):
                # クエリの取得（最後の発話を取得）
                context = turn.get('context', [])
                query = context[-1] if context else ""
                
                # レスポンスの取得
                labels = turn.get('labels', [])
                if labels:
                    response_data = labels[0]  # 最初のラベルを使用
                    response = response_data.get('response', '')
                    response_type = response_data.get('responseType', '')
                else:
                    response = ""
                    response_type = ""
                
                dialogue_turn = DialogueTurn(
                    dialogue_id=dialogue_id,
                    turn_id=str(turn_idx + 1),
                    query=query,
                    response=response,
                    response_type=response_type,
                    dialogue_history=dialogue_history
                )
                
                dialogue_turns.append(dialogue_turn)
                
                # 次のターンのために履歴を更新
                dialogue_history += f"Q: {query}\nA: {response}\n"
        
        return dialogue_turns
    
    def _load_ambignq_data(self, raw_data: List) -> List[DialogueTurn]:
        """AmbigNQ形式のデータを読み込む（会話のリストのリスト）"""
        dialogue_turns = []
        
        for conversation in raw_data:
            for turn in conversation:
                # クエリの取得
                query = turn.get('query', '')
                
                # 回答の取得（リストまたは文字列）
                answer = turn.get('answer', [])
                if isinstance(answer, list):
                    response = ' | '.join(answer) if answer else ''
                else:
                    response = str(answer) if answer else ''
                
                # レスポンスタイプの取得
                response_type = turn.get('response_type', '')
                
                # 会話履歴の取得
                dialogue_history_list = turn.get('dialogue_history', [])
                if isinstance(dialogue_history_list, list):
                    dialogue_history = '\n'.join(dialogue_history_list)
                else:
                    dialogue_history = str(dialogue_history_list) if dialogue_history_list else ''
                
                dialogue_turn = DialogueTurn(
                    dialogue_id=str(turn.get('conv_id', '')),
                    turn_id=str(turn.get('turn_id', '')),
                    query=query,
                    response=response,
                    response_type=response_type,
                    dialogue_history=dialogue_history
                )
                
                dialogue_turns.append(dialogue_turn)
        
        return dialogue_turns
    
    def convert_to_dataframe(self, dialogue_turns: List[DialogueTurn]) -> pd.DataFrame:
        """
        DialogueTurnのリストをDataFrameに変換
        
        Args:
            dialogue_turns: DialogueTurnのリスト
            
        Returns:
            DataFrame
        """
        data = []
        for turn in dialogue_turns:
            data.append({
                'dialogue_id': turn.dialogue_id,
                'turn_id': turn.turn_id,
                'query': turn.query,
                'response': turn.response,
                'response_type': turn.response_type
            })
        
        return pd.DataFrame(data)
