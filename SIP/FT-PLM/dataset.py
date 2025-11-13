"""
データセットの読み込みと前処理
"""
import json
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def load_data(json_path: str) -> List[Dict]:
    """
    JSONファイルからデータを読み込む
    
    Args:
        json_path: JSONファイルのパス
        
    Returns:
        データのリスト
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # ネストされたリストをフラット化
    flattened_data = []
    for conversation in data:
        for turn in conversation:
            flattened_data.append(turn)
    
    return flattened_data


def extract_response_type(response_type: str) -> str:
    """
    response_typeから前方のタイプを抽出
    
    Args:
        response_type: response_type文字列（例: "directAnswer [SEP] clarification"）
        
    Returns:
        前方のresponse_type
    """
    if '[SEP]' in response_type:
        return response_type.split('[SEP]')[0].strip()
    return response_type.strip()


def is_clarification(response_type: str) -> bool:
    """
    response_typeがclarificationかどうかを判定
    
    Args:
        response_type: response_type文字列
        
    Returns:
        clarificationの場合はTrue、それ以外はFalse
    """
    extracted = extract_response_type(response_type)
    return extracted.lower() == 'clarification'


class ClarificationDataset(Dataset):
    """
    Clarification予測用のデータセット
    """
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        response_type = item['response_type']
        label = 1 if is_clarification(response_type) else 0
        
        # トークナイズ
        encoding = self.tokenizer(
            query,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

