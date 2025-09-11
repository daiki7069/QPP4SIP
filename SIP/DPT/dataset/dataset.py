import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
from typing import List, Dict, Any, Tuple

class SIPDataset(Dataset):
    """
    SIP予測タスク用のデータセットクラス
    """
    def __init__(self, csv_file: str, tokenizer: BertTokenizer, max_length: int = 512):
        """
        Args:
            csv_file: CSVファイルのパス
            tokenizer: BERTトークナイザー
            max_length: 最大シーケンス長
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # コンテキストテキストを取得
        context = row['context']
        
        # トークナイゼーション
        encoding = self.tokenizer(
            context,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # ラベルを取得
        label = row['sip_label']
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'dialogue_id': row['dialogue_id'],
            'turn_index': row['turn_index'],
            'query': row['query'],
            'response': row['response'],
            'response_type': row['response_type']
        }

def create_data_loaders(train_csv: str, dev_csv: str, test_csv: str, 
                       batch_size: int = 16, max_length: int = 512) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    データローダーを作成する
    
    Args:
        train_csv: 訓練データのCSVファイルパス
        dev_csv: 開発データのCSVファイルパス
        test_csv: テストデータのCSVファイルパス
        batch_size: バッチサイズ
        max_length: 最大シーケンス長
        
    Returns:
        訓練、開発、テストのDataLoader
    """
    # トークナイザーを初期化
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # データセットを作成
    train_dataset = SIPDataset(train_csv, tokenizer, max_length)
    dev_dataset = SIPDataset(dev_csv, tokenizer, max_length)
    test_dataset = SIPDataset(test_csv, tokenizer, max_length)
    
    # DataLoaderを作成
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, dev_loader, test_loader 