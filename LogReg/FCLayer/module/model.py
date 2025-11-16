"""
全結合層による分類器モデル
"""
import torch
import torch.nn as nn
from typing import List


class FullyConnectedClassifier(nn.Module):
    """全結合層による分類器"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.1):
        super(FullyConnectedClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隠れ層を構築
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 出力層（2クラス分類）
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

