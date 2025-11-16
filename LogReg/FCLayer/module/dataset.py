"""
特徴量データセット
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class FeatureDataset(Dataset):
    """特徴量データセット"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

