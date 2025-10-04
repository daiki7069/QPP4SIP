"""
数学・数値計算関連のユーティリティ関数
"""
import torch
import numpy as np


def hamming_score(y_true, y_pred):
    """ハミングスコアを計算"""
    return ((y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)).mean()


def rounder(num, places=2):
    """数値を指定した桁数で四捨五入"""
    num = num * 100
    return round(num, places)


def neginf(dtype):
    """指定されたdtypeの負の無限大に近い有限数を返す"""
    if dtype is torch.float16:
        return -1e4
    else:
        return -1e9


def universal_sentence_embedding(sentences, mask, sqrt=True):
    """
    文の埋め込みを計算
    
    Args:
        sentences: [batch_size, seq_len, hidden_size]
        mask: [batch_size, seq_len]
        sqrt: 平方根を取るかどうか
        
    Returns:
        [batch_size, hidden_size]
    """
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)

    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()

    sentence_sums /= divisor
    return sentence_sums
