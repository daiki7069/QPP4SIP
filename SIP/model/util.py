import torch
import re
from typing import List, Optional

def create_class_weights(class_counts: list, method: str = 'balanced'):
    """
    クラス重みを計算
    
    Args:
        class_counts: 各クラスのサンプル数のリスト [n0, n1, n2, ...]
        method: 重み計算方法 ('balanced' for sklearn balanced方式)
    
    Returns:
        weights: クラス重みテンソル [num_classes]
    """
    if method == 'balanced':
        N = sum(class_counts)
        C = len(class_counts)  # クラス数
        weights = torch.tensor([N / (C * count) for count in class_counts], dtype=torch.float32)
        return weights
    else:
        raise ValueError(f"Unknown method: {method}")

def get_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor = None):
    """
    重み付きクロスエントロピー損失を計算
    
    Args:
        logits: モデルの出力ロジット [batch_size, num_classes]
        targets: 正解ラベル [batch_size]
        class_weights: クラス重み [num_classes] (Noneの場合は通常のCrossEntropyLoss)
    
    Returns:
        loss: 計算された損失
    """
    import torch.nn as nn
    
    if class_weights is not None:
        # 重み付きクロスエントロピー損失
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # 通常のクロスエントロピー損失
        criterion = nn.CrossEntropyLoss()
    
    return criterion(logits, targets)

def preprocess_dialogue_history(dialogue_history: str, max_turns: Optional[int] = None) -> str:
    """
    対話履歴を前処理する
    
    Args:
        dialogue_history: 生の対話履歴テキスト
        max_turns: 最大ターン数（Noneの場合は制限なし）
    
    Returns:
        前処理された対話履歴テキスト
    """
    if not dialogue_history or dialogue_history.strip() == '':
        return '[EMPTY]'
    
    # 基本的なクリーニング
    cleaned = dialogue_history.strip()
    
    # ターン数制限がある場合
    if max_turns is not None:
        # [SEP]で区切ってターンを分割
        turns = cleaned.split('[SEP]')
        if len(turns) > max_turns:
            # 最新のmax_turns個のターンのみを使用
            turns = turns[-max_turns:]
            cleaned = '[SEP]'.join(turns)
    
    return cleaned

def format_dialogue_for_bert(dialogue_history: str, knowledge: Optional[str] = None) -> str:
    """
    BERT入力用に対話履歴をフォーマットする
    
    Args:
        dialogue_history: 対話履歴
        knowledge: 知識情報（オプション）
    
    Returns:
        BERT入力用にフォーマットされたテキスト
    """
    if knowledge and knowledge.strip():
        # 知識情報がある場合は先頭に追加
        formatted = f"[Knowledge] {knowledge.strip()} [SEP] {dialogue_history}"
    else:
        formatted = dialogue_history
    
    return formatted

def validate_dialogue_history(dialogue_history: str) -> bool:
    """
    対話履歴の妥当性をチェックする
    
    Args:
        dialogue_history: 対話履歴テキスト
    
    Returns:
        妥当性の真偽値
    """
    if not dialogue_history or dialogue_history.strip() == '':
        return False
    
    # 基本的な形式チェック（User: や System: が含まれているか）
    if not re.search(r'(User:|System:)', dialogue_history):
        return False
    
    return True 