import torch

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