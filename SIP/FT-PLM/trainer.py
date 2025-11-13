"""
カスタムTrainerクラス
"""
import torch.nn as nn
from transformers import Trainer


class WeightedTrainer(Trainer):
    """
    クラス重み付き損失関数を使用するTrainer
    """
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        カスタム損失関数の計算
        
        Args:
            model: モデル
            inputs: 入力データ
            return_outputs: 出力も返すかどうか
            **kwargs: その他の引数
        
        Returns:
            loss: 損失値（return_outputs=Trueの場合は(loss, outputs)のタプル）
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.criterion(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

