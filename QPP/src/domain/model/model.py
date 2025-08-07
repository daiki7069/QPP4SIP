from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime
import json

@dataclass
class Model:
    """機械学習モデルエンティティ"""
    id: str
    name: str
    version: str
    model_type: str
    file_path: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: datetime
    training_dataset_id: str
    
    def to_json(self) -> str:
        """JSON形式でモデル情報を出力"""
        return json.dumps({
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "file_path": self.file_path,
            "hyperparameters": self.hyperparameters,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "training_dataset_id": self.training_dataset_id
        }, indent=2) 