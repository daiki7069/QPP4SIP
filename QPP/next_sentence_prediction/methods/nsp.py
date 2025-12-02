"""
Next Sentence Prediction (NSP) メソッド
BERT-base-uncased の NSP を使用して文の連続性を評価
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
from transformers import BertTokenizer, BertForNextSentencePrediction
from tqdm import tqdm


class NSP:
    """Next Sentence Prediction クラス"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased", 
        device: Optional[str] = None,
        batch_size: int = 32,
        use_multi_gpu: bool = True
    ):
        """
        NSPモデルを初期化
        
        Args:
            model_name: 使用するBERTモデル名
            device: 使用するデバイス（Noneの場合は自動選択）
            batch_size: バッチサイズ
            use_multi_gpu: 複数GPUを使用するかどうか
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # デバイスの設定
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # モデルとトークナイザをロード
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)
        
        # マルチGPU対応（GPU 0-3のみ使用）
        if use_multi_gpu and torch.cuda.device_count() > 1 and self.device.startswith('cuda'):
            # 使用可能なGPU: 0, 1, 2, 3
            available_gpus = [0, 1, 2, 3]
            # 実際に存在するGPUのみをフィルタリング
            device_ids = [gpu_id for gpu_id in available_gpus if gpu_id < torch.cuda.device_count()]
            
            if len(device_ids) > 1:
                print(f"Using GPUs {device_ids} with DataParallel")
                # モデルを最初のGPUに移動してからDataParallelでラップ
                self.model = self.model.to(f'cuda:{device_ids[0]}')
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
                # DataParallel使用時はdeviceを'cuda'に統一
                self.device = 'cuda'
            else:
                # 1枚しかない場合は通常通り
                self.model.to(self.device)
        else:
            self.model.to(self.device)
        
        self.model.eval()
    
    def predict(self, sent_a: str, sent_b: str) -> Tuple[float, float, str]:
        """
        文のペアが続きかどうかを予測
        
        Args:
            sent_a: 最初の文
            sent_b: 次の文
        
        Returns:
            (is_next_prob, not_next_prob, prediction): 
                IsNext確率、NotNext確率、予測結果（"IsNext" or "NotNext"）
        """
        result = self.predict_with_details(sent_a, sent_b)
        return result["is_next_prob"], result["not_next_prob"], result["prediction"]
    
    def predict_with_details(self, sent_a: str, sent_b: str) -> dict:
        """
        NSP結果を詳細情報付きで取得
        
        Returns:
            dict: {is_next_prob, not_next_prob, prediction, logits}
        """
        encoded = self.tokenizer(
            sent_a,
            sent_b,
            return_tensors="pt"
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits  # [batch, 2]
        
        probs = F.softmax(logits, dim=-1)[0]
        is_next_prob = probs[0].item()
        not_next_prob = probs[1].item()
        prediction = "IsNext" if is_next_prob > not_next_prob else "NotNext"
        
        return {
            "is_next_prob": is_next_prob,
            "not_next_prob": not_next_prob,
            "prediction": prediction,
            "logits": logits.cpu().numpy()
        }
    
    def predicts(
        self, 
        sentence_pairs: List[Tuple[str, str]], 
        show_progress: bool = False,
        desc: Optional[str] = None
    ) -> List[dict]:
        """
        複数の文ペアをバッチで予測（高速化）
        
        Args:
            sentence_pairs: [(sent_a, sent_b), ...] のリスト
            show_progress: 進捗バーを表示するかどうか
            desc: 進捗バーの説明文（Noneの場合はデフォルト）
        
        Returns:
            [dict, ...] のリスト（各dictはpredict_with_detailsの結果）
        """
        if len(sentence_pairs) == 0:
            return []
        
        results = []
        total_batches = (len(sentence_pairs) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(sentence_pairs), self.batch_size)
        
        if show_progress:
            progress_desc = desc if desc else "NSP batches"
            iterator = tqdm(
                iterator, 
                desc=progress_desc, 
                unit="batch",
                total=total_batches,
                leave=False,  # ネストされた進捗バーなのでleave=False
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        for i in iterator:
            batch_pairs = sentence_pairs[i:i + self.batch_size]
            batch_results = self._predict_batch(batch_pairs)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch(self, sentence_pairs: List[Tuple[str, str]]) -> List[dict]:
        """
        バッチでNSPを実行（内部メソッド）
        """
        sent_a_list = [pair[0] for pair in sentence_pairs]
        sent_b_list = [pair[1] for pair in sentence_pairs]
        
        # バッチでトークナイズ
        encoded = self.tokenizer(
            sent_a_list,
            sent_b_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # バッチで推論
        with torch.no_grad():
            outputs = self.model(**encoded)
            # DataParallel使用時も通常時も同じ形式で取得
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # フォールバック（通常は発生しない）
                logits = outputs
        
        # 各サンプルの結果を処理（GPU上で処理してからCPUに移動）
        probs = F.softmax(logits, dim=-1)  # [batch_size, 2]
        batch_results = []
        for j in range(len(sentence_pairs)):
            is_next_prob = probs[j][0].item()
            not_next_prob = probs[j][1].item()
            prediction = "IsNext" if is_next_prob > not_next_prob else "NotNext"
            
            batch_results.append({
                "is_next_prob": is_next_prob,
                "not_next_prob": not_next_prob,
                "prediction": prediction,
                "logits": logits[j].detach().cpu().numpy()
            })
        
        return batch_results
    
    def predict_batch(self, sentence_pairs: List[Tuple[str, str]]) -> List[Tuple[float, float, str]]:
        """
        複数の文ペアをバッチで予測（後方互換用）
        """
        results = []
        for sent_a, sent_b in sentence_pairs:
            is_next_prob, not_next_prob, prediction = self.predict(sent_a, sent_b)
            results.append((is_next_prob, not_next_prob, prediction))
        return results

