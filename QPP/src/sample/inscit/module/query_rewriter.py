from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Optional
import sys
import os

# モジュールのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.logger import get_logger


class QueryRewriter:
    """クエリ書き換えを行うクラス"""
    
    def __init__(self, model_name: str = "castorini/t5-base-canard", device: Optional[str] = None):
        """
        初期化
        
        Args:
            model_name: 使用するモデル名
            device: 使用するデバイス（Noneの場合は自動選択）
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.logger = get_logger("QueryRewriter")
        self._load_model()
    
    def _load_model(self):
        """モデルとトークナイザーを読み込む"""
        try:
            self.logger.progress(f"モデル '{self.model_name}' を読み込み中...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.success(f"モデルの読み込みが完了しました（デバイス: {self.device}）")
        except Exception as e:
            self.logger.error(f"モデルの読み込みに失敗しました: {e}")
            raise
    
    def rewrite_query(self, context: str, max_length: int = 200, num_beams: int = 4, 
                     repetition_penalty: float = 2.5, length_penalty: float = 1.0) -> str:
        """
        クエリを書き換える
        
        Args:
            context: 書き換えのためのコンテキスト
            max_length: 生成する最大長
            num_beams: ビームサーチの数
            repetition_penalty: 繰り返しペナルティ
            length_penalty: 長さペナルティ
            
        Returns:
            書き換えられたクエリ
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルが読み込まれていません")
        
        try:
            # コンテキストをトークン化
            tokenized_context = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
            
            # クエリを生成
            output_ids = self.model.generate(
                tokenized_context,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=True
            ).to(self.device)
            
            # トークンをデコード
            rewrite = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return rewrite
            
        except Exception as e:
            self.logger.error(f"クエリ書き換え中にエラーが発生しました: {e}")
            raise
    
    def rewrite_query_with_context(self, texts: List[str], separator: str = "|||") -> str:
        """
        テキストリストからコンテキストを作成してクエリを書き換える
        
        Args:
            texts: コンテキストとして使用するテキストのリスト
            separator: テキスト間の区切り文字
            
        Returns:
            書き換えられたクエリ
        """
        context = separator.join(texts)
        return self.rewrite_query(context)
    
    def batch_rewrite(self, contexts: List[str]) -> List[str]:
        """
        複数のコンテキストを一括で書き換える
        
        Args:
            contexts: 書き換えるコンテキストのリスト
            
        Returns:
            書き換えられたクエリのリスト
        """
        results = []
        for i, context in enumerate(contexts):
            try:
                rewrite = self.rewrite_query(context)
                results.append(rewrite)
                self.logger.progress(f"処理完了: {i+1}/{len(contexts)}")
            except Exception as e:
                self.logger.error(f"コンテキスト {i+1} の処理に失敗: {e}")
                results.append("")
        return results


def main():
    """メイン関数 - 使用例"""
    logger = get_logger("QueryRewriterTest")
    
    # サンプルテキスト
    sample_texts = [
        "Aside from cow's milk, what other animal milk is used in making cheese?",
        "Other sources of milk for cheese include goats and sheep's milk.",
        "Can cheese be made from soy milk?",
        "Yes,  the main ingredients are nuts, soy milk, and soy yogurt. Common plant-based proteins or vegetable proteins used in vegan cheeses are derived from edible sources of protein, such as soybeans.",
        "When did people first start making cheese?",
        "The production of cheese predates recorded history, beginning well over 7,000 years ago. There is no conclusive evidence indicating where cheese-making originated, possibly Europe, or Central Asia, the Middle East, or the Sahara.",
        "Do all cheeses need to be refrigerated?",
        "There's the Edam cheese which is a semi-hard cheese that does not spoil, only hardens.",
        "What are the effects of cheese on health?",
        "National health organizations like the American Heart Association among others recommend cheese consumption be minimized to reduce factors for cardiovascular diseases. Raw-milk cheeses may cause infectious diseases as well if not aged enough and pregnant women may face extra risks due to listeria risk from soft-ripened cheeses and blue-veined cheeses.",
        "What is the process of it?"
    ]
    
    try:
        logger.section("クエリ書き換えテスト")
        
        # QueryRewriterインスタンスを作成
        rewriter = QueryRewriter()
        
        # 方法1: コンテキストを直接指定
        context = "|||".join(sample_texts)
        rewrite1 = rewriter.rewrite_query(context)
        logger.info(f"方法1 - 元のクエリ: {sample_texts[-1]}")
        logger.info(f"方法1 - 書き換え: {rewrite1}")
        
        # 方法2: テキストリストを使用
        rewrite2 = rewriter.rewrite_query_with_context(sample_texts)
        logger.info(f"方法2 - 元のクエリ: {sample_texts[-1]}")
        logger.info(f"方法2 - 書き換え: {rewrite2}")
        
        logger.success("テストが完了しました")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()