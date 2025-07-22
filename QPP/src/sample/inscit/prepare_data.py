import os
import sys
from typing import Dict, List, Any, Optional

# モジュールのパスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util.load_data import DataLoader
from util.logger import get_logger
from module.query_rewriter import QueryRewriter


class DataPreprocessor:
    """データの前処理を行うクラス"""
    
    def __init__(self, input_file_path: str):
        """
        初期化
        
        Args:
            input_file_path: 入力JSONファイルのパス
        """
        self.input_file_path = input_file_path
        self.output_file_path = self._generate_output_path()
        self.data_loader = DataLoader(input_file_path)
        self.query_rewriter = QueryRewriter()
        self.logger = get_logger("DataPreprocessor")
    
    def _generate_output_path(self) -> str:
        """出力ファイルのパスを生成"""
        base_dir = os.path.dirname(self.input_file_path)
        base_name = os.path.basename(self.input_file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        return os.path.join(base_dir, f"{name_without_ext}_resolved.json")
    
    def process_dialogue(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """
        ダイアログを処理する
        
        Args:
            dialogue: 処理するダイアログ
            
        Returns:
            処理されたダイアログ
        """
        processed_dialogue = dialogue.copy()
        turns = self.data_loader.get_turns(dialogue)
        
        self.logger.info(f"ターン数: {len(turns)}")
        
        for i, turn in enumerate(turns):
            self.logger.progress(f"ターン {i+1} を処理中...")
            
            # contextを取得
            context = self.data_loader.get_context(turn)
            
            if context:
                # contextの最後の要素をqueryとして抽出
                query = context[-1]
                turn['query'] = query
                
                # context全体を入力としてquery rewriteを実行
                try:
                    context_text = "|||".join(context)
                    resolved_query = self.query_rewriter.rewrite_query(context_text)
                    turn['resolvedQuery'] = resolved_query
                    
                    self.logger.info(f"  元のクエリ: {query}")
                    self.logger.info(f"  書き換えクエリ: {resolved_query}")
                    
                except Exception as e:
                    self.logger.error(f"  クエリ書き換えに失敗: {e}")
                    turn['resolvedQuery'] = query  # 失敗時は元のクエリを使用
            else:
                self.logger.warning(f"  コンテキストが見つかりません")
                turn['query'] = ""
                turn['resolvedQuery'] = ""
        
        return processed_dialogue
    
    def process_data(self, start_index: int = 0, end_index: Optional[int] = None) -> Dict[str, Any]:
        """
        データを処理する
        
        Args:
            start_index: 処理開始インデックス（デフォルト: 0）
            end_index: 処理終了インデックス（Noneの場合は最後まで）
            
        Returns:
            処理されたデータ
        """
        self.logger.section("データ処理開始")
        self.logger.info(f"入力ファイル: {self.input_file_path}")
        self.logger.info(f"出力ファイル: {self.output_file_path}")
        
        # 元のデータを取得
        original_data = self.data_loader.get_data()
        processed_data = original_data.copy()
        
        # 処理対象のキーを取得
        top_keys = self.data_loader.get_top_keys()
        total_keys = len(top_keys)
        
        # 終了インデックスを設定
        if end_index is None:
            end_index = total_keys
        
        self.logger.info(f"総ダイアログ数: {total_keys}")
        self.logger.info(f"処理対象範囲: {start_index} から {end_index-1} まで")
        self.logger.info(f"処理対象数: {end_index - start_index}")
        
        # 指定された範囲のダイアログを処理
        for i in range(start_index, end_index):
            try:
                dialogue_key = top_keys[i]
                self.logger.subsection(f"ダイアログ {i+1}/{end_index} を処理中")
                self.logger.info(f"ダイアログキー: {dialogue_key}")
                
                # ダイアログを取得
                dialogue = self.data_loader.get_dialogue_by_key(dialogue_key)
                
                # ダイアログを処理
                processed_dialogue = self.process_dialogue(dialogue)
                
                # 処理されたダイアログで元のデータを更新
                processed_data[dialogue_key] = processed_dialogue
                
                self.logger.success(f"ダイアログ {i+1} の処理が完了しました")
                
            except Exception as e:
                self.logger.error(f"ダイアログ {i+1} の処理中にエラーが発生しました: {e}")
                # エラーが発生しても処理を続行
                continue
        
        self.logger.success("すべてのデータ処理が完了しました")
        return processed_data
    
    def save_processed_data(self, processed_data: Dict[str, Any]):
        """処理されたデータを保存"""
        try:
            self.data_loader.save_data(processed_data, self.output_file_path)
            self.logger.success(f"処理されたデータを保存しました: {self.output_file_path}")
        except Exception as e:
            self.logger.error(f"データの保存に失敗しました: {e}")
            raise


def main():
    """メイン関数"""
    input_file_path = '/mnt/disk6/daiki/Datasets/INSCIT/data/test.json' # FIXME: ファイル名を変更
    logger = get_logger("DataPreprocessorMain")
    
    try:
        # DataPreprocessorインスタンスを作成
        preprocessor = DataPreprocessor(input_file_path)
        
        # データを処理（すべてのダイアログ）
        processed_data = preprocessor.process_data()
        
        # 処理されたデータを保存
        preprocessor.save_processed_data(processed_data)
        
        logger.success("すべての処理が完了しました！")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")


if __name__ == '__main__':
    main()
