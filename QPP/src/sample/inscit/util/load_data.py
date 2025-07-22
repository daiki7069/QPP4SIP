import json
from typing import Dict, List, Any, Optional
from .logger import get_logger


class DataLoader:
    """JSONデータを読み込み、操作するためのクラス"""
    
    def __init__(self, file_path: str):
        """
        初期化
        
        Args:
            file_path: 読み込むJSONファイルのパス
        """
        self.file_path = file_path
        self.data = None
        self.logger = get_logger("DataLoader")
        self._load_data()
    
    def _load_data(self):
        """JSONデータを読み込む"""
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
            self.logger.success(f"データを正常に読み込みました: {self.file_path}")
        except Exception as e:
            self.logger.error(f"データの読み込みに失敗しました: {e}")
            raise
    
    def get_data(self) -> Dict[str, Any]:
        """読み込んだデータを取得"""
        return self.data
    
    def get_top_keys(self) -> List[str]:
        """最上位のキーを取得"""
        if self.data is None:
            raise RuntimeError("データが読み込まれていません")
        return list(self.data.keys())
    
    def get_dialogue_by_key(self, key: str) -> Dict[str, Any]:
        """指定されたキーのダイアログを取得"""
        if self.data is None:
            raise RuntimeError("データが読み込まれていません")
        if key not in self.data:
            raise KeyError(f"キー '{key}' が見つかりません")
        return self.data[key]
    
    def get_dialogue_by_index(self, index: int) -> Dict[str, Any]:
        """インデックスでダイアログを取得"""
        keys = self.get_top_keys()
        if index >= len(keys):
            raise IndexError(f"インデックス {index} が範囲外です")
        return self.get_dialogue_by_key(keys[index])
    
    def get_turns(self, dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ダイアログからターンを取得"""
        if 'turns' not in dialogue:
            raise KeyError("'turns'キーが見つかりません")
        return dialogue['turns']
    
    def get_labels(self, turn: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ターンからラベルを取得"""
        if 'labels' not in turn:
            raise KeyError("'labels'キーが見つかりません")
        return turn['labels']
    
    def get_response_type(self, label: Dict[str, Any]) -> str:
        """ラベルからレスポンスタイプを取得"""
        if 'responseType' not in label:
            raise KeyError("'responseType'キーが見つかりません")
        return label['responseType']
    
    def get_context(self, turn: Dict[str, Any]) -> List[str]:
        """ターンからコンテキストを取得"""
        if 'context' not in turn:
            raise KeyError("'context'キーが見つかりません")
        return turn['context']
    
    def save_data(self, data: Dict[str, Any], output_path: str):
        """データをJSONファイルとして保存"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.success(f"データを保存しました: {output_path}")
        except Exception as e:
            self.logger.error(f"データの保存に失敗しました: {e}")
            raise


def main():
    """メイン関数 - 使用例"""
    file_path = '/mnt/disk6/daiki/Datasets/INSCIT/data/test.json'
    logger = get_logger("DataLoaderTest")
    
    try:
        logger.section("データローダーテスト")
        
        # DataLoaderインスタンスを作成
        loader = DataLoader(file_path)
        
        # 最初のダイアログを取得
        first_dialogue = loader.get_dialogue_by_index(0)
        turns = loader.get_turns(first_dialogue)
        
        logger.info(f"ターン数: {len(turns)}")
        
        for i, turn in enumerate(turns):
            labels = loader.get_labels(turn)
            logger.info(f"ターン {i+1} のラベル数: {len(labels)}")
            
            for j, label in enumerate(labels):
                response_type = loader.get_response_type(label)
                logger.info(f"  ラベル {j+1} のレスポンスタイプ: {response_type}")
        
        logger.success("テストが完了しました")
                
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")


if __name__ == '__main__':
    main()