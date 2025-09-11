#!/usr/bin/env python3
"""
PKLファイルを読み込んでDatasetクラスで使用するためのヘルパー関数
"""
import pickle
import os

def load_pkl_file(file_path):
    """
    PKLファイルを読み込んで会話データを返す
    
    Args:
        file_path (str): PKLファイルのパス
        
    Returns:
        list: 会話データのリスト
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def convert_dialogue_to_conversations(dialogue_data):
    """
    ダイアログデータを会話データに変換
    
    Args:
        dialogue_data (dict): ダイアログデータ
        
    Returns:
        list: 会話データのリスト
    """
    conversations = []
    
    for dialogue_id, dialogue in dialogue_data.items():
        # 各ダイアログのターンを会話として扱う
        turns = dialogue.get('turns', [])
        conversation = []
        
        for turn_index, turn in enumerate(turns):
            # ターンを会話形式に変換
            labels = turn.get('labels', [])
            if not labels:
                continue
                
            first_label = labels[0]
            
            # QPP特徴量を取得
            qpp_features = {}
            qpp_feature_names = ['ndcg@1', 'ndcg@3', 'ndcg@5', 'precision@1', 'precision@3', 'precision@5', 'recall@1', 'recall@3', 'recall@5']
            for feature_name in qpp_feature_names:
                qpp_features[feature_name] = first_label.get(feature_name, 0.0)
            
            conversation_turn = {
                'turn_id': turn_index,
                'user_utterance': turn.get('query', ''),
                'user_I_label': 'clarification', # ユーザーは常に確認発話
                'system_utterance': first_label.get('response', ''),
                'system_I_label': first_label.get('responseType', ''),
                'qpp_features': qpp_features,  # QPP特徴量を追加
            }
            conversation.append(conversation_turn)
        
        if conversation:  # 空でない場合のみ追加
            conversations.append(conversation)
    
    return conversations

def load_dev_data():
    """
    開発用データを読み込んで会話データに変換
    
    Returns:
        list([turn_id, user_utterance, user_I_label, system_utterance, system_I_label]): 会話データのリスト
    """
    file_path = "/home/daiki_shibata/pj/QPP4SIP/SIP/SIP/dataset/dev_resolved_retrieved.pkl"
    dialogue_data = load_pkl_file(file_path)
    return convert_dialogue_to_conversations(dialogue_data)

if __name__ == "__main__":
    # テスト実行
    try:
        data = load_dev_data()
        print(f"データを正常に読み込みました")
        print(f"データの型: {type(data)}")
        if isinstance(data, list):
            print(f"会話数: {len(data)}")
            if data:
                print(f"最初の会話の型: {type(data[0])}")
                if isinstance(data[0], list):
                    print(f"最初の会話のターン数: {len(data[0])}")
                    print("\n=== 最初の会話の内容 ===")
                    for i, turn in enumerate(data[0][:3]):  # 最初の3ターンを表示
                        print(f"\n--- ターン {i+1} ---")
                        print(f"ユーザー発話: {turn.get('user_utterance', 'N/A')}")
                        print(f"ユーザー意図ラベル: {turn.get('user_I_label', 'N/A')}")
                        print(f"システム発話: {turn.get('system_utterance', 'N/A')}")
                        print(f"システム意図ラベル: {turn.get('system_I_label', 'N/A')}")
        elif isinstance(data, dict):
            print(f"ダイアログ数: {len(data)}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
