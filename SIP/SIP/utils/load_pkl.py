#!/usr/bin/env python3
"""
TSVから変換されたPKLファイルを読み込んでDatasetクラスで使用するためのヘルパー関数
"""
import pickle
import os

def convert_dialogue_to_conversations(dialogue_data):
    """
    リスト形式のPKLダイアログデータを会話データに変換
    
    Args:
        dialogue_data (list): リスト形式のPKLダイアログデータ
        
    Returns:
        list: 会話データのリスト
    """
    conversations = []
    
    # 会話ごとにグループ化
    dialogue_groups = {}
    for item in dialogue_data:
        dialogue_id = item.get('dialogue_id', 'unknown')
        if dialogue_id not in dialogue_groups:
            dialogue_groups[dialogue_id] = []
        dialogue_groups[dialogue_id].append(item)
    
    # 各会話を処理
    for dialogue_id, turns in dialogue_groups.items():
        conversation = []
        
        for turn_index, turn in enumerate(turns):
            response_type = turn.get('response_type', '')
            if not response_type:
                continue
            
            # response_typeが複数含まれている場合は最初の部分のみを使用
            clean_response_type = response_type.split('|')[0].strip() if '|' in response_type else response_type.strip()
            
            # QPP特徴量を構築（文字列をfloatに変換）
            qpp_features = {
                'ndcg@1': float(turn.get('ndcg@1', 0.0)),
                'ndcg@3': float(turn.get('ndcg@3', 0.0)),
                'ndcg@5': float(turn.get('ndcg@5', 0.0)),
                'precision@1': float(turn.get('precision@1', 0.0)),
                'precision@3': float(turn.get('precision@3', 0.0)),
                'precision@5': float(turn.get('precision@5', 0.0)),
                'recall@1': float(turn.get('recall@1', 0.0)),
                'recall@3': float(turn.get('recall@3', 0.0)),
                'recall@5': float(turn.get('recall@5', 0.0)),
            }
            
            # system_utteranceが複数含まれている場合は最初の部分のみを使用
            system_utterance = turn.get('response', '')
            clean_system_utterance = system_utterance.split('|')[0].strip() if '|' in system_utterance else system_utterance.strip()
            
            # SIPラベルの決定
            # clarification -> SIP: 1 (システムがイニシアチブを取る)
            # その他 -> SIP: 0 (システムがイニシアチブを取らない)
            sip_label = 1 if clean_response_type == 'clarification' else 0
            
            conversation_turn = {
                'turn_id': turn_index,
                'user_utterance': turn.get('query', ''),
                'user_I_label': 'Non-initiative',
                'system_utterance': clean_system_utterance,
                'system_I_label': sip_label,  # 数値ラベルに変更
                'response_type': clean_response_type,
                'qpp_features': qpp_features,
                'resolved_query': turn.get('query', ''),
            }
            conversation.append(conversation_turn)
        
        if conversation:
            conversations.append(conversation)
    
    return conversations

if __name__ == "__main__":
    # テスト実行
    try:
        # テスト用のPKLファイルを読み込み
        test_file = "dataset/bm25_train.pkl"
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                dialogue_data = pickle.load(f)
            conversations = convert_dialogue_to_conversations(dialogue_data)
            
            print(f"データを正常に読み込みました")
            print(f"会話数: {len(conversations)}")
            if conversations:
                print(f"最初の会話のターン数: {len(conversations[0])}")
                print("\n=== 最初の会話の内容 ===")
                for i, turn in enumerate(conversations[0][:3]):  # 最初の3ターンを表示
                    print(f"\n--- ターン {i+1} ---")
                    print(f"ユーザー発話: {turn.get('user_utterance', 'N/A')}")
                    print(f"ユーザー意図ラベル: {turn.get('user_I_label', 'N/A')}")
                    print(f"システム発話: {turn.get('system_utterance', 'N/A')}")
                    print(f"システム意図ラベル: {turn.get('system_I_label', 'N/A')}")
        else:
            print(f"テストファイルが見つかりません: {test_file}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")