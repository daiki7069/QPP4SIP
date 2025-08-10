import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import os

def load_inscit_data(file_path: str) -> List[Dict[str, Any]]:
    """
    INSCITデータセットを読み込み、SIP予測タスク用に変換する
    
    Args:
        file_path: INSCITデータファイルのパス
        
    Returns:
        SIP予測タスク用のデータリスト
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for dialogue_id, dialogue in data.items():
        turns = dialogue.get('turns', [])
        
        for i, turn in enumerate(turns):
            # 各ターンのラベルを処理
            labels = turn.get('labels', [])
            
            for label in labels:
                response_type = label.get('responseType', '')
                response = label.get('response', '')
                
                # SIPラベルの決定
                # clarification -> SIP: 1 (システムがイニシアチブを取る)
                # その他 -> SIP: 0 (システムがイニシアチブを取らない)
                sip_label = 1 if response_type == 'clarification' else 0
                
                # 対話履歴の構築
                dialogue_history = []
                for j in range(i):
                    prev_turn = turns[j]
                    prev_query = prev_turn.get('query', '')
                    prev_labels = prev_turn.get('labels', [])
                    if prev_labels:
                        prev_response = prev_labels[0].get('response', '')
                        dialogue_history.append(f"User: {prev_query}")
                        dialogue_history.append(f"System: {prev_response}")
                
                # 現在のターンの情報
                current_query = turn.get('query', '')
                
                # 知識ベースの情報（retrieved_evidenceから）
                evidence_texts = []
                for evidence in turn.get('retrieved_evidence', []):
                    evidence_texts.append(evidence.get('passage_text', ''))
                knowledge_text = ' '.join(evidence_texts)
                
                # 対話履歴と知識を結合
                history_text = ' '.join(dialogue_history)
                full_context = f"{history_text} [SEP] User: {current_query}"
                
                processed_data.append({
                    'dialogue_id': dialogue_id,
                    'turn_index': i,
                    'query': current_query,
                    'response': response,
                    'response_type': response_type,
                    'sip_label': sip_label,
                    'context': full_context,
                    'knowledge': knowledge_text,
                    'dialogue_history': history_text
                })
    
    return processed_data

def create_sip_dataset(train_path: str, dev_path: str, test_path: str, output_dir: str):
    """
    SIP予測タスク用のデータセットを作成する
    
    Args:
        train_path: 訓練データのパス
        dev_path: 開発データのパス
        test_path: テストデータのパス
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 各データセットを処理
    datasets = {
        'train': train_path,
        'dev': dev_path,
        'test': test_path
    }
    
    for split_name, file_path in datasets.items():
        print(f"Processing {split_name} dataset...")
        data = load_inscit_data(file_path)
        
        # DataFrameに変換
        df = pd.DataFrame(data)
        
        # 統計情報を表示
        print(f"{split_name} dataset statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  SIP=0 (no initiative): {len(df[df['sip_label'] == 0])}")
        print(f"  SIP=1 (initiative): {len(df[df['sip_label'] == 1])}")
        print(f"  Response type distribution:")
        print(df['response_type'].value_counts())
        print()
        
        # CSVファイルとして保存
        output_file = os.path.join(output_dir, f"{split_name}_sip.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {split_name} dataset to {output_file}")
        
        # ラベル分布を保存
        label_dist_file = os.path.join(output_dir, f"{split_name}_label_distribution.txt")
        with open(label_dist_file, 'w', encoding='utf-8') as f:
            f.write(f"{split_name} dataset label distribution:\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"SIP=0 (no initiative): {len(df[df['sip_label'] == 0])}\n")
            f.write(f"SIP=1 (initiative): {len(df[df['sip_label'] == 1])}\n")
            f.write(f"Response type distribution:\n")
            f.write(df['response_type'].value_counts().to_string())
            f.write("\n")

def main():
    """メイン関数"""
    # データファイルのパス
    train_path = "/mnt/disk6/daiki/QPP4SIP/Datasets/dialogue/train_resolved_retrieved.json"
    dev_path = "/mnt/disk6/daiki/QPP4SIP/Datasets/dialogue/dev_resolved_retrieved.json"
    test_path = "/mnt/disk6/daiki/QPP4SIP/Datasets/dialogue/test_resolved_retrieved.json"
    
    # 出力ディレクトリ
    output_dir = "data"
    
    # データセットを作成
    create_sip_dataset(train_path, dev_path, test_path, output_dir)
    
    print("SIP dataset creation completed!")

if __name__ == "__main__":
    main()