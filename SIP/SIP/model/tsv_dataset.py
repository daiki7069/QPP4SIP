import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np

class TSVDataset(Dataset):
    """
    TSVファイル形式のSIP予測タスク用データセットクラス
    QPP特徴量を含むINSCITデータセットに対応
    """
    def __init__(self, tsv_file: str, args, config, mlb=None):
        """
        Args:
            tsv_file: TSVファイルのパス
            args: 設定パラメータ
            config: 設定オブジェクト
            mlb: MultiLabelBinarizer（互換性のため）
        """
        super(TSVDataset, self).__init__()
        self.args = args
        self.config = config
        self.mlb = mlb
        
        # TSVファイルを読み込み
        self.data = pd.read_csv(tsv_file, sep='\t')
        
        # トークナイザーを初期化
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_utterance_len = getattr(args, 'max_utterance_len', 512)
        self.max_context_len = getattr(args, 'max_context_len', 512)
        
        # データを会話単位でグループ化
        self.conversations = self._group_by_conversation()
        
        # テンソル化
        self.conversations_tensor = []
        self.load()

    def _group_by_conversation(self):
        """データを会話単位でグループ化"""
        conversations = {}
        
        for _, row in self.data.iterrows():
            dialogue_id = row['dialogue_id']
            turn_id = row['turn_id']
            
            if dialogue_id not in conversations:
                conversations[dialogue_id] = []
            
            # SIPラベルの決定（clarification -> 1, その他 -> 0）
            sip_label = 1 if 'clarification' in str(row['response_type']) else 0
            
            # QPP特徴量を抽出
            qpp_features = self._extract_qpp_features(row)
            
            conversations[dialogue_id].append({
                'turn_id': turn_id,
                'query': row['query'],
                'response': row['response'],
                'response_type': row['response_type'],
                'dialogue_history': row['dialogue_history'],
                'sip_label': sip_label,
                'qpp_features': qpp_features
            })
        
        # 会話をターン順でソート
        for dialogue_id in conversations:
            conversations[dialogue_id].sort(key=lambda x: x['turn_id'])
        
        return list(conversations.values())

    def _extract_qpp_features(self, row):
        """QPP特徴量を抽出"""
        qpp_feature_names = [
            'mrr', 'found_ratio', 'mean_rank',
            'hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50',
            'precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50',
            'recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50',
            'f1@1', 'f1@5', 'f1@10', 'f1@20', 'f1@50',
            'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50'
        ]
        
        qpp_features = {}
        for name in qpp_feature_names:
            if name in row:
                qpp_features[name] = float(row[name]) if pd.notna(row[name]) else 0.0
            else:
                qpp_features[name] = 0.0
        
        return qpp_features

    def load(self):
        """データをテンソル化してロード"""
        for conversation_index, conversation in enumerate(self.conversations):
            conversation_content = {
                "turn_id": [],
                "user_utterance": [],
                "user_I_label": [],
                "system_utterance": [],
                "system_I_label": [],
                "context": [],
                "system_action_label": [],
                "system_action_sequence": [],
                "system_I_prediction": [],
                "qpp_features": [],
                "resolved_query": []
            }

            context_list = []

            for turn_index, turn in enumerate(conversation):
                # 現在のターンを処理
                conversation_content["turn_id"].append(turn_index)
                
                # ユーザー発話（クエリ）
                user_utterance = turn["query"]
                conversation_content["user_utterance"].append(
                    torch.tensor(
                        self.tokenizer.encode(
                            user_utterance, 
                            add_special_tokens=True, 
                            max_length=self.max_utterance_len, 
                            padding="max_length", 
                            truncation=True
                        )
                    )
                )
                
                # ユーザーイニシアチブラベル（常に0、ユーザーは質問のみ）
                conversation_content["user_I_label"].append(torch.tensor(0))
                
                # システム発話（レスポンス）
                system_utterance = turn["response"]
                conversation_content["system_utterance"].append(
                    torch.tensor(
                        self.tokenizer.encode(
                            system_utterance, 
                            add_special_tokens=True, 
                            max_length=self.max_utterance_len, 
                            padding="max_length", 
                            truncation=True
                        )
                    )
                )
                
                # システムイニシアチブラベル（SIPラベル）
                conversation_content["system_I_label"].append(torch.tensor(turn["sip_label"]))
                
                # QPP特徴量を処理
                qpp_features = turn["qpp_features"]
                # 主要なQPP特徴量を選択（必要に応じて調整）
                selected_qpp_features = [
                    'ndcg@1', 'ndcg@3', 'ndcg@5', 
                    'precision@1', 'precision@5', 
                    'recall@1', 'recall@5',
                    'f1@1', 'f1@5'
                ]
                qpp_tensor = torch.tensor(
                    [qpp_features.get(name, 0.0) for name in selected_qpp_features], 
                    dtype=torch.float32
                )
                conversation_content["qpp_features"].append(qpp_tensor)
                
                # resolved_query（クエリをそのまま使用）
                conversation_content["resolved_query"].append(turn["query"])
                
                # コンテキストを更新
                context_list.append(turn["query"])
                assert len(context_list) == turn_index * 2 + 1

                context_text = " ".join(context_list)

                context_tokens = self.tokenizer.tokenize(context_text)

                # コンテキストウィンドウサイズに調整
                if len(context_tokens) > (self.max_context_len - 2):
                    context_tokens_ = context_tokens[-(self.max_context_len - 2):]  # 510 tokens
                    context_tokens_ = ['[CLS]'] + context_tokens_ + ['[SEP]']
                else:
                    context_tokens_ = ['[CLS]'] + context_tokens + ['[SEP]'] + ['[PAD]'] * (self.max_context_len - 2 - len(context_tokens))

                assert len(context_tokens_) == self.max_context_len

                context_id = self.tokenizer.convert_tokens_to_ids(context_tokens_)
                conversation_content["context"].append(torch.tensor(context_id))
                context_list.append(turn["response"])

            assert len(conversation_content["user_utterance"]) == len(conversation_content["user_I_label"]) == len(
                conversation_content["system_utterance"]) == len(conversation_content["system_I_label"]) == len(
                conversation_content["context"]) == len(conversation_content["qpp_features"])

            user_utterance_conversation = torch.stack(conversation_content["user_utterance"])   # [?, max_utterance_len]
            user_I_label_conversation = torch.stack(conversation_content["user_I_label"])   # [?, 1]
            system_utterance_conversation = torch.stack(conversation_content["system_utterance"])   # [?, max_utterance_len]
            system_I_label_conversation = torch.stack(conversation_content["system_I_label"])   # [?, 1]
            context_conversation = torch.stack(conversation_content["context"])   # [?, max_context_len]
            qpp_features_conversation = torch.stack(conversation_content["qpp_features"])   # [?, 9] (9個のQPP特徴量)

            self.conversations_tensor.append(
                [
                    conversation_content["turn_id"],
                    user_utterance_conversation,
                    user_I_label_conversation,
                    system_utterance_conversation,
                    system_I_label_conversation,
                    context_conversation,
                    qpp_features_conversation,
                    conversation_content["resolved_query"]  # resolved_queryを追加
                ]
            )
            self.len = conversation_index + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        conversation_tensor = self.conversations_tensor[idx]
        return [conversation_tensor[0], conversation_tensor[1], conversation_tensor[2], conversation_tensor[3], conversation_tensor[4], conversation_tensor[5], conversation_tensor[6], conversation_tensor[7]]

def collate_fn(data):
    turn_id, user_utterance_conversations, user_I_label_conversations, system_utterance_conversations, system_I_label_conversations, context_conversations, qpp_features_conversations, resolved_query_conversations = zip(*data)
    return {
        "turn_id": turn_id[-1], # [batch_size, 1]
        "user_utterance": torch.stack(user_utterance_conversations), # [batch_size, ?, max_utterance_len]
        "user_I_label": torch.stack(user_I_label_conversations), # [batch_size, ?, 1]
        "system_utterance": torch.stack(system_utterance_conversations), # [batch_size, ?, max_utterance_len]
        "system_I_label": torch.stack(system_I_label_conversations), # [batch_size, ?, 1]
        "context": torch.stack(context_conversations), # [batch_size, ?, max_context_len]
        "qpp_features": torch.stack(qpp_features_conversations), # [batch_size, ?, 9]
        "resolved_query": resolved_query_conversations[-1]  # [batch_size, ?] - 文字列のリスト
    }
