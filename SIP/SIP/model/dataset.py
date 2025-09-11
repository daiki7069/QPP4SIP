import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class Dataset(Dataset):
    """
    """
    def __init__(self, args, config, mlb, conversations):
        super(Dataset, self).__init__()
        self.args = args
        self.config = config
        self.mlb = mlb
        self.conversations = conversations

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.max_utterance_len = getattr(args, 'max_utterance_len', 512)

        self.conversations_tensor = []
        self.load()

    def load(self):
        for conversation_index, conversation in enumerate(self.conversations):
            conversation_content = {"turn_id": [], "user_utterance": [], "user_I_label": [], "system_utterance": [],
                                    "system_I_label": [], "context": [], "system_action_label": [],
                                    "system_action_sequence": [],"system_I_prediction":[], "qpp_features": []}

            context_list = []

            for turn_index, turn in enumerate(conversation):
                # process current turn
                conversation_content["turn_id"].append(turn_index)
                conversation_content["user_utterance"].append(torch.tensor(self.tokenizer.encode(turn["user_utterance"], add_special_tokens=True, max_length=self.max_utterance_len, padding="max_length", truncation=True)))
                conversation_content["user_I_label"].append(torch.tensor(1) if turn["user_I_label"]=="clarification" else torch.tensor(0))
                conversation_content["system_utterance"].append(torch.tensor(self.tokenizer.encode(turn["system_utterance"], add_special_tokens=True, max_length=self.max_utterance_len, padding="max_length", truncation=True)))
                conversation_content["system_I_label"].append(torch.tensor(1) if turn["system_I_label"]=="clarification" else torch.tensor(0))
                
                # QPP特徴量を処理
                qpp_features = turn.get("qpp_features", {})
                qpp_feature_names = ['ndcg@1', 'ndcg@3', 'ndcg@5', 'precision@1', 'precision@3', 'precision@5', 'recall@1', 'recall@3', 'recall@5']
                qpp_tensor = torch.tensor([qpp_features.get(name, 0.0) for name in qpp_feature_names], dtype=torch.float32)
                conversation_content["qpp_features"].append(qpp_tensor)
                
                # コンテキストを更新
                context_list.append(turn["user_utterance"])
                assert len(context_list) == turn_index * 2 + 1

                context_text = " ".join(context_list)

                context_tokens = self.tokenizer.tokenize(context_text)

                # コンテキストウィンドウサイズに調整
                if len(context_tokens) > (self.args.max_context_len - 2):
                    context_tokens_ = context_tokens[-(self.args.max_context_len - 2):]  # 510 tokens
                    context_tokens_ = ['[CLS]'] + context_tokens_ + ['[SEP]']
                else:
                    context_tokens_ = ['[CLS]'] + context_tokens + ['[SEP]'] + ['[PAD]'] * (self.args.max_context_len - 2 - len(context_tokens))

                assert len(context_tokens_) == self.args.max_context_len

                context_id = self.tokenizer.convert_tokens_to_ids(context_tokens_)
                conversation_content["context"].append(torch.tensor(context_id))
                context_list.append(turn["system_utterance"])

            assert len(conversation_content["user_utterance"]) == len(conversation_content["user_I_label"]) == len(
                conversation_content["system_utterance"]) == len(conversation_content["system_I_label"])== len(
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
                    qpp_features_conversation
                ]
            )
            self.len = conversation_index + 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        conversation_tensor = self.conversations_tensor[idx]
        return [conversation_tensor[0], conversation_tensor[1], conversation_tensor[2], conversation_tensor[3], conversation_tensor[4], conversation_tensor[5], conversation_tensor[6]]

def collate_fn(data):
    turn_id, user_utterance_conversations, user_I_label_conversations, system_utterance_conversations, system_I_label_conversations, context_conversations, qpp_features_conversations = zip(*data)
    return {
        "turn_id": turn_id[-1], # [batch_size, 1]
        "user_utterance": torch.stack(user_utterance_conversations), # [batch_size, ?, max_utterance_len]
        "user_I_label": torch.stack(user_I_label_conversations), # [batch_size, ?, 1]
        "system_utterance": torch.stack(system_utterance_conversations), # [batch_size, ?, max_utterance_len]
        "system_I_label": torch.stack(system_I_label_conversations), # [batch_size, ?, 1]
        "context": torch.stack(context_conversations), # [batch_size, ?, max_context_len]
        "qpp_features": torch.stack(qpp_features_conversations) # [batch_size, ?, 9]
    }