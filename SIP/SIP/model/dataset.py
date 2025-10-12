from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer

class Dataset(Dataset):
    def __init__(self, args, conversations):
        super(Dataset, self).__init__()
        self.args = args
        self.conversations = conversations
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.conversations_tensor = []
        self.load()

    def load(self):
        for conversation_index, conversation in enumerate(self.conversations):
            # TODO: 空の会話をスキップとは？
            if not conversation or len(conversation) == 0:
                print(f"Warning: Empty conversation at index {conversation_index}, skipping...")
                continue
                
            conversation_content = { "turn_id": [], "user_utterance": [], "system_utterance": [], "response_type": [],
                                    "context": [], "mrr": [], "found_ratio": [],
                                    "mean_rank": [], "hit@1": [], "hit@5": [], "hit@10": [],
                                    "hit@20": [], "hit@50": [], "precision@1": [], "precision@5": [],
                                    "precision@10": [], "precision@20": [], "precision@50": [],
                                    "recall@1": [], "recall@5": [], "recall@10": [], "recall@20": [], "recall@50": [],
                                    "f1@1": [], "f1@5": [], "f1@10": [], "f1@20": [], "f1@50": [],
                                    "ndcg@1": [], "ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "ndcg@50": [], 
                                    "response_type_prediction": [], "query_type": [] }

            context_list = []

            for turn_index, turn in enumerate(conversation):
                conversation_content["turn_id"].append(f"{conversation_index}_{turn['turn_id']}")
                conversation_content["user_utterance"].append(torch.tensor(self.tokenizer.encode(turn["query"], add_special_tokens=True, max_length=self.args.max_utterance_len, padding="max_length", truncation=True)))
                # answerがリストの場合は最初の要素を取得
                answer_text = turn["answer"][0] if isinstance(turn["answer"], list) and len(turn["answer"]) > 0 else str(turn["answer"])
                conversation_content["system_utterance"].append(torch.tensor(self.tokenizer.encode(answer_text, add_special_tokens=True, max_length=self.args.max_utterance_len, padding="max_length", truncation=True)))
                conversation_content["response_type"].append(torch.tensor(1) if "clarification" in turn["response_type"] else torch.tensor(0))
                conversation_content["mrr"].append(torch.tensor(turn["mrr"]))
                conversation_content["found_ratio"].append(torch.tensor(turn["found_ratio"]))
                conversation_content["mean_rank"].append(torch.tensor(turn["mean_rank"]))
                conversation_content["hit@1"].append(torch.tensor(turn["hit@1"]))
                conversation_content["hit@5"].append(torch.tensor(turn["hit@5"]))
                conversation_content["hit@10"].append(torch.tensor(turn["hit@10"]))
                conversation_content["hit@20"].append(torch.tensor(turn["hit@20"]))
                conversation_content["hit@50"].append(torch.tensor(turn["hit@50"]))
                conversation_content["precision@1"].append(torch.tensor(turn["precision@1"]))
                conversation_content["precision@5"].append(torch.tensor(turn["precision@5"]))
                conversation_content["precision@10"].append(torch.tensor(turn["precision@10"]))
                conversation_content["precision@20"].append(torch.tensor(turn["precision@20"]))
                conversation_content["precision@50"].append(torch.tensor(turn["precision@50"]))
                conversation_content["recall@1"].append(torch.tensor(turn["recall@1"]))
                conversation_content["recall@5"].append(torch.tensor(turn["recall@5"]))
                conversation_content["recall@10"].append(torch.tensor(turn["recall@10"]))
                conversation_content["recall@20"].append(torch.tensor(turn["recall@20"]))
                conversation_content["recall@50"].append(torch.tensor(turn["recall@50"]))
                conversation_content["f1@1"].append(torch.tensor(turn["f1@1"]))
                conversation_content["f1@5"].append(torch.tensor(turn["f1@5"]))
                conversation_content["f1@10"].append(torch.tensor(turn["f1@10"]))
                conversation_content["f1@20"].append(torch.tensor(turn["f1@20"]))
                conversation_content["f1@50"].append(torch.tensor(turn["f1@50"]))
                conversation_content["ndcg@1"].append(torch.tensor(turn["ndcg@1"]))
                conversation_content["ndcg@5"].append(torch.tensor(turn["ndcg@5"]))
                conversation_content["ndcg@10"].append(torch.tensor(turn["ndcg@10"]))
                conversation_content["ndcg@20"].append(torch.tensor(turn["ndcg@20"]))
                conversation_content["ndcg@50"].append(torch.tensor(turn["ndcg@50"]))
                conversation_content["query_type"].append(torch.tensor(0))  # FIXME: デフォルトでNon-initiative

                conversation_content["response_type_prediction"].append(torch.tensor(1) if "clarification" in turn["response_type"] else torch.tensor(0))   # FIXME

                context_list.append(turn["query"])
                assert len(context_list) == turn_index * 2 + 1

                context_text = " ".join(context_list)

                context_tokens = self.tokenizer.tokenize(context_text)

                if len(context_tokens) > (self.args.max_context_len - 2):
                    context_tokens_ = context_tokens[-(self.args.max_context_len - 2):]  # 510 tokens
                    context_tokens_ = ['[CLS]'] + context_tokens_ + ['[SEP]']
                else:
                    context_tokens_ = ['[CLS]'] + context_tokens + ['[SEP]'] + ['[PAD]'] * (self.args.max_context_len - 2 - len(context_tokens))

                assert len(context_tokens_) == self.args.max_context_len

                context_id = self.tokenizer.convert_tokens_to_ids(context_tokens_)
                conversation_content["context"].append(torch.tensor(context_id))
                # answerがリストの場合は最初の要素を取得
                answer_text = turn["answer"][0] if isinstance(turn["answer"], list) and len(turn["answer"]) > 0 else str(turn["answer"])
                context_list.append(answer_text)

            assert len(conversation_content["user_utterance"]) == len(conversation_content["system_utterance"]) == len(conversation_content["response_type"]) == len(conversation_content["context"])

            user_utterance_conversation = torch.stack(conversation_content["user_utterance"])
            system_utterance_conversation = torch.stack(conversation_content["system_utterance"])
            response_type_conversation = torch.stack(conversation_content["response_type"])
            context_conversation = torch.stack(conversation_content["context"])
            response_type_prediction_conversation = torch.stack(conversation_content["response_type_prediction"])
            query_type_conversation = torch.stack(conversation_content["query_type"])
            qpp_feature_conversation = torch.stack(conversation_content[self.args.qpp_feature_name])    # TODO: 1特徴量のみ対応


            self.conversations_tensor.append(
                [
                    conversation_content["turn_id"],
                    user_utterance_conversation,
                    system_utterance_conversation,
                    response_type_conversation,
                    context_conversation,
                    response_type_prediction_conversation,
                    qpp_feature_conversation,
                    query_type_conversation
                ]
            )

            self.len = conversation_index + 1
            

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        conversation_tensor = self.conversations_tensor[index]
        return [conversation_tensor[0], conversation_tensor[1], conversation_tensor[2], conversation_tensor[3], conversation_tensor[4], conversation_tensor[5], conversation_tensor[6], conversation_tensor[7]]

def collate_fn(data):
    turn_id, user_utterance_conversations, system_utterance_conversations, response_type_conversations, context_conversations, response_type_prediction_conversations, qpp_feature_conversations, query_type_conversations = zip(*data)
    
    # デバイスを取得（CUDAが利用可能な場合はCUDAを使用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        "turn_id": turn_id[-1],
        "user_utterance": torch.stack(user_utterance_conversations),
        "system_utterance": torch.stack(system_utterance_conversations),
        "response_type": torch.stack(response_type_conversations),
        "context": torch.stack(context_conversations),
        "response_type_prediction": torch.stack(response_type_prediction_conversations),
        "qpp_feature": torch.stack(qpp_feature_conversations),
        "query_type": torch.stack(query_type_conversations)
    }