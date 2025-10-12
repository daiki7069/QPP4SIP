import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
import torch.nn.functional as F

from .music_model import DistanceCRF, PosteriorConversationEncoding, PriorConversationEncoding
from utils.math_utils import universal_sentence_embedding
from model.focal_loss import create_focal_loss_for_sip


class PolicyGating(nn.Module):
    def __init__(self, args, qpp_size=1):
        super().__init__()
        self.args = args
        self.qpp_index = args.qpp_index
        self.f_project = nn.Linear(qpp_size, args.hidden_size) # f_tをh_tと同じ次元に変換する層
        self.g_project = nn.Linear(args.hidden_size + qpp_size, args.hidden_size) # h_tとf_tを結合してh_tと同じ次元に変換する層

    def forward(self, h_t, f_t):
        """
        h_tにgateを用いてqpp特徴量f_tをゲーティング結合する(CRF制約のためbatch_sizeは1のみサポート)
        Args:
            h_t: [batch_size, hidden_size] ターンごとの隠れ状態 * batch_size
            f_t: [batch_size, qpp_size] ターンごとのQPP特徴量 * batch_size
        Returns:
            h_fused: [batch_size, hidden_size] 現在のターンのベクトルにqpp特徴量f_tを結合したベクトル * batch_size
            g_t: [batch_size, hidden_size] 現在のターンのゲート値ベクトル * batch_size
            g_t_mean: [batch_size] 現在のターンのゲート値の平均値 * batch_size
        """
        f_projected = self.f_project(f_t)
        gate_input = torch.cat([h_t, f_projected], dim=-1)
        g_t = torch.sigmoid(self.g_project(gate_input)) # [batch_size, hidden_size]

        h_fused = g_t * h_t + (1 - g_t) * f_projected
        g_t_mean = g_t.mean(dim=-1)
        return h_fused, g_t, g_t_mean


class GatedUtteranceEncoding(nn.Module):
    """
    Utterance Encoding with BERT and Policy Gating
    Encode user and system utterances at the sentence (utterance) level.
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.enc = BertModel.from_pretrained('bert-base-uncased')
        self.policy_gating = PolicyGating(args=args, qpp_size=1)    # TODO: argsに統合

    def forward(self, data):
        batch_size, conversation_len, max_utterance_len = data['user_utterance'].size()

        user_utterance = data['user_utterance'] # [batch, ?, max_utterance_len]
        user_utterance = user_utterance.reshape(-1, max_utterance_len)  # [batch * ?, max_utterance_len]
        user_utterance_mask = user_utterance.ne(0).detach()  # [batch * ?, max_utterance_len]

        system_utterance = data['system_utterance'] # [batch, ?, max_utterance_len]
        system_utterance = system_utterance.reshape(-1, max_utterance_len)  # [batch * ?, max_utterance_len]
        system_utterance_mask = system_utterance.ne(0).detach()  # [batch * ?, max_utterance_len]

        encoded_user_utterance = self.enc(user_utterance, attention_mask=user_utterance_mask.float())[0]    # [batch * ?, max_utterance_len, hidden_size]
        pooling_user_utterance = universal_sentence_embedding(encoded_user_utterance, user_utterance_mask)  # [batch * ?, hidden_size] 単語ベクトル->文単位のベクトル
        pooling_user_utterance /= np.sqrt(pooling_user_utterance.size()[-1])    # [batch * ?, hidden_size]
        pooling_user_utterance, g_t, g_t_mean = self.policy_gating(pooling_user_utterance, data['user_qpp'])  # [batch * ?, hidden_size], [batch * ?, 1], [batch * ?, 1] # TODO: user_qppの定義

        encoded_system_utterance = self.enc(system_utterance, attention_mask=system_utterance_mask.float())[0]    # [batch * ?, max_utterance_len, hidden_size]
        pooling_system_utterance = universal_sentence_embedding(encoded_system_utterance, system_utterance_mask)  # [batch * ?, hidden_size] 単語ベクトル->文単位のベクトル
        pooling_system_utterance /= np.sqrt(pooling_system_utterance.size()[-1])    # [batch * ?, hidden_size]
        # pooling_system_utterance, g_t, g_t_mean = self.policy_gating(pooling_system_utterance, data['system_qpp'])  # [batch * ?, hidden_size], [batch * ?, 1], [batch * ?, 1] # TODO: system_qppの定義

        return pooling_user_utterance.reshape(batch_size, conversation_len, -1), pooling_system_utterance.reshape(batch_size, conversation_len, -1)  # [batch, ?, hidden_size], [batch, ?, hidden_size]


class GatedBILSTMCRF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.gated_utterance_encoding=GatedUtteranceEncoding(args=args)
        self.posterior_conversation_encoding = PosteriorConversationEncoding(args=args)
        self.prior_conversation_encoding = PriorConversationEncoding(args=args)
        self.distance_crf = DistanceCRF(args=args)
        self.prior_e_project = nn.Linear(2 * self.args.hidden_size, 2)
        self.posterior_e_project = nn.Linear(2 * self.args.hidden_size, 2)

        # Focal Loss for handling class imbalance
        self.focal_loss = create_focal_loss_for_sip(
            class_imbalance_ratio=getattr(args, 'class_imbalance_ratio', 6.5),
            gamma=getattr(args, 'focal_gamma', 2.0)
        )

    def forward(self, data):
        pooling_user_utterance, pooling_system_utterance = self.gated_utterance_encoding(data)    # [batch=1, ?, hidden_size], [batch=1, ?, hidden_size]
        batch_size, pair_num, hidden_size = pooling_user_utterance.size()
        # バッチサイズは設計上1のみサポート（CRFの制約のため）

        previous_utterance_sequence = []
        previous_I_label_sequence = []

        logger ={"role":[], "system_I":[]}

        I_label_sequence_batch = []

        if self.args.mode == 'train':
            prior_emission_score_batch = []
            posterior_emission_score_batch = []

            gold_score_batch = []
            total_score_batch = []

        elif self.args.mode == 'inference':
            predicted_path_batch = []
            predicted_path_batch_from_emission = []
            emission_scores_batch = []  # 予測確率を保存するためのリスト

        # traverse all turns (user-system pairs) in a conversation
        for i in range(pair_num):
            # batch size is always one
            previous_utterance_sequence.append(pooling_user_utterance[:, i, :].unsqueeze(1))  # add user's utterance [1, 1, hidden_size] -> 直前のターンの隠れ状態
            prior_utterance_sequence = torch.cat(previous_utterance_sequence, 1)  # [1, 2i+1, hidden_size] -> ユーザーのターンまでの隠れ状態系列
            assert prior_utterance_sequence.shape[1]==2*i+1

            logger["role"].append("user")
            logger["system_I"].append(0)
            
            previous_utterance_sequence.append(pooling_system_utterance[:, i, :].unsqueeze(1))  # add corresponding system's utterance [1, 1, hidden_size] -> 直前のターンの隠れ状態
            posterior_utterance_sequence = torch.cat(previous_utterance_sequence, 1)  # [1, 2i+2, hidden_size] -> システムのターンまでの隠れ状態系列
            assert posterior_utterance_sequence.shape[1] == 2*(i+1)

            logger["role"].append("system")
            logger["system_I"].append(data['response_type'][:, i].squeeze().item())  # add 1 or 0

            assert len(logger["role"]) == len(logger["system_I"]) == 2 * (i + 1)
            
            previous_I_label_sequence.append(data['query_type'][:, i].unsqueeze(1))  # add user's utterance I label [1, 1]
            previous_I_label_sequence.append(data['response_type'][:, i].unsqueeze(1))  # add system's utterance I label [1,1]
            I_label_sequence = torch.cat(previous_I_label_sequence, 1)  # [1, 2i+2]
            assert I_label_sequence.shape[1] == 2 * (i + 1)

            I_label_sequence_batch.append(I_label_sequence.squeeze().tolist())  # [[2], [4], ...] only used for inference

            state = {"who2who": [], "position": [], "Intime": [], "Distance": [], "overall": []}
            
            for turn_index in range(len(logger["role"])):
                if turn_index == 0:
                    state["who2who"].append(-1)
                    state["position"].append(-1)
                    state["Intime"].append(-1)
                    state["Distance"].append(-1)
                    state["overall"].append(-1)
                else:
                    state["overall"].append(0)

                    if turn_index <= 19:
                        state["position"].append(turn_index - 1)
                    else:
                        state["position"].append(-1)

                    if logger["role"][turn_index - 1] == "system":
                        # we don't concentrate on this perspective
                        state["who2who"].append(1)
                        state["Intime"].append(-1)
                        state["Distance"].append(-1)

                    else:
                        # user to system | the last one is user
                        state["who2who"].append(0)
                        I_times = sum(logger["system_I"][0:turn_index - 1])
                        if I_times == 0:
                            state["Intime"].append(0)
                            state["Distance"].append(-1)
                        else:
                            if I_times == 1:
                                state["Intime"].append(1)
                            else:
                                state["Intime"].append(1)

                            last_system_I_turn = -1
                            for turn_index_ in range(len(logger["system_I"][0:turn_index - 1])):
                                if logger["system_I"][turn_index_] == 1:
                                    last_system_I_turn = turn_index_

                            distance = turn_index - last_system_I_turn

                            if distance == 2:
                                state["Distance"].append(0)
                            else:
                                state["Distance"].append(1)

            assert len(state["Distance"]) == len(state["Intime"]) == len(state["who2who"]) == len(
                state["position"]) == len(state["overall"]) == len(logger["role"]) == len(logger["system_I"])

            prior_hidden_sequence = self.prior_conversation_encoding(prior_utterance_sequence)
            prior_emission_score = self.prior_e_project(prior_hidden_sequence[:, -1, :])

            if self.args.mode == 'train':
                # obtain emission score
                posterior_hidden_sequence = self.posterior_conversation_encoding(posterior_utterance_sequence)
                posterior_emission_scores = self.posterior_e_project(posterior_hidden_sequence)

                gold_score, total_score= self.distance_crf(posterior_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_sequence[:, -1, :].squeeze(0), posterior_hidden_sequence.squeeze(0), state)

                gold_score_batch.append(gold_score.unsqueeze(0))
                total_score_batch.append(total_score.unsqueeze(0))
                
                prior_emission_score_batch.append(prior_emission_score.unsqueeze(1))
                posterior_emission_score_batch.append(posterior_emission_scores[:, -1, :].unsqueeze(1))

        # End of training cycle
            elif self.args.mode == 'inference':
                partial_posterior_hidden_sequence = self.posterior_conversation_encoding(prior_utterance_sequence)  #  [1, 2i+1, 2*hidden_size_BiLSTM]
                partial_posterior_emission_scores = self.posterior_e_project(partial_posterior_hidden_sequence)  # [1, 2i+1, 2]

                combined_emission_scores = torch.cat([partial_posterior_emission_scores, prior_emission_score.unsqueeze(1)], 1)  # [1, 2i+2, 2]
                predicted_path = self.distance_crf(combined_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_sequence[:, -1, :].squeeze(0), partial_posterior_hidden_sequence.squeeze(0) , state)
                assert len(predicted_path) == combined_emission_scores.shape[1] == (partial_posterior_emission_scores.shape[1] + 1) == (partial_posterior_hidden_sequence.shape[1]+1) == (prior_hidden_sequence.shape[1]+1)
                predicted_path_batch.append(predicted_path) # [[2], [4], ...]
                predicted_path_batch_from_emission.append(combined_emission_scores.squeeze(0).max(1)[1].tolist())  # [1, 2i+2, 2] --> [2i+2, 2] -->[2i+2]
                
        if self.args.mode == 'train':
            gold_score_tensor = torch.cat(gold_score_batch)  # [pair_num]
            total_score_tensor = torch.cat(total_score_batch)  # [pair_num]

            prior_emission_score_tensor = torch.cat(prior_emission_score_batch, 1).squeeze(0)  # [pair_num, 2]
            posterior_emission_score_tensor = torch.cat(posterior_emission_score_batch, 1).squeeze(0)  # [pair_num,2]

            assert pair_num == prior_emission_score_tensor.shape[0] == posterior_emission_score_tensor.shape[0]
            assert prior_emission_score_tensor.shape[1] == posterior_emission_score_tensor.shape[1] # 2

            loss_distance_crf = torch.mean(total_score_tensor - gold_score_tensor) # average each sample
            loss_mle_e = F.mse_loss(prior_emission_score_tensor, posterior_emission_score_tensor.detach())  # [pair_num, 2]
            
            # Focal Loss for emission scores (prior network)
            # Convert labels to appropriate format for focal loss
            # We'll use the system I labels for focal loss calculation
            system_labels = data['response_type'].squeeze(0)  # [pair_num]
            focal_loss_value = self.focal_loss(prior_emission_score_tensor, system_labels)

            return {"loss_distance_crf": loss_distance_crf, "loss_mle_e": loss_mle_e, "loss_focal": focal_loss_value}

        elif self.args.mode == 'inference':
            assert len(predicted_path_batch[0])==len(predicted_path_batch_from_emission[0])==len(I_label_sequence_batch[0])==2
            
            if len(predicted_path_batch)>1:
                assert len(predicted_path_batch[1])==len(predicted_path_batch_from_emission[1])==len(I_label_sequence_batch[1])==4

            return predicted_path_batch