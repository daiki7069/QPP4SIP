from model.utils import universal_sentence_embedding
from model.focal_loss import create_focal_loss_for_sip
import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn.functional as F

class utterance_encoding(nn.Module):
    """
    Utterance Encoding with BERT
    Encode user and system utterances at the sentence (utterance) level.
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        
        self.enc = BertModel.from_pretrained('bert-base-uncased')

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

        encoded_system_utterance = self.enc(system_utterance, attention_mask=system_utterance_mask.float())[0]    # [batch * ?, max_utterance_len, hidden_size]
        pooling_system_utterance = universal_sentence_embedding(encoded_system_utterance, system_utterance_mask)  # [batch * ?, hidden_size] 単語ベクトル->文単位のベクトル
        pooling_system_utterance /= np.sqrt(pooling_system_utterance.size()[-1])    # [batch * ?, hidden_size]

        return pooling_user_utterance.reshape(batch_size, conversation_len, -1), pooling_system_utterance.reshape(batch_size, conversation_len, -1)  # [batch, ?, hidden_size], [batch, ?, hidden_size]

class posterior_conversation_encoding(nn.Module):
    """
    Conversation Encoding with BiLSTM for posterior network
    Encode the entire conversation up to the current turn.
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.lstm = nn.LSTM(self.args.hidden_size, self.args.hidden_size, dropout=self.args.dropout, num_layers=self.args.BiLSTM_layers, bidirectional=True, batch_first=True)

    def forward(self, input):
        output, (_,_) = self.lstm(input) # [batch_size, ?, hidden_size*2]
        return output   # [batch_size, ?, hidden_size*2]

class prior_conversation_encoding(nn.Module):
    """
    Conversation Encoding with BiLSTM for prior network
    Encode the entire conversation up to the current turn.
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.lstm = nn.LSTM(self.args.hidden_size, self.args.hidden_size, dropout=self.args.dropout, num_layers=self.args.BiLSTM_layers, bidirectional=True, batch_first=True)

    def forward(self, input):
        output, (_,_)= self.lstm(input) # [batch_size, ?, hidden_size*2]
        return output   # [batch_size, ?, hidden_size*2]

class qpp_feature_fusion(nn.Module):
    """
    QPP特徴量融合モジュール
    BERTの[CLS]表現にQPP特徴量を結合してMLPで埋め込み
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.qpp_feature_dim = 9  # ndcg@1, ndcg@3, ndcg@5, precision@1, precision@3, precision@5, recall@1, recall@3, recall@5
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.args.hidden_size + self.qpp_feature_dim, self.args.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.hidden_size, self.args.hidden_size)
        )

    def forward(self, bert_representation, qpp_features):
        """
        Args:
            bert_representation: [batch_size, hidden_size] BERTの[CLS]表現
            qpp_features: [batch_size, 9] QPP特徴量
        Returns:
            fused_representation: [batch_size, hidden_size] 融合された表現
        """
        # BERT表現とQPP特徴量を結合
        combined = torch.cat([bert_representation, qpp_features], dim=-1)
        # MLPで埋め込み
        fused_representation = self.fusion_mlp(combined)
        return fused_representation

class qpp_auxiliary_head(nn.Module):
    """
    QPP補助ヘッド（マルチタスク学習用）
    ndcg@3を予測する回帰ヘッド
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.qpp_head = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.hidden_size // 2, 1)  # ndcg@3の予測
        )

    def forward(self, bert_representation):
        """
        Args:
            bert_representation: [batch_size, hidden_size] BERTの[CLS]表現
        Returns:
            qpp_prediction: [batch_size, 1] ndcg@3の予測値
        """
        return self.qpp_head(bert_representation)

class qpp_policy_gating(nn.Module):
    """
    QPPポリシー連動モジュール
    QPPスコアに基づいてイニシアチブ発動の閾値を調整
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.qpp_gate = nn.Sequential(
            nn.Linear(1, 16),  # ndcg@3を入力
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1の範囲でゲート値を出力
        )

    def forward(self, qpp_score):
        """
        Args:
            qpp_score: [batch_size, 1] ndcg@3スコア
        Returns:
            gate_value: [batch_size, 1] ゲート値（0-1）
        """
        return self.qpp_gate(qpp_score)

class qpp4sip_crf(nn.Module):
    """
    QPP4SIP用のCRF実装
    3つのパターンに対応
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        # 基本遷移行列
        self.matrice_all = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.matrice_u2s = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.matrice_s2u = nn.Parameter(torch.randn(2, 2) * 0.1)

        # QPP4SIP特有の遷移行列
        self.matrice_qpp_feature = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.matrice_qpp_policy = nn.Parameter(torch.randn(2, 2) * 0.1)

    def forward(self, emission_scores, label, prior, posterior_sequence, state, qpp_gate=None):
        """
        QPP4SIP用のCRF forward pass
        """
        conversation_len = emission_scores.shape[0]

        assert emission_scores.shape[0] == label.shape[0] 
        assert conversation_len % 2 == 0

        if self.args.mode=="train":
            assert posterior_sequence.shape[0] % 2==0
        elif self.args.mode == "inference":
            assert posterior_sequence.shape[0] % 2==1
            
        padding_matrice = torch.zeros(2, 2, device=emission_scores.device)
        
        # 遷移行列の組み合わせ
        who2who_subsidiary = torch.stack([self.matrice_u2s, self.matrice_s2u, padding_matrice],0)  # [3, 2, 2]
        qpp_subsidiary = torch.stack([self.matrice_qpp_feature, self.matrice_qpp_policy, padding_matrice], 0) # [3, 2, 2]
        overall_subsidiary = torch.stack([self.matrice_all, padding_matrice],0)  # [2, 2, 2]
        bank = [who2who_subsidiary, qpp_subsidiary, overall_subsidiary]  # [3, x, 2, 2]

        if self.args.mode=="train":
            # obtain golden score
            front_pointers= label[:-1]  # remove the last label
            back_pointers = label[1:]  # remove the first label
            assert front_pointers.shape[0]==back_pointers.shape[0]==conversation_len-1

            sum_emission_score = torch.sum(emission_scores[range(conversation_len), label])

            sum_transition_score =0
            for index in range(conversation_len):
                if index > 0: # sanity check
                    if self.args.qpp4sip_pattern == "feature_fusion":
                        combined_matrice = bank[0][state["who2who"][index]] + bank[1][state["qpp_feature"][index]]
                    elif self.args.qpp4sip_pattern == "auxiliary_head":
                        combined_matrice = bank[0][state["who2who"][index]] + bank[2][state["overall"][index]]
                    elif self.args.qpp4sip_pattern == "policy_gating":
                        # QPPポリシー連動の場合、ゲート値を考慮
                        if qpp_gate is not None and index < len(qpp_gate):
                            gate_value = qpp_gate[index]
                            combined_matrice = bank[0][state["who2who"][index]] + gate_value * bank[1][state["qpp_policy"][index]]
                        else:
                            combined_matrice = bank[0][state["who2who"][index]]
                    else:
                        # デフォルトは基本CRF
                        combined_matrice = bank[2][state["overall"][index]]

                    sum_transition_score += combined_matrice[front_pointers[index-1], back_pointers[index-1]]

            gold_score = sum_emission_score + sum_transition_score

            # obtain total score
            alpha = torch.full((1, 2), 0.0, device=emission_scores.device) # [1, 2]
            for index in range(conversation_len):
                if self.args.qpp4sip_pattern == "feature_fusion":
                    combined_matrice = bank[0][state["who2who"][index]] + bank[1][state["qpp_feature"][index]]
                elif self.args.qpp4sip_pattern == "auxiliary_head":
                    combined_matrice = bank[0][state["who2who"][index]] + bank[2][state["overall"][index]]
                elif self.args.qpp4sip_pattern == "policy_gating":
                    if qpp_gate is not None and index < len(qpp_gate):
                        gate_value = qpp_gate[index]
                        combined_matrice = bank[0][state["who2who"][index]] + gate_value * bank[1][state["qpp_policy"][index]]
                    else:
                        combined_matrice = bank[0][state["who2who"][index]]
                else:
                    combined_matrice = bank[2][state["overall"][index]]

                if index ==0:
                    assert torch.equal(combined_matrice, torch.zeros(2,2, device=emission_scores.device).int())

                alpha = torch.logsumexp(alpha.T + emission_scores[index].unsqueeze(0) + combined_matrice, dim=0, keepdim=True)  # [1, 2]  row vector

            total_score = torch.logsumexp(alpha.T, dim=0).squeeze()

            return gold_score, total_score

        elif self.args.mode=="inference":
            backtrace = []
            alpha = torch.full((1, 2), 0.0, device= emission_scores.device) # [1, 2]  row vector

            for index in range(conversation_len):
                if self.args.qpp4sip_pattern == "feature_fusion":
                    combined_matrice = bank[0][state["who2who"][index]] + bank[1][state["qpp_feature"][index]]
                elif self.args.qpp4sip_pattern == "auxiliary_head":
                    combined_matrice = bank[0][state["who2who"][index]] + bank[2][state["overall"][index]]
                elif self.args.qpp4sip_pattern == "policy_gating":
                    if qpp_gate is not None and index < len(qpp_gate):
                        gate_value = qpp_gate[index]
                        combined_matrice = bank[0][state["who2who"][index]] + gate_value * bank[1][state["qpp_policy"][index]]
                    else:
                        combined_matrice = bank[0][state["who2who"][index]]
                else:
                    combined_matrice = bank[2][state["overall"][index]]

                if index == 0:
                    assert torch.equal(combined_matrice, torch.zeros(2, 2, device=emission_scores.device).int())

                alpha = alpha.T + emission_scores[index].unsqueeze(0) + combined_matrice  # [2, 2]

                viterbivars_t, bptrs_t = torch.max(alpha, dim=0) # [2], [2]

                backtrace.append(bptrs_t)
                alpha = viterbivars_t.unsqueeze(0)  # [1, 2]  row vector

            # backtrack
            best_tag_id = alpha.flatten().argmax().item()
            best_path = [best_tag_id]

            assert torch.equal(backtrace[0], torch.zeros(2, device=emission_scores.device).int())

            for bptrs_t in reversed(backtrace[1:]):  # ignore the first one
                best_tag_id = bptrs_t[best_tag_id].item()
                best_path.append(best_tag_id)

            best_path.reverse()

            assert len(best_path) % 2 == 0
            assert len(best_path) == conversation_len

            return best_path

class QPP4SIPBILSTMCRF(nn.Module):
    """
    QPP4SIP用のBILSTM-CRFモデル
    3つの実装パターンに対応:
    1) feature_fusion: 追加特徴（Feature Fusion）
    2) auxiliary_head: マルチタスク学習（Aux Head）
    3) policy_gating: ポリシー連動（Decision/Gating）
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.utterance_encoding=utterance_encoding(args=args)
        self.posterior_conversation_encoding = posterior_conversation_encoding(args=args)
        self.prior_conversation_encoding = prior_conversation_encoding(args=args)
        self.crf = qpp4sip_crf(args=args)
        self.prior_e_project = nn.Linear(2 * self.args.hidden_size, 2)
        self.posterior_e_project = nn.Linear(2 * self.args.hidden_size, 2)

        # QPP4SIP特有のモジュール
        if self.args.qpp4sip_pattern == "feature_fusion":
            self.qpp_feature_fusion = qpp_feature_fusion(args=args)
        elif self.args.qpp4sip_pattern == "auxiliary_head":
            self.qpp_auxiliary_head = qpp_auxiliary_head(args=args)
        elif self.args.qpp4sip_pattern == "policy_gating":
            self.qpp_policy_gating = qpp_policy_gating(args=args)
        
        # Focal Loss for handling class imbalance
        self.focal_loss = create_focal_loss_for_sip(
            class_imbalance_ratio=getattr(args, 'class_imbalance_ratio', 6.5),
            gamma=getattr(args, 'focal_gamma', 2.0)
        )

    def forward(self, data):
        # pooling_user_utterance [batch=1, ?, hidden_size]
        # pooling_system_utterance [batch=1, ?, hidden_size]
        pooling_user_utterance, pooling_system_utterance = self.utterance_encoding(data)
        batch_size, pair_num, hidden_size = pooling_user_utterance.size()

        # QPP特徴量を取得
        qpp_features = data.get('qpp_features', None)  # [batch_size, ?, 9]

        previous_utterance_sequence = []
        previous_I_label_sequence = []

        logger ={"role":[], "system_I":[]}

        I_label_sequence_batch = []

        # QPP補助損失用（推論モードでも初期化）
        qpp_loss_batch = []

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
            previous_utterance_sequence.append(pooling_user_utterance[:, i, :].unsqueeze(1))  # add user's utterance [1, 1, hidden_size]
            prior_utterance_sequence = torch.cat(previous_utterance_sequence, 1)  # [1, 2i+1, hidden_size]
            assert prior_utterance_sequence.shape[1]==2*i+1

            logger["role"].append("user")
            logger["system_I"].append(0)

            previous_utterance_sequence.append(pooling_system_utterance[:, i, :].unsqueeze(1))  # add corresponding system's utterance [1, 1, hidden_size]
            posterior_utterance_sequence = torch.cat(previous_utterance_sequence, 1)  # [1, 2i+2, hidden_size]
            assert posterior_utterance_sequence.shape[1] == 2*(i+1)

            logger["role"].append("system")
            logger["system_I"].append(data['system_I_label'][:, i].squeeze().item())  # add 1 or 0

            assert len(logger["role"]) == len(logger["system_I"]) == 2 * (i + 1)

            previous_I_label_sequence.append(data['user_I_label'][:, i].unsqueeze(1))  # add user's utterance I label [1, 1]
            previous_I_label_sequence.append(data['system_I_label'][:, i].unsqueeze(1))  # add system's utterance I label [1,1]
            I_label_sequence = torch.cat(previous_I_label_sequence, 1)  # [1, 2i+2]
            assert I_label_sequence.shape[1] == 2 * (i + 1)

            I_label_sequence_batch.append(I_label_sequence.squeeze().tolist())  # [[2], [4], ...] only used for inference

            state = {"who2who": [], "qpp_feature": [], "qpp_policy": [], "overall": []}

            for turn_index in range(len(logger["role"])):
                if turn_index == 0:
                    state["who2who"].append(-1)
                    state["qpp_feature"].append(-1)
                    state["qpp_policy"].append(-1)
                    state["overall"].append(-1)
                else:
                    state["overall"].append(0)

                    if logger["role"][turn_index - 1] == "system":
                        # system to user
                        state["who2who"].append(1)
                        state["qpp_feature"].append(0)
                        state["qpp_policy"].append(0)
                    else:
                        # user to system
                        state["who2who"].append(0)
                        state["qpp_feature"].append(1)
                        state["qpp_policy"].append(1)

            assert len(state["qpp_feature"]) == len(state["qpp_policy"]) == len(state["who2who"]) == len(state["overall"]) == len(logger["role"]) == len(logger["system_I"])

            prior_hidden_squence = self.prior_conversation_encoding(prior_utterance_sequence)
            prior_emission_score = self.prior_e_project(prior_hidden_squence[:, -1, :])

            # QPP特徴量の処理
            qpp_gate = None
            if qpp_features is not None and i < qpp_features.shape[1]:
                current_qpp_features = qpp_features[:, i, :]  # [batch_size, 9]
                
                # デバッグ情報
                if torch.isnan(current_qpp_features).any():
                    print(f"警告: QPP特徴量にNaNが含まれています (ターン {i})")
                    print(f"QPP特徴量: {current_qpp_features}")
                    # NaNを0で置換
                    current_qpp_features = torch.where(torch.isnan(current_qpp_features), 
                                                     torch.zeros_like(current_qpp_features), 
                                                     current_qpp_features)
                
                # QPP特徴量の正規化（0-1の範囲にスケーリング）
                # 全て0の場合は正規化をスキップ
                if current_qpp_features.sum() > 0:
                    current_qpp_features = torch.clamp(current_qpp_features, 0.0, 1.0)
                
                if self.args.qpp4sip_pattern == "feature_fusion":
                    # 特徴融合: BERT表現にQPP特徴量を結合
                    bert_cls = prior_hidden_squence[:, -1, :]  # [batch_size, hidden_size*2]
                    # 最初の半分を取ってBERTの[CLS]表現として使用
                    bert_cls_half = bert_cls[:, :self.args.hidden_size]  # [batch_size, hidden_size]
                    
                    # QPP特徴量が全て0の場合は、小さなノイズを追加して学習を安定化
                    if current_qpp_features.sum() == 0:
                        current_qpp_features = torch.randn_like(current_qpp_features) * 0.01
                    
                    fused_representation = self.qpp_feature_fusion(bert_cls_half, current_qpp_features)
                    # 融合された表現を元の次元に戻す
                    prior_emission_score = self.prior_e_project(torch.cat([fused_representation, bert_cls[:, self.args.hidden_size:]], dim=-1))
                    
                    # QPP特徴量の正則化損失を追加（特徴量の品質向上）
                    feature_regularization = torch.mean(torch.norm(current_qpp_features, p=2, dim=1))
                    qpp_loss_batch.append(("feature_reg", feature_regularization * 0.01))  # 小さな重みで正則化
                    
                elif self.args.qpp4sip_pattern == "auxiliary_head":
                    # 補助ヘッド: QPP予測タスクを追加
                    bert_cls = prior_hidden_squence[:, -1, :]
                    bert_cls_half = bert_cls[:, :self.args.hidden_size]
                    qpp_prediction = self.qpp_auxiliary_head(bert_cls_half)
                    # ndcg@3の真値を取得（3番目の特徴量）
                    qpp_target = current_qpp_features[:, 1:2]  # ndcg@3
                    
                    # QPP特徴量が全て0の場合は、小さなノイズを追加
                    if current_qpp_features.sum() == 0:
                        qpp_target = torch.randn_like(qpp_target) * 0.01
                    
                    qpp_loss = F.mse_loss(qpp_prediction, qpp_target)
                    qpp_loss_batch.append(("qpp_prediction", qpp_loss))
                    
                elif self.args.qpp4sip_pattern == "policy_gating":
                    # ポリシー連動: QPPスコアに基づいてゲート値を計算
                    ndcg3_score = current_qpp_features[:, 1:2]  # ndcg@3
                    
                    # QPP特徴量が全て0の場合は、小さなノイズを追加
                    if current_qpp_features.sum() == 0:
                        ndcg3_score = torch.randn_like(ndcg3_score) * 0.01
                    
                    qpp_gate = self.qpp_policy_gating(ndcg3_score)
                    
                    # ゲート値の正則化損失を追加（ゲートの学習安定化）
                    gate_regularization = torch.mean(torch.norm(qpp_gate, p=2, dim=1))
                    qpp_loss_batch.append(("gate_reg", gate_regularization * 0.01))  # 小さな重みで正則化

            if self.args.mode == 'train':
                # obtain emission score
                posterior_hidden_squence = self.posterior_conversation_encoding(posterior_utterance_sequence)
                posterior_emission_scores = self.posterior_e_project(posterior_hidden_squence)

                # NaNチェック
                if torch.isnan(posterior_emission_scores).any():
                    print(f"警告: posterior_emission_scoresにNaNが含まれています (ターン {i})")
                    posterior_emission_scores = torch.where(torch.isnan(posterior_emission_scores), 
                                                           torch.zeros_like(posterior_emission_scores), 
                                                           posterior_emission_scores)
                
                if torch.isnan(prior_emission_score).any():
                    print(f"警告: prior_emission_scoreにNaNが含まれています (ターン {i})")
                    prior_emission_score = torch.where(torch.isnan(prior_emission_score), 
                                                      torch.zeros_like(prior_emission_score), 
                                                      prior_emission_score)

                gold_score, total_score= self.crf(posterior_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_squence[:, -1, :].squeeze(0), posterior_hidden_squence.squeeze(0), state, qpp_gate)

                gold_score_batch.append(gold_score.unsqueeze(0))
                total_score_batch.append(total_score.unsqueeze(0))

                prior_emission_score_batch.append(prior_emission_score.unsqueeze(1))
                posterior_emission_score_batch.append(posterior_emission_scores[:, -1, :].unsqueeze(1))

        # End of training cycle
            elif self.args.mode == 'inference':
                partial_posterior_hidden_squence = self.posterior_conversation_encoding(prior_utterance_sequence)  #  [1, 2i+1, 2*hidden_size_BiLSTM]
                partial_posterior_emission_scores = self.posterior_e_project(partial_posterior_hidden_squence)  # [1, 2i+1, 2]

                combined_emission_scores = torch.cat([partial_posterior_emission_scores, prior_emission_score.unsqueeze(1)], 1)  # [1, 2i+2, 2]
                predicted_path = self.crf(combined_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_squence[:, -1, :].squeeze(0), partial_posterior_hidden_squence.squeeze(0) , state, qpp_gate)
                assert len(predicted_path) == combined_emission_scores.shape[1] == (partial_posterior_emission_scores.shape[1] + 1) == (partial_posterior_hidden_squence.shape[1]+1) == (prior_hidden_squence.shape[1]+1)
                predicted_path_batch.append(predicted_path) # [[2], [4], ...]
                predicted_path_batch_from_emission.append(combined_emission_scores.squeeze(0).max(1)[1].tolist())  # [1, 2i+2, 2] --> [2i+2, 2] -->[2i+2]
                
                # 予測確率を保存（softmaxを適用）
                emission_probs = torch.softmax(combined_emission_scores.squeeze(0), dim=1)  # [2i+2, 2]
                emission_scores_batch.append(emission_probs)  # 各ターンの予測確率を保存

        if self.args.mode == 'train':
            gold_score_tensor = torch.cat(gold_score_batch)  # [pair_num]
            total_score_tensor = torch.cat(total_score_batch)  # [pair_num]

            prior_emission_score_tensor = torch.cat(prior_emission_score_batch, 1).squeeze(0)  # [pair_num, 2]
            posterior_emission_score_tensor = torch.cat(posterior_emission_score_batch, 1).squeeze(0)  # [pair_num,2]

            assert pair_num == prior_emission_score_tensor.shape[0] == posterior_emission_score_tensor.shape[0]
            assert prior_emission_score_tensor.shape[1] == posterior_emission_score_tensor.shape[1] # 2

            loss_crf = torch.mean(total_score_tensor - gold_score_tensor) # average each sample
            loss_mle_e = F.mse_loss(prior_emission_score_tensor, posterior_emission_score_tensor.detach())  # [pair_num, 2]
            
            # Focal Loss for emission scores (prior network)
            # Convert labels to appropriate format for focal loss
            # We'll use the system I labels for focal loss calculation
            system_labels = data['system_I_label'].squeeze(0)  # [pair_num]
            focal_loss_value = self.focal_loss(prior_emission_score_tensor, system_labels)

            result = {"loss_crf": loss_crf, "loss_mle_e": loss_mle_e, "loss_focal": focal_loss_value}
            
            # QPP関連損失を追加（重み付けを調整）
            if qpp_loss_batch:
                # 損失の種類ごとに処理
                qpp_losses = {}
                total_qpp_loss = 0
                
                for loss_type, loss_value in qpp_loss_batch:
                    if loss_type == "qpp_prediction":
                        # QPP予測損失は0.1倍にスケールダウン
                        weighted_loss = loss_value * 0.1
                        qpp_losses["loss_qpp_pred"] = weighted_loss
                        total_qpp_loss += weighted_loss
                    else:
                        # 正則化損失はそのまま
                        qpp_losses[f"loss_{loss_type}"] = loss_value
                        total_qpp_loss += loss_value
                
                result.update(qpp_losses)
                # 総損失にQPP損失とFocal Lossを追加
                result["total_loss"] = loss_crf + loss_mle_e + focal_loss_value + total_qpp_loss
                
            else:
                # QPP損失がない場合でもFocal Lossを追加
                result["total_loss"] = loss_crf + loss_mle_e + focal_loss_value
                
            return result

        elif self.args.mode == 'inference':
            assert len(predicted_path_batch[0])==len(predicted_path_batch_from_emission[0])==len(I_label_sequence_batch[0])==2
            
            if len(predicted_path_batch)>1:
                assert len(predicted_path_batch[1])==len(predicted_path_batch_from_emission[1])==len(I_label_sequence_batch[1])==4

            # 予測パスと予測確率の両方を返す
            return {
                'predicted_paths': predicted_path_batch,
                'emission_scores': emission_scores_batch
            }
