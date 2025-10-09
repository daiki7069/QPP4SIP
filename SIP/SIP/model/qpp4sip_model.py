from utils.math_utils import universal_sentence_embedding
from model.focal_loss import create_focal_loss_for_sip
from model.music_model import distance_crf, utterance_encoding, posterior_conversation_encoding, prior_conversation_encoding  # 共通クラスをインポート
import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn.functional as F


class qpp_feature_fusion(nn.Module):
    """
    QPP特徴量融合モジュール
    BERTの[CLS]表現にQPP特徴量を結合してMLPで埋め込み
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        # QPP特徴量の次元を設定から取得（デフォルトは9）
        self.qpp_feature_dim = getattr(args, 'qpp_feature_dim', 9)
        # 使用する特徴量のインデックスを設定から取得
        self.qpp_feature_indices = getattr(args, 'qpp_feature_indices', None)
        if self.qpp_feature_indices is not None:
            self.actual_qpp_dim = len(self.qpp_feature_indices)
        else:
            self.actual_qpp_dim = self.qpp_feature_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.args.hidden_size + self.actual_qpp_dim, self.args.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.hidden_size, self.args.hidden_size)
        )

    def forward(self, bert_representation, qpp_features):
        """
        Args:
            bert_representation: [batch_size, hidden_size] BERTの[CLS]表現
            qpp_features: [batch_size, qpp_feature_dim] QPP特徴量
        Returns:
            fused_representation: [batch_size, hidden_size] 融合された表現
        """
        # 特徴量選択（指定されたインデックスのみを使用）
        if self.qpp_feature_indices is not None:
            selected_features = qpp_features[:, self.qpp_feature_indices]
        else:
            selected_features = qpp_features
        
        # BERT表現とQPP特徴量を結合
        combined = torch.cat([bert_representation, selected_features], dim=-1)
        # MLPで埋め込み
        fused_representation = self.fusion_mlp(combined)
        return fused_representation

class qpp_auxiliary_head(nn.Module):
    """
    QPP補助ヘッド（マルチタスク学習用）
    指定されたQPP特徴量を予測する回帰ヘッド
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        # 予測対象の特徴量インデックス（デフォルトはndcg@3）
        self.target_feature_index = getattr(args, 'qpp_target_feature_index', 1)  # ndcg@3
        self.qpp_head = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.args.dropout),
            nn.Linear(self.args.hidden_size // 2, 1)  # 単一特徴量の予測
        )

    def forward(self, bert_representation):
        """
        Args:
            bert_representation: [batch_size, hidden_size] BERTの[CLS]表現
        Returns:
            qpp_prediction: [batch_size, 1] 指定されたQPP特徴量の予測値
        """
        return self.qpp_head(bert_representation)

class qpp_policy_gating(nn.Module):
    """
    QPPポリシー連動モジュール
    指定されたQPPスコアに基づいてイニシアチブ発動の閾値を調整
    """
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        # ゲート計算に使用する特徴量インデックス（デフォルトはndcg@3）
        self.gate_feature_index = getattr(args, 'qpp_gate_feature_index', 1)  # ndcg@3
        self.qpp_gate = nn.Sequential(
            nn.Linear(1, 16),  # 単一特徴量を入力
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0-1の範囲でゲート値を出力
        )

    def forward(self, qpp_score):
        """
        Args:
            qpp_score: [batch_size, 1] 指定されたQPPスコア
        Returns:
            gate_value: [batch_size, 1] ゲート値（0-1）
        """
        return self.qpp_gate(qpp_score)


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
        self.distance_crf = distance_crf(args=args)  # 共通のCRFクラスを使用
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

            prior_hidden_squence = self.prior_conversation_encoding(prior_utterance_sequence)
            prior_emission_score = self.prior_e_project(prior_hidden_squence[:, -1, :])

            # QPP特徴量の処理
            qpp_gate = None
            if qpp_features is not None and i < qpp_features.shape[1]:
                current_qpp_features = qpp_features[:, i, :]  # [batch_size, qpp_feature_dim]
                
                # 特徴量のバリデーション
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
                
                # 特徴量選択の適用
                if hasattr(self, 'qpp_feature_fusion') and self.qpp_feature_fusion.qpp_feature_indices is not None:
                    # 特徴量選択が設定されている場合は、選択された特徴量のみを使用
                    selected_features = current_qpp_features[:, self.qpp_feature_fusion.qpp_feature_indices]
                else:
                    selected_features = current_qpp_features
                
                if self.args.qpp4sip_pattern == "feature_fusion":
                    # 特徴融合: BERT表現にQPP特徴量を結合
                    bert_cls = prior_hidden_squence[:, -1, :]  # [batch_size, hidden_size*2]
                    # 最初の半分を取ってBERTの[CLS]表現として使用
                    bert_cls_half = bert_cls[:, :self.args.hidden_size]  # [batch_size, hidden_size]
                    
                    # 選択された特徴量を使用
                    if selected_features.sum() == 0:
                        selected_features = torch.randn_like(selected_features) * 0.01
                    
                    fused_representation = self.qpp_feature_fusion(bert_cls_half, selected_features)
                    # 融合された表現を元の次元に戻す
                    prior_emission_score = self.prior_e_project(torch.cat([fused_representation, bert_cls[:, self.args.hidden_size:]], dim=-1))
                    
                    # QPP特徴量の正則化損失を追加（特徴量の品質向上）
                    feature_regularization = torch.mean(torch.norm(selected_features, p=2, dim=1))
                    qpp_loss_batch.append(("feature_reg", feature_regularization * 0.01))  # 小さな重みで正則化
                    
                elif self.args.qpp4sip_pattern == "auxiliary_head":
                    # 補助ヘッド: QPP予測タスクを追加
                    bert_cls = prior_hidden_squence[:, -1, :]
                    bert_cls_half = bert_cls[:, :self.args.hidden_size]
                    qpp_prediction = self.qpp_auxiliary_head(bert_cls_half)
                    # 指定された特徴量の真値を取得
                    target_index = self.qpp_auxiliary_head.target_feature_index
                    qpp_target = current_qpp_features[:, target_index:target_index+1]
                    
                    # QPP特徴量が全て0の場合は、小さなノイズを追加
                    if current_qpp_features.sum() == 0:
                        qpp_target = torch.randn_like(qpp_target) * 0.01
                    
                    qpp_loss = F.mse_loss(qpp_prediction, qpp_target)
                    qpp_loss_batch.append(("qpp_prediction", qpp_loss))
                    
                elif self.args.qpp4sip_pattern == "policy_gating":
                    # ポリシー連動: QPPスコアに基づいてゲート値を計算
                    gate_index = self.qpp_policy_gating.gate_feature_index
                    gate_score = current_qpp_features[:, gate_index:gate_index+1]
                    
                    # QPP特徴量が全て0の場合は、小さなノイズを追加
                    if current_qpp_features.sum() == 0:
                        gate_score = torch.randn_like(gate_score) * 0.01
                    
                    qpp_gate = self.qpp_policy_gating(gate_score)
                    
                    # Policy Gating: 通常の予測とQPP情報を統合
                    # 1. 通常の予測値（gatingなし）
                    base_emission_score = prior_emission_score.clone()
                    
                    # 2. QPP情報に基づく予測値（QPP値自体を予測に反映）
                    qpp_informed_score = torch.tanh(gate_score) * 2.0  # QPP値を予測スコアに変換
                    # prior_emission_scoreの形状に合わせて調整
                    qpp_informed_score = qpp_informed_score.expand_as(prior_emission_score)
                    
                    # 3. ゲート値による統合: g * base + (1-g) * qpp_informed
                    prior_emission_score = qpp_gate * base_emission_score + (1 - qpp_gate) * qpp_informed_score
                    
                    # ゲート値の正則化損失を追加（ゲートの学習安定化）
                    gate_regularization = torch.mean(torch.norm(qpp_gate, p=2, dim=1))
                    qpp_loss_batch.append(("gate_reg", gate_regularization * 0.01))  # 小さな重みで正則化

            if self.args.mode == 'train':
                # obtain emission score
                posterior_hidden_squence = self.posterior_conversation_encoding(posterior_utterance_sequence)
                posterior_emission_scores = self.posterior_e_project(posterior_hidden_squence)
                
                # Policy Gatingの場合、posterior_emission_scoresにもゲート値を適用
                if self.args.qpp4sip_pattern == "policy_gating" and qpp_gate is not None:
                    # 1. 通常の予測値（gatingなし）
                    base_posterior_scores = posterior_emission_scores.clone()
                    
                    # 2. QPP情報に基づく予測値（QPP値自体を予測に反映）
                    qpp_informed_posterior = torch.tanh(gate_score) * 2.0  # QPP値を予測スコアに変換
                    qpp_informed_posterior = qpp_informed_posterior.unsqueeze(1).expand_as(posterior_emission_scores)
                    
                    # 3. ゲート値による統合: g * base + (1-g) * qpp_informed
                    posterior_emission_scores = qpp_gate.unsqueeze(1) * base_posterior_scores + (1 - qpp_gate.unsqueeze(1)) * qpp_informed_posterior

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

                gold_score, total_score= self.distance_crf(posterior_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_squence[:, -1, :].squeeze(0), posterior_hidden_squence.squeeze(0), state)

                gold_score_batch.append(gold_score.unsqueeze(0))
                total_score_batch.append(total_score.unsqueeze(0))

                prior_emission_score_batch.append(prior_emission_score.unsqueeze(1))
                posterior_emission_score_batch.append(posterior_emission_scores[:, -1, :].unsqueeze(1))

                # End of training cycle
            elif self.args.mode == 'inference':
                partial_posterior_hidden_squence = self.posterior_conversation_encoding(prior_utterance_sequence)  #  [1, 2i+1, 2*hidden_size_BiLSTM]
                partial_posterior_emission_scores = self.posterior_e_project(partial_posterior_hidden_squence)  # [1, 2i+1, 2]

                combined_emission_scores = torch.cat([partial_posterior_emission_scores, prior_emission_score.unsqueeze(1)], 1)  # [1, 2i+2, 2]
                predicted_path = self.distance_crf(combined_emission_scores.squeeze(0), I_label_sequence.squeeze(0), prior_hidden_squence[:, -1, :].squeeze(0), partial_posterior_hidden_squence.squeeze(0) , state)
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

            loss_distance_crf = torch.mean(total_score_tensor - gold_score_tensor) # average each sample
            loss_mle_e = F.mse_loss(prior_emission_score_tensor, posterior_emission_score_tensor.detach())  # [pair_num, 2]
            
            # Focal Loss for emission scores (prior network)
            # Convert labels to appropriate format for focal loss
            # We'll use the system I labels for focal loss calculation
            system_labels = data['system_I_label'].squeeze(0)  # [pair_num]
            focal_loss_value = self.focal_loss(prior_emission_score_tensor, system_labels)

            result = {"loss_distance_crf": loss_distance_crf, "loss_mle_e": loss_mle_e, "loss_focal": focal_loss_value}
            
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
                result["total_loss"] = loss_distance_crf + loss_mle_e + focal_loss_value + total_qpp_loss
                
            else:
                # QPP損失がない場合でもFocal Lossを追加
                result["total_loss"] = loss_distance_crf + loss_mle_e + focal_loss_value
                
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
