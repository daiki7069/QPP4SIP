from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import json
import os
from collections import defaultdict
import torch
import codecs
import time
import sys
import numpy as np


class Trainer(object):
    def __init__(self, args, model, writer=None):
        super(Trainer, self).__init__()
        self.args = args

        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model

        self.eval_model = self.model

        self.accumulation_count = 0
        self.writer = writer

    def train_batch(self, epoch, data, optimizer, scheduler=None):
        self.accumulation_count += 1
        loss = self.model(data)

        sum_loss = torch.cat([loss[name].reshape(1) for name in loss]).sum()/self.args.accumulation_steps
        sum_loss.backward()

        if self.accumulation_count % self.args.accumulation_steps == 0:
            if self.args.task=="SIP":
                self.writer.add_scalar('Loss/overall', sum_loss.item(), scheduler.state_dict()['_step_count'])
                self.writer.add_scalar('Loss/crf', loss["loss_crf"].item(), scheduler.state_dict()['_step_count'])
                self.writer.add_scalar('Loss/mle_e', loss["loss_mle_e"].item(), scheduler.state_dict()['_step_count'])
                self.writer.add_scalars('Loss/all', {'overall': sum_loss.item(),'crf': loss["loss_crf"].item(),'mle_e': loss["loss_mle_e"].item()},scheduler.state_dict()['_step_count'])
            elif self.args.task in ["AP", "SIP-AP"]:
                self.writer.add_scalar('Loss', sum_loss.item(), scheduler.state_dict()['_step_count'])
            else:
                raise NotImplementedError

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        loss_dict = dict()
        for i in loss:
            loss_dict[i] = loss[i].cpu().item()

        return loss_dict

    def serialize(self, epoch, scheduler, saved_model_path):
        # ディレクトリが存在しない場合は作成
        os.makedirs(saved_model_path, exist_ok=True)
        
        fuse_dict = {"model": self.eval_model.state_dict(), "scheduler": scheduler.state_dict()}
        torch.save(fuse_dict, os.path.join(saved_model_path, '.'.join([str(epoch), 'pkl'])))
        print("Saved epoch {} model".format(epoch))

    def train_epoch(self, train_dataset, train_collate_fn, epoch, optimizer, scheduler=None):
        self.model.train()  

        train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.args.batch_size, shuffle=True)

        start_time = time.perf_counter()
        step = 0

        for j, data in enumerate(train_loader, 0):
            if torch.cuda.is_available():
                data_cuda = dict()
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_cuda[key] = value.cuda()
                    else:
                        data_cuda[key] = value
                data = data_cuda

            loss_dict = self.train_batch(epoch, data, optimizer=optimizer, scheduler=scheduler)
            step += 1

            if j >= 0 and j % 100 == 0:
                elapsed_time = time.perf_counter() - start_time
                log_message = 'Training: {} on the {} dataset'.format(self.args.name, self.args.dataset)
                print(log_message)
                
                loss_message = 'Epoch:{}, Step:{}, Loss:{}, Time:{}, LR:{}'.format(epoch, scheduler.state_dict()['_step_count'],loss_dict, round(elapsed_time, 2), scheduler.get_last_lr())
                print(loss_message)
                
                # ログファイルに出力
                if hasattr(self.args, 'log_path'):
                    log_file = os.path.join(self.args.log_path, f"train_{self.args.qpp4sip_pattern if hasattr(self.args, 'qpp4sip_pattern') else 'music'}.log")
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{log_message}\n")
                        f.write(f"{loss_message}\n")
                
                sys.stdout.flush()

        sys.stdout.flush()

    def infer(self, epoch_id, dataset, collate_fn):
        self.eval_model.eval()
        with torch.no_grad():
            test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.batch_size,shuffle=False, collate_fn=collate_fn, num_workers=0)

            accumulative_turn_id = []
            accumulative_prediction = []
            accumulative_true_label = []  # 正解ラベル
            accumulative_qpp_values = []  # QPPの値
            accumulative_prediction_probs = []  # initiativeの予測確率
            accumulative_resolved_query = []  # resolvedQuery

            for k, data in enumerate(test_loader, 0):
                if (k + 1) == 1 or (k + 1) % 100 == 0:
                    inference_message = "{} on the {} dataset: doing {} / total {} in epoch {}".format(self.args.name,self.args.dataset, k + 1,len(test_loader), epoch_id)
                    print(inference_message)
                    
                    # ログファイルに出力
                    if hasattr(self.args, 'log_path'):
                        log_file = os.path.join(self.args.log_path, f"inference_{self.args.dataset_type}_{self.args.qpp4sip_pattern if hasattr(self.args, 'qpp4sip_pattern') else 'music'}.log")
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(f"{inference_message}\n")

                if torch.cuda.is_available():
                    data_cuda = dict()
                    for key, value in data.items():
                        if isinstance(value, torch.Tensor):
                            data_cuda[key] = value.cuda()
                        else:
                            data_cuda[key] = value
                    data = data_cuda

                # [pair_num, ?]
                model_output = self.eval_model(data)
                
                # モデルの出力形式を確認
                if isinstance(model_output, dict):
                    predicted = model_output['predicted_paths']
                    emission_scores = model_output['emission_scores']
                else:
                    # 後方互換性のため、古い形式もサポート
                    predicted = model_output
                    emission_scores = None

                assert len(predicted)==len(data["turn_id"])

                for idx, turn_id in enumerate(data["turn_id"]):
                    # 会話インデックスとturn_idを組み合わせた一意の識別子を生成
                    unique_id = f"conv_{k}_turn_{turn_id}"
                    accumulative_turn_id.append(unique_id)
                    
                    # 予測ラベル
                    if self.args.task=="SIP":
                        accumulative_prediction.append("Initiative" if int(predicted[idx][-1])==1 else "Non-initiative")
                    elif self.args.task in ["AP", "SIP-AP"]:
                        accumulative_prediction.append(predicted[idx])
                    else:
                        raise NotImplementedError
                    
                    # 正解ラベル
                    true_label = "Initiative" if int(data["system_I_label"][0, idx].item()) == 1 else "Non-initiative"
                    accumulative_true_label.append(true_label)
                    
                    # QPPの値（ndcg@1を使用）
                    qpp_value = data["qpp_features"][0, idx, 0].item()  # ndcg@1
                    accumulative_qpp_values.append(qpp_value)
                    
                    # initiativeの予測確率
                    if emission_scores is not None:
                        # 最後のターン（システム発話）のinitiative確率を取得
                        last_turn_probs = emission_scores[idx][-1]  # [2] - [non-initiative_prob, initiative_prob]
                        initiative_prob = last_turn_probs[1].item()  # initiativeの確率
                        accumulative_prediction_probs.append(initiative_prob)
                    else:
                        accumulative_prediction_probs.append(0.0)  # デフォルト値
                    
                    # resolvedQuery
                    resolved_query = data["resolved_query"][idx] if "resolved_query" in data else ""
                    accumulative_resolved_query.append(resolved_query)

            with open(os.path.join(self.args.output_path, self.args.dataset_type+"."+str(epoch_id)+".txt"), 'w') as w:
                if self.args.task == "SIP":
                    # ヘッダー行を追加
                    w.write("turn_id\tpredicted_label\ttrue_label\tqpp_value\tinitiative_prob\tresolved_query\n")
                    
                    for index, turn_id in enumerate(accumulative_turn_id):
                        # 追加項目を含む出力形式
                        w.write(turn_id + '\t' + 
                               str(accumulative_prediction[index]) + '\t' +  # 予測ラベル
                               str(accumulative_true_label[index]) + '\t' +  # 正解ラベル
                               str(accumulative_qpp_values[index]) + '\t' +  # QPPの値
                               str(accumulative_prediction_probs[index]) + '\t' +  # initiativeの予測確率
                               str(accumulative_resolved_query[index]) + '\n')  # resolvedQuery
                elif self.args.task in ["AP", "SIP-AP"]:
                    for index, turn_id in enumerate(accumulative_turn_id):
                        assert isinstance(accumulative_prediction[index], list)
                        w.write(turn_id + '\t' + ",".join(accumulative_prediction[index]) + '\n')
                else:
                    raise NotImplementedError

        return None
