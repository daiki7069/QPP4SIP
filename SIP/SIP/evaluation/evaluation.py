import argparse
import json
import os
import numpy as np
import torch
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer
import sys
sys.path.append('./model')
sys.path.append('./utils')
sys.path.append('./config')
from utils.random_utils import replicability
from utils.math_utils import hamming_score, rounder
from config.dataset_config import Config
from utils.load_pkl import convert_dialogue_to_conversations

def evaluation_SIP(prediction_path=None, label_path=None):
    """
    SIP (System Initiative Prediction) タスクの評価を行う
    
    Args:
        prediction_path: 予測結果ファイルのパス
        label_path: ラベルファイルのパス
    
    Returns:
        dict: 評価結果の辞書
    """
    prediction_list = []
    label_list = []

    id2label = {}
    # pickleファイルを読み込み（torch.saveで保存されているためtorch.loadを使用）
    import torch
    dialogue_data = torch.load(label_path)
    
    # 会話ごとにグループ化
    dialogue_groups = {}
    for conversation_index, conversation in enumerate(dialogue_data):
        for turn in conversation:
            dialogue_id = turn.get('conv_id', f'conversation_{conversation_index}')
            if dialogue_id not in dialogue_groups:
                dialogue_groups[dialogue_id] = []
            dialogue_groups[dialogue_id].append(turn)

    # 会話IDとturn_idを組み合わせた一意の識別子を作成
    for conv_idx, (conv_id, turns) in enumerate(dialogue_groups.items()):
        for turn_idx, turn in enumerate(turns):
            # 推論結果と同じ形式の一意の識別子を生成（conversation_index_turn_id形式）
            unique_id = f"{conv_idx}_{turn['turn_id']}"

            # response_typeをInitiative/Non-initiativeに変換
            response_type = turn.get('response_type', '')
            if response_type in ['clarification', 'clarify']:
                id2label[unique_id] = 'Initiative'
            else:
                id2label[unique_id] = 'Non-initiative'

    with open(prediction_path, 'r') as r:
        lines = r.readlines()
        # ヘッダー行をスキップ
        if lines and lines[0].startswith("turn_id\t"):
            lines = lines[1:]
        
        for line in lines:
            parts = line.rstrip().split("\t")
            if len(parts) >= 2:
                turn_id = parts[0]
                prediction = parts[1]
                prediction_list.append(prediction)
                # 一意の識別子でラベルを取得
                label_list.append(id2label[turn_id])

    print(f"Debug: id2label length: {len(id2label)}")
    print(f"Debug: prediction_list length: {len(prediction_list)}")
    print(f"Debug: label_list length: {len(label_list)}")
    
    # 長さが一致しない場合は警告を出して続行
    if len(id2label) != len(prediction_list):
        print(f"Warning: Length mismatch - id2label: {len(id2label)}, prediction_list: {len(prediction_list)}")
        # 短い方に合わせる
        min_len = min(len(id2label), len(prediction_list))
        prediction_list = prediction_list[:min_len]
        label_list = label_list[:min_len]
        print(f"Adjusted to length: {min_len}")

    # ラベルを数値に変換
    label_list = [1 if i == "Initiative" else 0 for i in label_list]
    prediction_list = [1 if i == "Initiative" else 0 for i in prediction_list]

    # 評価指標の計算
    acc = accuracy_score(label_list, prediction_list)
    matrix = confusion_matrix(label_list, prediction_list, labels=[0, 1])
    acc_per_label = matrix.diagonal() / matrix.sum(axis=1)
    total_num = matrix.sum(axis=1).tolist()
    hit_num = matrix.diagonal().tolist()
    f1 = f1_score(label_list, prediction_list, average="macro")
    precision = precision_score(label_list, prediction_list, average="macro")
    recall = recall_score(label_list, prediction_list, average="macro")
    precision_detail = precision_score(label_list, prediction_list, average=None)
    recall_detail = recall_score(label_list, prediction_list, average=None)

    result_dict = {
        "f1": rounder(f1),
        "p": rounder(precision),
        "r": rounder(recall),
        "acc": rounder(acc),
        "acc_per_label": [rounder(i) for i in acc_per_label],
        "total_num": total_num,
        "hit_num": hit_num,
        "p_per_label": [rounder(i) for i in precision_detail],
        "r_per_label": [rounder(i) for i in recall_detail]
    }

    print(result_dict)
    return result_dict

def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(description='QPP4SIP Evaluation')
    parser.add_argument('--prediction_path', type=str, required=True, help='予測結果ファイルのパス')
    parser.add_argument('--label_path', type=str, required=True, help='ラベルファイルのパス')
    parser.add_argument('--task', type=str, default='SIP', choices=['SIP', 'AP'], help='評価するタスク')
    parser.add_argument('--dataset_type', type=str, default='test', choices=['train', 'dev', 'test'], help='データセットの種類')
    parser.add_argument('--epoch_num', type=int, default=20, help='評価するエポック数')
    parser.add_argument('--dataset', type=str, default='INSCIT', help='データセット名（INSCIT固定）')
    
    args = parser.parse_args()
    
    # 結果ファイルのパス
    result_file = os.path.join(args.prediction_path, f"result.{args.dataset_type}.txt")
    
    # 既存の結果ファイルを削除
    if os.path.exists(result_file):
        os.remove(result_file)
        print(f"既存の結果ファイルを削除しました: {result_file}")
    
    # 各エポックの評価を実行
    for epoch_id in range(1, args.epoch_num + 1):
        prediction_path_ = os.path.join(args.prediction_path, f"{args.dataset_type}.{epoch_id}.txt")

        if os.path.exists(prediction_path_):
            print(f"エポック {epoch_id} の評価を開始します")
            
            if args.task == "AP":
                result_dict = evaluation_AP(prediction_path_, args.label_path, mlb)
            else:  # SIP
                result_dict = evaluation_SIP(prediction_path_, args.label_path)

            # 結果をファイルに追記保存
            with open(result_file, 'a+', encoding='utf-8') as w:
                w.write(f"{epoch_id}: {result_dict}\n")
        else:
            print(f"予測ファイルが見つかりません: {prediction_path_}")

def generate_summary(all_results, dataset_type="dev"):
    """
    評価結果のサマリーを生成する
    
    Args:
        all_results (dict): 全エポックの評価結果
        dataset_type (str): データセットタイプ
        
    Returns:
        str: サマリーテキスト
    """
    if not all_results:
        return ""
    
    summary_lines = []
    summary_lines.append("")
    summary_lines.append(f"=== {dataset_type.upper()} データセット 評価結果サマリー ===")
    summary_lines.append(f"評価エポック数: {len(all_results)}")
    summary_lines.append("")
    
    # ヘッダー行
    summary_lines.append("エポック     精度       再現率      F1スコア    正解率      Initiative   Non-initiative ")
    summary_lines.append("-" * 80)
    
    # 各エポックの結果
    for epoch_id in sorted(all_results.keys()):
        results = all_results[epoch_id]
        precision = results['p']
        recall = results['r']
        f1 = results['f1']
        accuracy = results['acc']
        acc_per_label = results['acc_per_label']
        
        # Initiative と Non-initiative の精度
        initiative_acc = acc_per_label[1] if len(acc_per_label) > 1 else 0.0
        non_initiative_acc = acc_per_label[0] if len(acc_per_label) > 0 else 0.0
        
        summary_lines.append(f"{epoch_id:<8} {precision:>8.2f} {recall:>8.2f} {f1:>8.2f} {accuracy:>8.2f} {initiative_acc:>8.2f} {non_initiative_acc:>12.2f}")
    
    summary_lines.append("")
    
    # 最良の結果を検索
    best_f1_epoch = max(all_results.keys(), key=lambda x: all_results[x]['f1'])
    best_acc_epoch = max(all_results.keys(), key=lambda x: all_results[x]['acc'])
    
    best_f1_score = all_results[best_f1_epoch]['f1']
    best_acc_score = all_results[best_acc_epoch]['acc']
    
    summary_lines.append("=== 最良の結果 ===")
    summary_lines.append(f"最高F1スコア: {best_f1_score:.2f} (エポック {best_f1_epoch})")
    summary_lines.append(f"最高正解率: {best_acc_score:.2f} (エポック {best_acc_epoch})")
    summary_lines.append("")
    
    # 最良F1スコアのエポックの詳細結果
    best_results = all_results[best_f1_epoch]
    summary_lines.append(f"=== エポック {best_f1_epoch} の詳細結果 ===")
    summary_lines.append(f"精度 (Precision): {best_results['p']:.2f}")
    summary_lines.append(f"再現率 (Recall): {best_results['r']:.2f}")
    summary_lines.append(f"F1スコア: {best_results['f1']:.2f}")
    summary_lines.append(f"正解率 (Accuracy): {best_results['acc']:.2f}")
    summary_lines.append("")
    
    # ラベル別結果
    summary_lines.append("ラベル別結果:")
    total_num = best_results['total_num']
    hit_num = best_results['hit_num']
    acc_per_label = best_results['acc_per_label']
    
    if len(total_num) >= 2 and len(hit_num) >= 2:
        non_initiative_hit = hit_num[0]
        non_initiative_total = total_num[0]
        initiative_hit = hit_num[1]
        initiative_total = total_num[1]
        
        non_initiative_pct = (non_initiative_hit / non_initiative_total * 100) if non_initiative_total > 0 else 0
        initiative_pct = (initiative_hit / initiative_total * 100) if initiative_total > 0 else 0
        
        summary_lines.append(f"  Non-initiative: {non_initiative_hit}/{non_initiative_total} ({non_initiative_pct:.2f}%)")
        summary_lines.append(f"  Initiative: {initiative_hit}/{initiative_total} ({initiative_pct:.2f}%)")
    
    return "\n".join(summary_lines)

if __name__ == '__main__':
    main()