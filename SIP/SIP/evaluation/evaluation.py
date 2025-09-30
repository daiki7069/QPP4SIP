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
from utils import replicability, Config, hamming_score, rounder
from load_pkl import convert_dialogue_to_conversations

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
    # pickleファイルを読み込み
    with open(label_path, 'rb') as f:
        dialogue_data = pickle.load(f)
    
    # 会話IDとturn_idを組み合わせた一意の識別子を作成
    for conv_idx, (conv_id, conv_data) in enumerate(dialogue_data.items()):
        for turn_idx, turn in enumerate(conv_data['turns']):
            # 推論結果と同じ形式の一意の識別子を生成
            unique_id = f"conv_{conv_idx}_turn_{turn_idx}"
            
            # TSVから変換したデータの場合と元のデータの場合を分けて処理
            if 'labels' in turn and turn['labels'] and len(turn['labels']) > 0:
                # 元のINSCITデータ形式の場合
                response_type = turn['labels'][0]['responseType']
                # responseTypeをInitiative/Non-initiativeに変換
                if response_type in ['clarification', 'clarify']:
                    id2label[unique_id] = 'Initiative'
                else:
                    id2label[unique_id] = 'Non-initiative'
            elif 'system_I_label' in turn:
                # TSVから変換したデータ形式の場合
                system_i_label = turn['system_I_label']
                if system_i_label == 'clarification':
                    id2label[unique_id] = 'Initiative'
                else:
                    id2label[unique_id] = 'Non-initiative'
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

def evaluation_AP(prediction_path=None, label_path=None, mlb=None):
    """
    AP (Action Prediction) タスクの評価を行う
    
    Args:
        prediction_path: 予測結果ファイルのパス
        label_path: ラベルファイルのパス
        mlb: MultiLabelBinarizerオブジェクト
    
    Returns:
        dict: 評価結果の辞書
    """
    prediction_list = []
    label_list = []

    id2label = {}
    # pickleファイルを読み込み
    with open(label_path, 'rb') as f:
        dialogue_data = pickle.load(f)
    
    # 会話IDとturn_idを組み合わせた一意の識別子を作成
    for conv_idx, (conv_id, conv_data) in enumerate(dialogue_data.items()):
        for turn_idx, turn in enumerate(conv_data['turns']):
            # 推論結果と同じ形式の一意の識別子を生成
            unique_id = f"conv_{conv_idx}_turn_{turn_idx}"
            # labelsの最初の要素のresponseTypeを使用
            if turn['labels'] and len(turn['labels']) > 0:
                response_type = turn['labels'][0]['responseType']
                id2label[unique_id] = list(mlb.fit_transform([[response_type]])[0])
            else:
                id2label[unique_id] = list(mlb.fit_transform([['directAnswer']])[0])

    with open(prediction_path, 'r') as r:
        for line in r:
            if len(line.rstrip().split()) == 1:
                actions = []
            else:
                turn_id, prediction = line.rstrip().split("\t")
                actions = prediction.split(",")

            prediction_list.append(list(mlb.fit_transform([actions])[0]))
            # 一意の識別子でラベルを取得
            label_list.append(id2label[turn_id])

    print(len(id2label), len(prediction_list))
    assert len(id2label) == len(prediction_list)

    # 評価指標の計算
    f1 = f1_score(label_list, prediction_list, average="macro")
    precision = precision_score(label_list, prediction_list, average="macro")
    recall = recall_score(label_list, prediction_list, average="macro")
    acc_exact = accuracy_score(label_list, prediction_list)
    acc_hamming = hamming_score(np.array(label_list), np.array(prediction_list))
    jaccard = jaccard_score(np.array(label_list), np.array(prediction_list), average="samples")
    precision_detail = precision_score(label_list, prediction_list, average=None)
    recall_detail = recall_score(label_list, prediction_list, average=None)

    prediction_sys_action_num_all_turns = [sum(prediction) for prediction in prediction_list]

    result_dict = {
        "f1": rounder(f1),
        "p": rounder(precision),
        "r": rounder(recall),
        "jaccard": rounder(jaccard),
        "acc_hamming": rounder(acc_hamming),
        "acc_exact": rounder(acc_exact),
        "aver_sys_action_num": sum(prediction_sys_action_num_all_turns) / len(prediction_sys_action_num_all_turns),
        "min_sys_action_num": min(prediction_sys_action_num_all_turns),
        "max_sys_action_num": max(prediction_sys_action_num_all_turns),
        "p_per_label": [rounder(i) for i in precision_detail],
        "r_per_label": [rounder(i) for i in recall_detail],
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
    parser.add_argument('--dataset', type=str, default='QPP4SIP', help='データセット名')
    
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

if __name__ == '__main__':
    main()