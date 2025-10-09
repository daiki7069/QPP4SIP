#!/usr/bin/env python3
"""
学習・推論のエントリポイント
モデルの学習、推論、評価を実行する
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import Dataset, collate_fn
from model.music_model import BILSTMCRF
from model.qpp4sip_model import QPP4SIPBILSTMCRF
from model.trainer import Trainer
from utils.random_utils import replicability
from config.dataset_config import Config
from utils.load_pkl import convert_dialogue_to_conversations
from config.qpp_config import QPPExperimentConfig


def get_model(args):
    """
    モデル引数に基づいて適切なモデルを返す関数
    """
    if args.model == "music":
        return BILSTMCRF(args)
    elif args.model == "qpp4sip":
        return QPP4SIPBILSTMCRF(args)
    else:
        raise ValueError(f"サポートされていないモデル: {args.model}")


def get_qpp_feature_id(args):
    """
    QPP特徴量のIDを生成する関数
    
    Args:
        args: コマンドライン引数
        
    Returns:
        str: QPP特徴量のID（例: "012" for ndcg系のみ）
    """
    if args.model != "qpp4sip":
        return ""
    
    # QPP特徴量のインデックスを取得
    if hasattr(args, 'qpp_feature_indices') and args.qpp_feature_indices:
        # 指定された特徴量インデックスを使用
        feature_indices = args.qpp_feature_indices
    else:
        # デフォルトは全特徴量（0-8）
        feature_indices = list(range(9))
    
    # インデックスを文字列に変換して結合
    feature_id = ''.join(map(str, sorted(feature_indices)))
    return feature_id


def run_train(args):
    """
    学習を実行する関数
    """
    print("=== 学習開始 ===")
    print(f"モデル: {args.model}")
    print(f"QPP4SIPパターン: {getattr(args, 'qpp4sip_pattern', 'N/A')}")
    print(f"エポック数: {args.epoch_num}")
    print(f"学習率: {args.learning_rate}")
    # バッチサイズは設計上1のみサポート（CRFの制約のため）
    batch_size = 1
    print(f"バッチサイズ: {batch_size} (固定)")
    
    # 設定の初期化
    config = Config(args)
    mlb = MultiLabelBinarizer()
    
    # データの読み込み
    print(f"データを読み込み中: {args.input_path}")
    with open(args.input_path, 'rb') as f:
        dialogue_data = pickle.load(f)
    
    # ダイアログデータを会話データに変換
    conversations = convert_dialogue_to_conversations(dialogue_data)
    print(f"会話数: {len(conversations)}")
    
    # QPP特徴量の検証
    if args.model == "qpp4sip":
        # conversations[0]は会話のターンのリストなので、最初のターンのQPP特徴量を確認
        qpp_features = None
        if conversations and len(conversations) > 0 and len(conversations[0]) > 0:
            # 実際のデータセットからQPP特徴量を構築
            turn = conversations[0][0]
            qpp_feature_names = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10', 'recall@1', 'recall@5', 'recall@10']
            qpp_features = {name: turn.get(name, 0.0) for name in qpp_feature_names}
        QPPExperimentConfig.validate_qpp_features(qpp_features)
    
    # モデルの初期化
    model = get_model(args)
    
    # GPU使用の設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    model = model.to(device)
    
    # オプティマイザーの設定
    if args.model == "music":
        from torch import optim
        from transformers import get_constant_schedule
        
        model_optimizer = optim.Adam([
            {"params": model.utterance_encoding.parameters()},
            {"params": model.posterior_conversation_encoding.lstm.parameters()},
            {"params": model.prior_conversation_encoding.lstm.parameters()},
            {"params": model.prior_e_project.parameters()},
            {"params": model.posterior_e_project.parameters()},
            {"params": model.distance_distance_crf.parameters(), "lr": args.lr_distance_crf}
        ], lr=args.learning_rate)
    else:
        from torch import optim
        from transformers import get_constant_schedule
        
        model_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model_scheduler = get_constant_schedule(model_optimizer)
    
    # TensorBoardライターの設定
    writer = SummaryWriter(args.log_path)
    
    # 学習の実行
    trainer = Trainer(args, model, writer)
    model_optimizer.zero_grad()
    
    for i in range(1, args.epoch_num + 1):
        print(f"エポック {i}/{args.epoch_num} 開始")
        
        dataset = Dataset(args, config, mlb, conversations)
        trainer.train_epoch(dataset, collate_fn, i, model_optimizer, model_scheduler)
        trainer.serialize(i, model_scheduler, saved_model_path=args.saved_model_path)
        
        print(f"エポック {i}/{args.epoch_num} 完了")
    
    writer.close()
    print("=== 学習完了 ===")


def run_inference(args):
    """
    推論を実行する関数
    """
    print("=== 推論開始 ===")
    print(f"モデル: {args.model}")
    print(f"QPP4SIPパターン: {getattr(args, 'qpp4sip_pattern', 'N/A')}")
    print(f"チェックポイント: {args.saved_model_path}")
    
    # 設定の初期化
    config = Config(args)
    mlb = MultiLabelBinarizer()
    
    # データの読み込み
    print(f"データを読み込み中: {args.input_path}")
    with open(args.input_path, 'rb') as f:
        dialogue_data = pickle.load(f)
    
    # ダイアログデータを会話データに変換
    conversations = convert_dialogue_to_conversations(dialogue_data)
    print(f"会話数: {len(conversations)}")
    
    # データセットの作成
    dataset = Dataset(args, config, mlb, conversations)
    
    # 各エポックの推論を実行
    for epoch_id in range(1, args.epoch_num + 1):
        checkpoint_path = os.path.join(args.saved_model_path, f"{epoch_id}.pkl")
        
        if os.path.exists(checkpoint_path):
            print(f"エポック {epoch_id} の推論を開始します")
            
            # モデルの初期化
            model = get_model(args)
            
            # GPU使用の設定
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用デバイス: {device}")
            
            # チェックポイントの読み込み
            print(f"チェックポイントを読み込み中: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            
            # デバイスに移動
            model = model.to(device)
            
            # 推論の実行
            trainer = Trainer(args, model)
            trainer.infer(epoch_id, dataset, collate_fn)
            
            print(f"エポック {epoch_id} の推論が完了しました")
        else:
            print(f"チェックポイントが見つかりません: {checkpoint_path}")
    
    print("=== 推論完了 ===")


def run_evaluation(args):
    """
    評価を実行する関数
    """
    print("=== 評価開始 ===")
    
    from evaluation.evaluation import evaluation_SIP, generate_summary
    
    # 全エポックの評価結果を格納
    all_results = {}
    
    # 各エポックの評価を実行
    for epoch_id in range(1, args.epoch_num + 1):
        prediction_path = os.path.join(args.output_path, f"dev.{epoch_id}.txt")
        print(f"チェック中: {prediction_path}")
        
        if os.path.exists(prediction_path):
            print(f"エポック {epoch_id} の評価を開始")
            results = evaluation_SIP(prediction_path=prediction_path, label_path=args.input_path)
            all_results[epoch_id] = results
            print(f"エポック {epoch_id}: F1={results['f1']:.2f}, Acc={results['acc']:.2f}")
        else:
            print(f"予測ファイルが見つかりません: {prediction_path}")
    
    # 全エポックの結果を1つのファイルに保存
    if all_results:
        result_file = os.path.join(args.output_path, "result.dev.txt")
        with open(result_file, 'w') as f:
            for epoch_id, results in all_results.items():
                # numpyのデータ型を通常のfloatに変換
                clean_results = {}
                for key, value in results.items():
                    if isinstance(value, (list, np.ndarray)):
                        clean_results[key] = [float(v) if hasattr(v, 'item') else v for v in value]
                    elif hasattr(value, 'item'):
                        clean_results[key] = float(value)
                    else:
                        clean_results[key] = value
                f.write(f"{epoch_id}: {clean_results}\n")
        
        # サマリーを生成して追記
        summary = generate_summary(all_results, "dev")
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"全結果とサマリーを保存しました: {result_file}")
    else:
        print("警告: 評価可能なエポックが見つかりませんでした")
        print(f"出力ディレクトリ: {args.output_path}")
        print(f"期待されるファイル形式: dev.{{epoch_id}}.txt")
    
    print("=== 評価完了 ===")


def run_multi_seed_inference(args):
    """
    複数シードによる推論を実行する関数
    """
    print("=== 複数シード推論開始 ===")
    print(f"モデル: {args.model}")
    print(f"QPP4SIPパターン: {getattr(args, 'qpp4sip_pattern', 'N/A')}")
    print(f"チェックポイント: {args.saved_model_path}")
    print(f"実行回数: {args.num_runs}")
    
    # 設定の初期化
    config = Config(args)
    mlb = MultiLabelBinarizer()
    
    # データの読み込み
    print(f"データを読み込み中: {args.input_path}")
    with open(args.input_path, 'rb') as f:
        dialogue_data = pickle.load(f)
    
    # ダイアログデータを会話データに変換
    conversations = convert_dialogue_to_conversations(dialogue_data)
    print(f"会話数: {len(conversations)}")
    
    # データセットの作成
    dataset = Dataset(args, config, mlb, conversations)
    
    # 各エポックの複数シード推論を実行
    for epoch_id in range(1, args.epoch_num + 1):
        checkpoint_path = os.path.join(args.saved_model_path, f"{epoch_id}.pkl")
        
        if os.path.exists(checkpoint_path):
            print(f"エポック {epoch_id} の複数シード推論を開始します")
            
            # 複数回実行してファイルを出力
            for run_idx in range(args.num_runs):
                print(f"実行 {run_idx + 1}/{args.num_runs}")
                
                # シードを設定
                np.random.seed(42 + run_idx)
                torch.manual_seed(42 + run_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(42 + run_idx)
                
                # モデルの初期化
                model = get_model(args)
                
                # GPU使用の設定
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # チェックポイントの読み込み
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                model = model.to(device)
                
                # 推論の実行（ファイル名にepoch_idとrun_idxを追加）
                # 推論モードに設定
                args.mode = 'inference'
                trainer = Trainer(args, model)
                # 一時的にファイル名を変更
                original_dataset_type = args.dataset_type
                args.dataset_type = f"dev_{epoch_id}_{run_idx + 1}"
                trainer.infer(epoch_id, dataset, collate_fn)
                args.dataset_type = original_dataset_type  # 元に戻す
            
            print(f"エポック {epoch_id} の複数シード推論が完了しました")
        else:
            print(f"チェックポイントが見つかりません: {checkpoint_path}")
    
    print("=== 複数シード推論完了 ===")
    print("各実行の結果ファイルが出力されました。別スクリプトで集計してください。")




def main():
    parser = argparse.ArgumentParser(description="学習・推論のエントリポイント")
    
    # 基本設定
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["train", "inference", "evaluation", "multi_seed_inference"], 
                       help="実行モード")
    parser.add_argument("--task", type=str, default="SIP", help="タスク名")
    parser.add_argument("--name", type=str, default="QPP4SIP", help="モデル名")
    parser.add_argument("--dataset", type=str, default="INSCIT", help="データセット名（INSCIT固定）")
    parser.add_argument("--model", type=str, default="music", 
                       choices=["music", "qpp4sip"], 
                       help="モデルタイプ")
    parser.add_argument("--qpp4sip_pattern", type=str, default="feature_fusion", 
                       choices=["feature_fusion", "auxiliary_head", "policy_gating"], 
                       help="QPP4SIP実装パターン")
    
    # パス設定
    parser.add_argument("--input_path", type=str, required=True, help="入力データのパス")
    parser.add_argument("--output_path", type=str, default="./output", help="出力パス")
    parser.add_argument("--saved_model_path", type=str, default="./checkpoints", help="チェックポイントのパス")
    parser.add_argument("--log_path", type=str, default="log", help="ログのパス")
    
    # モデルのパラメータ
    parser.add_argument("--hidden_size", type=int, default=768, help="隠れ層のサイズ")
    parser.add_argument("--dropout", type=float, default=0.1, help="ドロップアウト率")
    parser.add_argument("--BiLSTM_layers", type=int, default=1, help="BiLSTM層数")
    
    # 学習・推論のパラメータ
    parser.add_argument("--epoch_num", type=int, default=20, help="エポック数")
    # バッチサイズは設計上1のみサポート（CRFの制約のため）
    # parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学習率")
    
    # 複数シード推論用のパラメータ
    parser.add_argument("--num_runs", type=int, default=10, help="複数シード推論の実行回数")
    parser.add_argument("--lr_distance_crf", type=float, default=1e-3, help="CRF学習率")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="勾配累積ステップ数")
    parser.add_argument("--clip", type=float, default=1.0, help="勾配クリッピング")
    
    # データのパラメータ
    parser.add_argument("--max_utterance_len", type=int, default=128, help="最大発話長")
    parser.add_argument("--max_context_len", type=int, default=384, help="最大コンテキスト長")
    
    # その他
    parser.add_argument("--random_seed", type=int, default=42, help="ランダムシード")
    
    # Focal Loss パラメータ
    parser.add_argument("--class_imbalance_ratio", type=float, default=6.5, 
                       help="クラス不均衡比 (non-initiative:initiative)")
    parser.add_argument("--focal_gamma", type=float, default=2.0, 
                       help="Focal Lossのgammaパラメータ")
    
    # QPP特徴量設定
    parser.add_argument("--qpp_feature_indices", type=int, nargs="+", default=None, 
                       help="使用するQPP特徴量のインデックス")
    parser.add_argument("--qpp_target_feature_index", type=int, default=1, 
                       help="auxiliary_headで予測する特徴量のインデックス")
    parser.add_argument("--qpp_gate_feature_index", type=int, default=1, 
                       help="policy_gatingで使用する特徴量のインデックス")
    parser.add_argument("--qpp_index", type=int, default=1, 
                       help="policy_gatingで使用する特徴量のインデックス")
    
    args = parser.parse_args()
    
    # ランダムシードの設定
    replicability(seed=args.random_seed)
    
    # dataset_typeを自動判定
    if "dev" in args.input_path:
        args.dataset_type = "dev"
    elif "test" in args.input_path:
        args.dataset_type = "test"
    elif "train" in args.input_path:
        args.dataset_type = "train"
    else:
        raise ValueError(f"input_pathからdataset_typeを判定できません: {args.input_path}")
    
    # モデル別にoutputパスを設定（コマンドライン引数で指定されていない場合のみ）
    if args.model == "qpp4sip":
        pattern = args.qpp4sip_pattern
        feature_id = get_qpp_feature_id(args)
        if feature_id:
            if args.output_path == "./output":  # デフォルト値の場合のみ上書き
                args.output_path = f"./output/qpp4sip_{pattern}_{feature_id}"
            if args.saved_model_path == "./checkpoints":  # デフォルト値の場合のみ上書き
                args.saved_model_path = f"./checkpoints/qpp4sip_{pattern}_{feature_id}"
            if args.log_path == "log":  # デフォルト値の場合のみ上書き
                args.log_path = f"./logs/qpp4sip_{pattern}_{feature_id}"
        else:
            if args.output_path == "./output":  # デフォルト値の場合のみ上書き
                args.output_path = f"./output/qpp4sip_{pattern}"
            if args.saved_model_path == "./checkpoints":  # デフォルト値の場合のみ上書き
                args.saved_model_path = f"./checkpoints/qpp4sip_{pattern}"
            if args.log_path == "log":  # デフォルト値の場合のみ上書き
                args.log_path = f"./logs/qpp4sip_{pattern}"
    elif args.model == "music":
        args.output_path = f"./output/music"
        args.saved_model_path = f"./checkpoints/music"
        args.log_path = f"./logs/music"
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.saved_model_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    # バージョン情報の表示
    print("torch_version:{}".format(torch.__version__))
    print("CUDA_version:{}".format(torch.version.cuda))
    print("cudnn_version:{}".format(cudnn.version()))
    
    # モードに応じて実行
    if args.mode == "train":
        run_train(args)
    elif args.mode == "inference":
        run_inference(args)
    elif args.mode == "evaluation":
        run_evaluation(args)
    elif args.mode == "multi_seed_inference":
        run_multi_seed_inference(args)
    else:
        raise NotImplementedError(f"サポートされていないモード: {args.mode}")


if __name__ == "__main__":
    main()
