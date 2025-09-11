import sys
sys.path.append('./')

import argparse
from transformers import get_constant_schedule
import os
import torch
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

from model.dataset import Dataset, collate_fn
from model.music_model import BILSTMCRF
from model.qpp4sip_model import QPP4SIPBILSTMCRF
from model.trainer import Trainer
from model.utils import replicability, Config
from model.load_pkl import convert_dialogue_to_conversations


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


def train(args):
    """
    学習を実行する関数
    """
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

    # モデルの初期化
    model = get_model(args)
    
    # GPU使用の設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    model = model.to(device)
    
    # 参考元のようにCRF用の学習率を分けて設定
    if args.model == "music":
        model_optimizer = optim.Adam([
            {"params": model.utterance_encoding.parameters()},
            {"params": model.posterior_conversation_encoding.lstm.parameters()},
            {"params": model.prior_conversation_encoding.lstm.parameters()},
            {"params": model.prior_e_project.parameters()},
            {"params": model.posterior_e_project.parameters()},
            {"params": model.crf.parameters(), "lr": args.lr_crf}
        ], lr=args.learning_rate)
    else:
        model_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model_scheduler = get_constant_schedule(model_optimizer)

    writer = SummaryWriter(args.log_path)

    trainer = Trainer(args, model, writer)
    model_optimizer.zero_grad()

    for i in range(1, args.epoch_num+1):
        dataset = Dataset(args, config, mlb, conversations)
        trainer.train_epoch(dataset, collate_fn, i, model_optimizer, model_scheduler)
        trainer.serialize(i, model_scheduler, saved_model_path=args.saved_model_path)
    writer.close()


def inference(args):
    """
    推論を実行する関数
    """
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
            
            # 推論の実行（trainerクラスのinferメソッドを使用）
            trainer = Trainer(args, model)
            trainer.infer(epoch_id, dataset, collate_fn)
            
            print(f"エポック {epoch_id} の推論が完了しました")
        else:
            print(f"チェックポイントが見つかりません: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基本設定
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], 
                       help="実行モード")
    parser.add_argument("--task", type=str, default="SIP", help="タスク名")
    parser.add_argument("--name", type=str, default="QPP4SIP", help="モデル名")
    parser.add_argument("--dataset", type=str, default="QPP4SIP", help="データセット名")
    parser.add_argument("--model", type=str, default="music", choices=["music", "qpp4sip"], help="モデルタイプ (music: music_model.py, qpp4sip: qpp4sip_model.py)")
    parser.add_argument("--qpp4sip_pattern", type=str, default="feature_fusion", choices=["feature_fusion", "auxiliary_head", "policy_gating"], 
                       help="QPP4SIP実装パターン (feature_fusion: 追加特徴, auxiliary_head: マルチタスク学習, policy_gating: ポリシー連動)")

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
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学習率")
    parser.add_argument("--lr_crf", type=float, default=1e-3, help="CRF学習率")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="勾配累積ステップ数")
    parser.add_argument("--clip", type=float, default=1.0, help="勾配クリッピング")
    
    # データのパラメータ
    parser.add_argument("--max_utterance_len", type=int, default=128, help="最大発話長")
    parser.add_argument("--max_context_len", type=int, default=384, help="最大コンテキスト長")
    
    # その他
    parser.add_argument("--random_seed", type=int, default=42, help="ランダムシード")
    
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
    
    # モデル別にoutputパスを設定
    if args.model == "qpp4sip":
        pattern = args.qpp4sip_pattern
        args.output_path = f"./output/qpp4sip_{pattern}"
        args.saved_model_path = f"./checkpoints/qpp4sip_{pattern}"
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
        train(args)
    elif args.mode == "inference":
        inference(args)
    else:
        raise NotImplementedError(f"サポートされていないモード: {args.mode}")


def main():
    """メインエントリーポイント"""
    args = parse_args()
    run(args)