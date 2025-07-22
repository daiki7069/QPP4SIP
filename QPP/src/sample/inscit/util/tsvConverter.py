#!/usr/bin/env python3
import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any


class TsvConverter:
    """TSVファイルをJSON形式に変換するクラス"""
    
    def __init__(self, encoding: str = 'utf-8', chunk_size: int = 10000):
        """
        初期化
        
        Args:
            encoding (str): ファイルエンコーディング
            chunk_size (int): チャンクサイズ（行数）
        """
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.data = []
        self.headers = []
    
    def load_tsv_chunked(self, tsv_file_path: str, chunk_index: int = 0) -> bool:
        """
        TSVファイルをチャンク単位で読み込む
        
        Args:
            tsv_file_path (str): TSVファイルのパス
            chunk_index (int): チャンクのインデックス（0から開始）
            
        Returns:
            bool: 読み込み成功時True、失敗時False
        """
        try:
            with open(tsv_file_path, 'r', encoding=self.encoding) as tsv_file:
                # ヘッダー行を読み込み
                header_line = tsv_file.readline().strip()
                self.headers = [h.strip() for h in header_line.split('\t')]
                
                # チャンクの開始位置までスキップ
                start_line = chunk_index * self.chunk_size
                for _ in range(start_line):
                    tsv_file.readline()
                
                # チャンクサイズ分のデータを読み込み
                self.data = []
                for _ in range(self.chunk_size):
                    line = tsv_file.readline()
                    if not line:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # タブで分割
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        row = {
                            'id': parts[0].strip(),
                            'contents': parts[1].strip(),
                            'title': parts[2].strip()
                        }
                        self.data.append(row)
            
            print(f"チャンク {chunk_index} 読み込み完了: {len(self.data)}件のデータ")
            return True
            
        except FileNotFoundError:
            print(f"エラー: ファイル '{tsv_file_path}' が見つかりません")
            return False
        except Exception as e:
            print(f"エラー: {e}")
            return False
    
    def get_total_chunks(self, tsv_file_path: str) -> int:
        """
        ファイルの総チャンク数を取得
        
        Args:
            tsv_file_path (str): TSVファイルのパス
            
        Returns:
            int: 総チャンク数
        """
        try:
            with open(tsv_file_path, 'r', encoding=self.encoding) as tsv_file:
                # ヘッダー行をスキップ
                tsv_file.readline()
                
                # 総行数をカウント
                total_lines = sum(1 for _ in tsv_file)
                total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size
                
                return total_chunks
                
        except Exception as e:
            print(f"エラー: {e}")
            return 0
    
    def load_tsv(self, tsv_file_path: str) -> bool:
        """
        TSVファイルを読み込む
        
        Args:
            tsv_file_path (str): TSVファイルのパス
            
        Returns:
            bool: 読み込み成功時True、失敗時False
        """
        try:
            with open(tsv_file_path, 'r', encoding=self.encoding) as tsv_file:
                # まず通常のCSVリーダーで試行
                tsv_reader = csv.DictReader(tsv_file, delimiter='\t')
                
                # ヘッダーを確認
                if len(tsv_reader.fieldnames) == 1 and ' ' in tsv_reader.fieldnames[0]:
                    # スペース区切りの場合、ファイルをリセットして手動で処理
                    tsv_file.seek(0)
                    lines = tsv_file.readlines()
                    
                    # ヘッダー行を分割
                    header_line = lines[0].strip()
                    self.headers = [h.strip() for h in header_line.split()]
                    
                    # データ行を処理
                    self.data = []
                    for line in lines[1:]:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 行を分割（最初の2つのスペースで分割）
                        parts = line.split('    ', 2)  # 4つのスペースで分割
                        if len(parts) >= 3:
                            row = {
                                'id': parts[0].strip(),
                                'text': parts[1].strip(),
                                'title': parts[2].strip()
                            }
                            # textカラムをcontentsに変更
                            row['contents'] = row.pop('text')
                            self.data.append(row)
                else:
                    # 通常のタブ区切りファイル
                    self.headers = tsv_reader.fieldnames
                    
                    # データを読み込み
                    self.data = []
                    for row in tsv_reader:
                        # textカラムをcontentsに変更
                        if 'text' in row:
                            row['contents'] = row.pop('text')
                        
                        self.data.append(row)
            
            print(f"TSVファイル読み込み完了: {len(self.data)}件のデータ")
            print(f"ヘッダー: {self.headers}")
            return True
            
        except FileNotFoundError:
            print(f"エラー: ファイル '{tsv_file_path}' が見つかりません")
            return False
        except Exception as e:
            print(f"エラー: {e}")
            return False
    
    def get_data(self) -> List[Dict[str, Any]]:
        """
        読み込んだデータを取得
        
        Returns:
            List[Dict[str, Any]]: データのリスト
        """
        return self.data
    
    def get_headers(self) -> List[str]:
        """
        ヘッダー情報を取得
        
        Returns:
            List[str]: ヘッダーのリスト
        """
        return self.headers
    
    def get_row_count(self) -> int:
        """
        データ行数を取得
        
        Returns:
            int: データ行数
        """
        return len(self.data)
    
    def save_json(self, json_file_path: str, pretty_print: bool = True, 
                  output_format: str = 'lines') -> bool:
        """
        JSONファイルに保存
        
        Args:
            json_file_path (str): 出力JSONファイルのパス
            pretty_print (bool): 整形して出力するかどうか
            output_format (str): 出力形式 ('lines' または 'array')
                - 'lines': 1行1つのJSONオブジェクト（JSONL形式）
                - 'array': 配列形式（従来の形式）
            
        Returns:
            bool: 保存成功時True、失敗時False
        """
        try:
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                if output_format == 'lines':
                    # 1行1つのJSONオブジェクト形式（JSONL）
                    for row in self.data:
                        # 並び順を指定: id, contents, title
                        ordered_row = {
                            'id': row.get('id', ''),
                            'contents': row.get('contents', ''),
                            'title': row.get('title', '')
                        }
                        
                        if pretty_print:
                            json.dump(ordered_row, json_file, ensure_ascii=False, indent=2)
                        else:
                            json.dump(ordered_row, json_file, ensure_ascii=False)
                        json_file.write('\n')
                else:
                    # 配列形式（従来の形式）
                    if pretty_print:
                        json.dump(self.data, json_file, ensure_ascii=False, indent=2)
                    else:
                        json.dump(self.data, json_file, ensure_ascii=False)
            
            print(f"JSONファイル保存完了: {json_file_path} (形式: {output_format})")
            return True
            
        except Exception as e:
            print(f"エラー: {e}")
            return False
    
    def convert2json(self, tsv_file_path: str, json_file_path: Optional[str] = None, 
                pretty_print: bool = True, output_format: str = 'lines') -> Optional[str]:
        """
        TSVファイルをJSON形式に変換
        
        Args:
            tsv_file_path (str): 入力TSVファイルのパス
            json_file_path (str): 出力JSONファイルのパス（Noneの場合は自動生成）
            pretty_print (bool): 整形して出力するかどうか
            output_format (str): 出力形式 ('lines' または 'array')
            
        Returns:
            str: 出力JSONファイルのパス（失敗時はNone）
        """
        # TSVファイルを読み込み
        if not self.load_tsv(tsv_file_path):
            return None
        
        # 出力ファイル名が指定されていない場合は自動生成
        if json_file_path is None:
            tsv_path = Path(tsv_file_path)
            json_file_path = tsv_path.with_suffix('.jsonl')
        else:
            json_file_path = Path(json_file_path)
            
            # 出力先がディレクトリの場合、ファイル名を自動生成
            if json_file_path.is_dir() or (not json_file_path.suffix and not json_file_path.exists()):
                tsv_path = Path(tsv_file_path)
                json_file_path = json_file_path / f"{tsv_path.stem}.jsonl"
        
        # 既存ファイルの確認
        if json_file_path.exists():
            print(f"警告: ファイル '{json_file_path}' は既に存在します。")
            return None
        
        # 出力ディレクトリが存在しない場合は作成
        json_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # JSONファイルに保存
        if self.save_json(str(json_file_path), pretty_print, output_format):
            return str(json_file_path)
        else:
            return None
    
    def preview_data(self, max_rows: int = 5) -> None:
        """
        データのプレビューを表示
        
        Args:
            max_rows (int): 表示する最大行数
        """
        if not self.data:
            print("データが読み込まれていません")
            return
        
        print(f"\n=== データプレビュー（最大{max_rows}行）===")
        print(f"ヘッダー: {self.headers}")
        print(f"総行数: {len(self.data)}")
        print("\nデータ:")
        
        for i, row in enumerate(self.data[:max_rows]):
            print(f"行 {i+1}: {row}")
        
        if len(self.data) > max_rows:
            print(f"... 他 {len(self.data) - max_rows} 行")

    def convert2json_chunked(self, tsv_file_path: str, output_dir: str, 
                           pretty_print: bool = True, output_format: str = 'lines') -> bool:
        """
        TSVファイルをチャンク単位でJSON形式に変換
        
        Args:
            tsv_file_path (str): 入力TSVファイルのパス
            output_dir (str): 出力ディレクトリのパス
            pretty_print (bool): 整形して出力するかどうか
            output_format (str): 出力形式 ('lines' または 'array')
            
        Returns:
            bool: 変換成功時True、失敗時False
        """
        # 総チャンク数を取得
        total_chunks = self.get_total_chunks(tsv_file_path)
        if total_chunks == 0:
            return False
        
        print(f"総チャンク数: {total_chunks}")
        
        # 出力ディレクトリを作成
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 各チャンクを処理
        for chunk_index in range(total_chunks):
            print(f"\nチャンク {chunk_index + 1}/{total_chunks} を処理中...")
            
            # チャンクを読み込み
            if not self.load_tsv_chunked(tsv_file_path, chunk_index):
                print(f"チャンク {chunk_index} の読み込みに失敗しました")
                continue
            
            # 出力ファイル名を生成
            tsv_path = Path(tsv_file_path)
            json_file_path = output_path / f"{tsv_path.stem}_chunk_{chunk_index:04d}.jsonl"
            
            # JSONファイルに保存
            if not self.save_json(str(json_file_path), pretty_print, output_format):
                print(f"チャンク {chunk_index} の保存に失敗しました")
                continue
            
            print(f"チャンク {chunk_index} 完了: {json_file_path}")
        
        print(f"\n全チャンクの処理が完了しました。出力先: {output_dir}")
        return True


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='TSVファイルをJSON形式に変換します')
    parser.add_argument('input_file', help='入力TSVファイルのパス')
    parser.add_argument('-o', '--output', help='出力JSONファイルのパス（省略時は自動生成）')
    parser.add_argument('-e', '--encoding', default='utf-8', help='ファイルエンコーディング（デフォルト: utf-8）')
    parser.add_argument('-p', '--preview', action='store_true', help='データのプレビューを表示')
    parser.add_argument('--no-pretty', action='store_true', help='JSONを整形せずに出力')
    parser.add_argument('-f', '--format', choices=['lines', 'array'], default='lines',
                       help='出力形式: lines=1行1つのJSONオブジェクト, array=配列形式（デフォルト: lines）')
    parser.add_argument('--chunked', default=True, action='store_true', help='大きなファイルをチャンク単位で処理')
    parser.add_argument('--chunk-size', type=int, default=10000, help='チャンクサイズ（行数、デフォルト: 10000）')
    
    args = parser.parse_args()
    
    # コンバーターを作成
    convert2jsoner = TsvConverter(args.encoding, args.chunk_size)
    
    # プレビューオプションが指定されている場合
    if args.preview:
        if convert2jsoner.load_tsv(args.input_file):
            convert2jsoner.preview_data()
        return
    
    # チャンキング処理が指定されている場合
    if args.chunked:
        if args.output is None:
            print("エラー: チャンキング処理では出力ディレクトリを指定してください（-o オプション）")
            return
        
        success = convert2jsoner.convert2json_chunked(args.input_file, args.output, not args.no_pretty, args.format)
        if success:
            print("チャンキング変換が正常に完了しました。")
        else:
            print("チャンキング変換に失敗しました。")
        return
    
    # 通常の変換実行
    result = convert2jsoner.convert2json(args.input_file, args.output, not args.no_pretty, args.format)
    
    if result:
        print("変換が正常に完了しました。")
    else:
        print("変換に失敗しました。")


if __name__ == "__main__":
    main() 