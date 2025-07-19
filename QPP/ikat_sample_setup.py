import subprocess
import os
import gdown

def run_shell_script(script_path):
    """シェルスクリプトを実行する"""
    try:
        print(f"実行中: {script_path}")
        result = subprocess.run(['bash', script_path], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print("シェルスクリプトが正常に実行されました")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"シェルスクリプトの実行中にエラーが発生しました: {e}")
        print(f"エラー出力: {e.stderr}")
        return False

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))    # 現在のディレクトリを取得
    sample_setup_script = os.path.join(current_dir, 'script', 'sample_setup.sh')   # sample_setup.shのパス
    bm25_index_script = os.path.join(current_dir, 'script', 'sample_create_bm25_index.sh')   # BM25インデックス作成スクリプトのパス
    
    # 1. ディレクトリセットアップ
    print("=== ステップ1: ディレクトリセットアップ ===")
    if not run_shell_script(sample_setup_script):
        print("セットアップスクリプトの実行に失敗しました。処理を中止します。")
        return
    
    # 2. データダウンロード
    print("\n=== ステップ2: データダウンロード ===")
    url = 'https://drive.google.com/uc?id=1touBjwkPByH69utT9_sevr5nYT0TTZ2M'
    output = '/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/collection/simplewiki-2020-11-01.passages.jsonl'
    print(f"ダウンロード中: {url}")
    gdown.download(url, output, quiet=False)

    url = 'https://drive.google.com/uc?id=1zPSiAqLmbx9QFGm6walnuMUl7xoJmRB7'
    output = '/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/test.json'
    print(f"ダウンロード中: {url}")
    gdown.download(url, output, quiet=False)
    
    # 3. BM25インデックス作成
    print("\n=== ステップ3: BM25インデックス作成 ===")
    if not run_shell_script(bm25_index_script):
        print("BM25インデックス作成スクリプトの実行に失敗しました。")
        return
    
    print("\n=== セットアップ完了 ===")
    print("すべての処理が正常に完了しました！")

if __name__ == "__main__":
    main()