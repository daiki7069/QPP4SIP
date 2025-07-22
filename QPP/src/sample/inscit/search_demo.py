#!/usr/bin/env python3
import json
import argparse
from pyserini.search.lucene import LuceneSearcher


def search_demo(index_path: str, top_k: int = 5):
    """
    対話的な検索デモ
    
    Args:
        index_path (str): Luceneインデックスのパス
        top_k (int): 取得する上位結果数
    """
    try:
        # セッチャーを初期化
        searcher = LuceneSearcher(index_path)
        print(f"インデックス読み込み完了: {index_path}")
        print(f"取得結果数: {top_k}")
        print("-" * 50)
        
        while True:
            # クエリ入力
            query = input("\n検索クエリを入力してください (終了: quit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("検索を終了します。")
                break
            
            if not query:
                print("クエリを入力してください。")
                continue
            
            try:
                # 検索実行
                hits = searcher.search(query, k=top_k)
                
                print(f"\n検索結果: '{query}'")
                print("=" * 50)
                
                if not hits:
                    print("該当する結果が見つかりませんでした。")
                    continue
                
                # 結果表示
                for i, hit in enumerate(hits, 1):
                    print(f"\n{i}. スコア: {hit.score:.4f}")
                    print(f"   ID: {hit.docid}")
                    
                    # ドキュメント内容を取得
                    doc = searcher.doc(hit.docid)
                    parsed_doc = json.loads(doc.raw())
                    
                    # 内容を表示
                    if 'contents' in parsed_doc:
                        contents = parsed_doc['contents']
                        # 長すぎる場合は切り詰め
                        if len(contents) > 200:
                            contents = contents[:200] + "..."
                        print(f"   内容: {contents}")
                    
                    if 'title' in parsed_doc:
                        print(f"   タイトル: {parsed_doc['title']}")
                
            except Exception as e:
                print(f"検索エラー: {e}")
    
    except Exception as e:
        print(f"インデックス読み込みエラー: {e}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='対話的なBM25検索デモ')
    parser.add_argument('--index_path', 
                       default='/mnt/disk6/daiki/Datasets/INSCIT/models/DPR/retrieval_data/wikipedia/index/',
                       help='Luceneインデックスのパス（デフォルト: INSCIT Wikipedia index）')
    parser.add_argument('-k', '--topk', type=int, default=5, help='取得する上位結果数（デフォルト: 5）')
    
    args = parser.parse_args()
    
    search_demo(args.index_path, args.topk)


if __name__ == "__main__":
    main() 