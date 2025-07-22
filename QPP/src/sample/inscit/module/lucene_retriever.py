import json
from typing import List, Dict, Any
from pyserini.search.lucene import LuceneSearcher


class LuceneRetriever:
    """
    Lucene検索エンジンを使用して文書を検索・取得するクラス
    """
    
    def __init__(self, index_path: str):
        """
        Lucene検索エンジンを初期化
        
        Args:
            index_path: Luceneインデックスのパス
        """
        self.index_path = index_path
        self.searcher = LuceneSearcher(index_path)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        クエリに対してLucene検索を実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する文書数
            
        Returns:
            検索結果のリスト。各要素は以下のキーを持つ辞書:
            - passage_id: 文書ID
            - passage_text: 文書内容
            - passage_titles: タイトルのリスト（[SEP]で分割）
        """
        # Lucene検索を実行
        hits = self.searcher.search(query, top_k)
        
        results = []
        for hit in hits:
            # 文書の内容を取得
            doc = self.searcher.doc(hit.docid)
            
            # 文書の内容を解析（JSON形式を想定）
            try:
                doc_content = json.loads(doc.raw())
                contents = doc_content.get('contents', '')
                title = doc_content.get('title', '')
            except (json.JSONDecodeError, AttributeError):
                # JSONでない場合はrawテキストを使用
                contents = doc.raw() if hasattr(doc, 'raw') else str(doc)
                title = doc.get('title', '') if hasattr(doc, 'get') else ''
            
            # タイトルを[SEP]で分割してリスト化
            titles = title.split('[SEP]') if '[SEP]' in title else [title]
            titles = [title.strip() for title in titles if title.strip()]
            
            results.append({
                'passage_id': hit.docid,
                'passage_text': contents,
                'passage_titles': titles,
                'score': hit.score
            })
        
        return results 