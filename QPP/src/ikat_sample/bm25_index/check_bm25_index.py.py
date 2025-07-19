from pyserini.search.lucene import LuceneSearcher
import json


def main(query):
    index_path = '/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/index'   # Luceneインデックスのパス
    searcher = LuceneSearcher(index_path)
    hits = searcher.search(query)

    for i in range(len(hits)):
        print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')

    # 最もスコアが高いドキュメントの内容を表示
    best_ranked_doc = searcher.doc(hits[0].docid)
    parsed_doc = json.loads(best_ranked_doc.raw())
    print(parsed_doc['contents'])


if __name__ == "__main__":
    query = 'global warming'    # 検索クエリ

    main(query)