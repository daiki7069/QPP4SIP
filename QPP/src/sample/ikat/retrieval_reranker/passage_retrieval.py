import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder


def rerank_passages(query, passages, reranker):
    res = []
    query_passage_pairs = [[query, passage['passage_text']] for passage in passages]
    scores = reranker.predict(query_passage_pairs)

    for passage, score in zip(passages, scores):
        passage['reranker_score'] = score
        res.append(passage)

    ranked_passages = sorted(passages, key=lambda x: x['reranker_score'], reverse=True)
    return ranked_passages


def retrieve_using_bm25(query):
    index_path = '/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/index'
    searcher = LuceneSearcher(index_path)
    hits = searcher.search(query)
    candidate_set = []
    for i in range(len(hits)):
        print('Rank: {} | PassageID: {} | Score: {}'.format(i+1, hits[i].docid, hits[i].score))
        doc = searcher.doc(hits[i].docid)
        parsed_doc = json.loads(doc.raw())
        print(parsed_doc['contents'])
        candidate_set.append({
            'passage_id': hits[i].docid,
            'bm25_rank': i+1,
            'bm25_score': hits[i].score,
            'passage_text': parsed_doc['contents']
        })
        print('=================================')
    return candidate_set


def main(query):
    candidate_set = retrieve_using_bm25(query)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    reranked_passages = rerank_passages(query, candidate_set, reranker)
    print(json.dumps(reranked_passages, indent=4, default=lambda o: float(o) if isinstance(o, np.float32) else o))

if __name__ == "__main__":
    query = "Can you compare mozzarella with plant-based cheese?"
    main(query)