import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher

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


def generate_response(passages, model, tokenizer):
    text = ' '.join(passages)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs,
            max_length=250,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main(query):
    summarizer = AutoModelForSeq2SeqLM.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')
    summarizer_tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-summarize-news')
    
    # We use the top-3 reranked passages to generate a response
    candidate_set = retrieve_using_bm25(query)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    reranked_passages = rerank_passages(query, candidate_set, reranker)
    passages = [passage['passage_text'] for passage in reranked_passages][:3]
    print(json.dumps(passages, indent=4))

    generate_response(passages, summarizer, summarizer_tokenizer)


if __name__ == "__main__":
    query = "Can you compare mozzarella with plant-based cheese?"
    main(query)