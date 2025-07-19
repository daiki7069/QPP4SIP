from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sentence_transformers import CrossEncoder
import json


def rewrite_query(context: str, model, tokenizer, device) -> str:
    tokenized_context = tokenizer.encode(context, return_tensors="pt").to(device)
    output_ids = model.generate(
        tokenized_context,
        max_length=200,
        num_beams=4,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    ).to(device)

    rewrite = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return rewrite


def get_ptkb_statements(query, num_ptkb, ptkb, reranker):
    # Find the similarity of PTKB statements with the given query
    similarity_scores = [reranker.predict([[query, ptkb_statement]])[0] for ptkb_statement in ptkb.values()]

    # Pair each statement with its similarity score
    statement_score_pairs = list(zip(list(ptkb.values()), similarity_scores))

    # Sort the pairs based on the similarity scores in descending order
    sorted_pairs = sorted(statement_score_pairs, key=lambda x: x[1], reverse=True)

    # Extract the sorted responses
    sorted_ptkb_statements = [pair[0] for pair in sorted_pairs]

    # Return required number of PTKB statements
    return ' '.join(sorted_ptkb_statements[:num_ptkb])


def extract_context_with_ptkb_statements(json_data, number, turn_id, ptkb_statements):
    # Find the correct dictionary with the given number
    data = None
    for item in json_data:
        if item['number'] == number:
            data = item
            break

    # If we couldn't find the data for the given number
    if not data:
        print("No data found for the given number.")
        return "No data found for the given number."

    # Extract the utterance and response values
    texts = [ptkb_statements]
    current_utterance = ""
    for turn in data['turns']:
        if turn['turn_id'] < turn_id:
            texts.append(turn['utterance'])
            texts.append(turn['response'])
        elif turn['turn_id'] == turn_id:
            current_utterance = turn['utterance']
            texts.append(current_utterance)

    # Join the texts with "|||" separator
    context = '|||'.join(texts)

    return current_utterance, context


def main(query, num_ptkb, ptkb):
    with open('/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/test.json', 'r') as f:
        topics = json.load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rewriter = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard").to(device).eval()
    rewriter_tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    ptkb_statements = get_ptkb_statements(query, num_ptkb, ptkb, reranker)
    print(ptkb_statements)

    number_to_search = "10-1"
    turn_id_to_search = 6
    utterance, context = extract_context_with_ptkb_statements(topics, number_to_search, turn_id_to_search, ptkb_statements)
    print(f"Raw Utterance: {utterance}")
    print(f"Turn Context: {context}")

    rewrite = rewrite_query(context, rewriter, rewriter_tokenizer, device)
    print(f"Query Rewrite: {rewrite}")


if __name__ == "__main__":
    query = "Can you compare the first two?"
    ptkb = {
        "1": "I want to know about healthy cooking techniques.",
        "2": "I am lactose intolerant.",
        "3": "I'm looking for a speaker set to match my TV.",
        "4": "I'm willing to drive a long distance to find a cheaper TV.",
        "5": "I'm hoping to find some offers and discounts for TV.",
        "6": "I like to eat fruits and vegetables.",
        "7": "I don't read much.",
        "8": "I want to cook healthy and tasty recipes for my family.",
        "9": "I am on a diet and prefer low-calorie food.",
        "10": "I want to know about the nutritional value of the ingredients I use.",
        "11": "I'm looking for a new TV to replace my current one.",
        "12": "I want a TV that is okay for light and size of my living room."
    }
    num_ptkb = 3
    main(query, num_ptkb, ptkb)