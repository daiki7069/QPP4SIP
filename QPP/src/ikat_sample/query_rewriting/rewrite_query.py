# Load model and tokenizer from HuggingFace
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json


def extract_context(json_data, number, turn_id):
    # Find the correct dictionary with the given number
    data = None
    for item in json_data:
        if item['number'] == number:
            data = item
            break

    # If we couldn't find the data for the given number
    if not data:
      print("No data found for the given number.")
      return "No data found for the given number.", None

    # Extract the utterance and response values
    texts = []
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rewriter = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard").to(device).eval()
    rewriter_tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")

    with open('/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/test.json', 'r') as f:
        topics = json.load(f)

    number_to_search = "10-1"
    turn_id_to_search = 6
    utterance, context = extract_context(topics, number_to_search, turn_id_to_search)
    print(f"Raw Utterance: {utterance}")
    print(f"Turn Context: {context}")

    rewrite = rewrite_query(context, rewriter, rewriter_tokenizer, device)
    print(f"Raw Utterance: {utterance}")
    print(f"Query Rewrite: {rewrite}")


if __name__ == "__main__":
    main()