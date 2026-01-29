import os
import sys
from transformers import AutoTokenizer

def find_longest_token():
    model_id = "Xenova/gpt-4o"
    print(f"Loading tokenizer {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    max_len = 0
    longest_tokens = []

    for token, token_id in vocab.items():
        # Using character length as requested
        # Note: transformers get_vocab keys are usually strings, 
        # but some might represent special tokens.
        token_str = str(token)
        length = len(token_str)
        
        if length > max_len:
            max_len = length
            longest_tokens = [(token_str, token_id)]
        elif length == max_len:
            longest_tokens.append((token_str, token_id))

    print(f"\nMax length: {max_len}")
    print(f"Number of tokens with this length: {len(longest_tokens)}")
    print("\nTop 5 tokens with max length (or all if fewer):")
    for t_str, t_id in longest_tokens[:5]:
        # Escape control characters for safer printing
        safe_str = t_str.encode('unicode_escape').decode('utf-8')
        print(f"ID: {t_id}, Length: {len(t_str)}, Content: '{safe_str}'")

if __name__ == "__main__":
    find_longest_token()
