import json
import sys
import os

def find_longest_in_json(filepath, label):
    print(f"\n--- Analyzing {label} ({filepath}) ---")
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    # Different tokenizer.json formats might store vocab differently
    # Usually it's in model['vocab'] or just data['vocab']
    vocab = {}
    if 'model' in data and 'vocab' in data['model']:
        vocab = data['model']['vocab']
    elif 'vocab' in data:
        vocab = data['vocab']
    else:
        # Fallback: search for something that looks like a vocab
        for key, value in data.items():
            if isinstance(value, dict) and len(value) > 1000:
                vocab = value
                break
    
    if not vocab:
        print("Could not find vocabulary in JSON structure.")
        return

    print(f"Vocabulary size: {len(vocab)}")

    max_len = 0
    longest_tokens = []
    
    max_len_non_space = 0
    longest_non_space_tokens = []

    for token, token_id in vocab.items():
        token_str = str(token)
        # HF tokenizers often use byte-level representations like Ġ for space
        # But for character length, we'll just count the string length
        length = len(token_str)
        
        if length > max_len:
            max_len = length
            longest_tokens = [(token_str, token_id)]
        elif length == max_len:
            longest_tokens.append((token_str, token_id))
            
        # Check if it's purely whitespace or common space markers
        is_space_only = token_str.isspace() or all(c in ["Ġ", " ", "\u2581", "\u0120", "\u0100", "\t", "\n", "\r"] for c in token_str)
        
        if not is_space_only:
            if length > max_len_non_space:
                max_len_non_space = length
                longest_non_space_tokens = [(token_str, token_id)]
            elif length == max_len_non_space:
                longest_non_space_tokens.append((token_str, token_id))

    print(f"\nOverall Max length: {max_len}")
    print(f"Number of tokens with this length: {len(longest_tokens)}")
    for t_str, t_id in longest_tokens[:5]:
        safe_str = t_str.encode('unicode_escape').decode('utf-8')
        print(f"ID: {t_id}, Length: {len(t_str)}, Content: '{safe_str}'")

    print(f"\nMax length (non-space): {max_len_non_space}")
    print(f"Number of tokens with this length: {len(longest_non_space_tokens)}")
    print("\nTop 5 non-space tokens with max length:")
    for t_str, t_id in longest_non_space_tokens[:5]:
        safe_str = t_str.encode('unicode_escape').decode('utf-8')
        print(f"ID: {t_id}, Length: {len(t_str)}, Content: '{safe_str}'")

if __name__ == "__main__":
    find_longest_in_json("gemma3_tokenizer.json", "Gemma 3 (Gemini Proxy)")
    find_longest_in_json("claude_tokenizer.json", "Claude (Approximation)")
