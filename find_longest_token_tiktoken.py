import tiktoken

def find_longest_token():
    encoding_names = ["o200k_base", "o200k_harmony"]
    for encoding_name in encoding_names:
        print(f"\n--- Loading encoding {encoding_name} ---")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Error loading encoding: {e}")
        return

    # In tiktoken, we can't directly list all tokens if they are not in the mergeable_ranks
    # But o200k_base should have a standard vocab.
    # For o200k_base, the vocab is often stored in internal attributes or we can iterate over potential IDs.
    
    # Actually, tiktoken's Encoding object has _mergeable_ranks and _special_tokens
    # We can combine these.
    
    vocab = {}
    if hasattr(encoding, '_mergeable_ranks'):
        # keys are bytes
        for b, rank in encoding._mergeable_ranks.items():
            vocab[b.decode('utf-8', errors='replace')] = rank
            
    if hasattr(encoding, '_special_tokens'):
        for s, rank in encoding._special_tokens.items():
            vocab[s] = rank

    if not vocab:
        print("Could not retrieve vocabulary from tiktoken encoding.")
        return

    print(f"Vocabulary size: {len(vocab)}")

    max_len = 0
    longest_tokens = []
    
    max_len_non_space = 0
    longest_non_space_tokens = []

    for token_str, token_id in vocab.items():
        length = len(token_str)
        
        if length > max_len:
            max_len = length
            longest_tokens = [(token_str, token_id)]
        elif length == max_len:
            longest_tokens.append((token_str, token_id))
            
        if not token_str.isspace():
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
    find_longest_token()
