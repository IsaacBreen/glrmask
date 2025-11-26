from typing import Dict, List, Tuple

def greedy_tokenizer(text_bytes: bytes, id_to_token: Dict[int, bytes]) -> List[int]:
    """
    Tokenizes text_bytes using a greedy longest-match strategy against id_to_token.
    Returns a list of token IDs.
    """
    # Build a Trie for fast prefix matching.
    # The key '<ID>' stores the token ID for a complete token.
    trie_root = {}
    for token_id, token_bytes in id_to_token.items():
        node = trie_root
        for byte_val in token_bytes:
            node = node.setdefault(byte_val, {})
        node['<ID>'] = token_id

    tokens = []
    pos = 0
    while pos < len(text_bytes):
        # Find the longest possible token match starting at the current position.
        node = trie_root
        longest_match_id = -1
        longest_match_len = 0
        
        # Traverse the Trie with bytes from the current position.
        for i in range(len(text_bytes) - pos):
            current_byte = text_bytes[pos + i]
            if current_byte in node:
                node = node[current_byte]
                if '<ID>' in node:
                    # Found a valid token, record it and keep searching for a longer one.
                    longest_match_id = node['<ID>']
                    longest_match_len = i + 1
            else:
                # No further matches possible from this prefix.
                break
        
        if longest_match_len > 0:
            tokens.append(longest_match_id)
            pos += longest_match_len
        else:
            # Fallback: if no token matches, skip one byte (or raise error)
            # For benchmarking, we expect valid input, so raising error is appropriate
            # but to be robust we might just skip.
            # Let's raise for now to catch issues.
            raise ValueError(f"Failed to tokenize. No token found for prefix: {text_bytes[pos:pos+20]!r}")
            
    return tokens
