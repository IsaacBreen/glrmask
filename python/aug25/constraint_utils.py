import typing as _t

_INVALID_REP = (1 << 32) - 1


def _bytes_from_int_list(values: _t.Sequence[int]) -> bytes:
    return bytes(values)


def _extract_from_trie(trie_node: dict, prefix: list[int], result: dict[int, bytes]) -> None:
    """Recursively extract token IDs and bytes from a trie structure."""
    if "_" in trie_node:
        # This node has a token ID
        token_id = trie_node["_"]
        if isinstance(token_id, (int, float)):
            result[int(token_id)] = bytes(prefix)
    
    for key, child in trie_node.items():
        if key == "_":
            continue
        try:
            byte_val = int(key)
            if 0 <= byte_val <= 255:
                _extract_from_trie(child, prefix + [byte_val], result)
        except ValueError:
            continue


def extract_id_to_token_map(data: _t.Mapping[str, _t.Any]) -> dict[int, bytes]:
    """Return a mapping from original LLM token ID to token bytes.
    
    Supports multiple formats:
    - New trie format (vocab_trie with "trie" key)
    - Hex tokens format (vocab_trie with "tokens" dict mapping hex bytes to token ID)
    - Legacy commit_vocab format (representatives + original_to_representative)
    - Very old original_llm_vocab format
    """
    # Try new trie format first
    vocab_trie = data.get("vocab_trie")
    if vocab_trie:
        # New trie format with nested structure
        if "trie" in vocab_trie:
            result: dict[int, bytes] = {}
            _extract_from_trie(vocab_trie["trie"], [], result)
            return result
        
        # Hex tokens format: tokens is a dict mapping hex-encoded bytes to token IDs
        if "tokens" in vocab_trie:
            tokens = vocab_trie["tokens"]
            result: dict[int, bytes] = {}
            for hex_bytes, token_id in tokens.items():
                try:
                    byte_seq = bytes.fromhex(hex_bytes)
                    result[token_id] = byte_seq
                except ValueError:
                    continue
            return result
    
    # Fall back to legacy commit_vocab format
    commit_vocab = data.get("commit_vocab")
    if commit_vocab:
        representatives = [
            _bytes_from_int_list(rep)
            for rep in commit_vocab.get("representatives", [])
        ]
        mapping = commit_vocab.get("original_to_representative", [])

        id_to_token: dict[int, bytes] = {}
        for original_id, rep_idx in enumerate(mapping):
            if rep_idx is None or rep_idx == _INVALID_REP:
                continue
            if not isinstance(rep_idx, int):
                continue
            if rep_idx < 0 or rep_idx >= len(representatives):
                continue
            id_to_token[original_id] = representatives[rep_idx]
        return id_to_token
    
    # Very old format
    legacy_vocab = data.get("original_llm_vocab")
    if legacy_vocab and "llm_token_map" in legacy_vocab:
        return {v: bytes(k) for k, v in legacy_vocab.get("llm_token_map", [])}
    
    return {}
