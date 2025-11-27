import typing as _t

_INVALID_REP = (1 << 32) - 1


def _bytes_from_int_list(values: _t.Sequence[int]) -> bytes:
    return bytes(values)


def extract_id_to_token_map(data: _t.Mapping[str, _t.Any]) -> dict[int, bytes]:
    """Return a mapping from original LLM token ID to representative bytes."""
    legacy_vocab = data.get("original_llm_vocab")
    if legacy_vocab and "llm_token_map" in legacy_vocab:
        return {v: bytes(k) for k, v in legacy_vocab.get("llm_token_map", [])}

    commit_vocab = data.get("commit_vocab")
    if not commit_vocab:
        return {}

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
