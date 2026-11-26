"""Test helpers for inspecting original IDs in packed masks."""
import numpy as np

def token_allowed(words, token_id):
    return token_id // 32 < len(words) and bool(int(words[token_id // 32]) & (1 << (token_id % 32)))

def unpack_mask(words, size=None):
    # Cast to little-endian u32 so the helper also works on big-endian hosts.
    result = np.unpackbits(np.asarray(words, dtype="<u4").view(np.uint8), bitorder="little")
    if size is not None:
        if len(result) < size:
            result = np.pad(result, (0, size - len(result)))
        result = result[:size]
    return result.astype(bool)
