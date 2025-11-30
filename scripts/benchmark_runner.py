def parse_len_ranges(ranges: list[str] | None) -> tuple[set[int], int | None]:
    """
    Parses a list of string representations of integer ranges.
    e.g., ["1", "3-5", "8-"] -> ({1, 3, 4, 5}, 8)
    """
    if not ranges:
        return set(), None

    allowed_lengths = set()
    min_len_unbounded = None

    for r in ranges:
        if '-' in r:
            parts = r.split('-', 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid range format: {r}")
            start_str, end_str = parts

            if not start_str:
                raise ValueError(f"Invalid range format: {r}. Start must be specified.")

            try:
                start = int(start_str)
            except ValueError:
                raise ValueError(f"Invalid start of range in '{r}'")

            if not end_str: # e.g. "8-"
                if min_len_unbounded is not None:
                    min_len_unbounded = min(min_len_unbounded, start)
                else:
                    min_len_unbounded = start
            else: # e.g. "3-5"
                try:
                    end = int(end_str)
                except ValueError:
                    raise ValueError(f"Invalid end of range in '{r}'")
                if start > end:
                    raise ValueError(f"Invalid range: start ({start}) > end ({end}) in '{r}'")
                allowed_lengths.update(range(start, end + 1))
        else:
            try:
                allowed_lengths.add(int(r))
            except ValueError:
                raise ValueError(f"Invalid length value: {r}")

    return allowed_lengths, min_len_unbounded

def filter_vocab(vocab: dict[str, int], allowed_lengths: set[int], min_len_unbounded: int | None) -> dict[str, int]:
    """
    Applies filters to the vocabulary based on token byte length.
    """
    if not allowed_lengths and min_len_unbounded is None:
        return vocab

    print(f"Filtering vocabulary by token byte length...")

    filtered = {}
    for token_str, token_id in vocab.items():
        # Convert GPT-2 byte-level BPE Unicode characters to actual bytes
        processed_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n").replace("ĉ", "\t").replace("č", "\r")
        token_len = len(processed_str.encode('utf-8'))

        keep = False
        if token_len in allowed_lengths:
            keep = True
        elif min_len_unbounded is not None and token_len >= min_len_unbounded:
            keep = True

        if keep:
            filtered[token_str] = token_id

    print(f"  -> Filtered vocabulary from {len(vocab)} to {len(filtered)} tokens.")
    return filtered


def bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping from byte values to unicode strings for GPT-2 byte-level BPE.
    See: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
