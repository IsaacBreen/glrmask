#!/usr/bin/env python3
import json
import os
from pathlib import Path

import numpy as np
import _glrmask as gm

FILTER_BYTES = b"()[]{}-+"
GRAMMAR = Path("/tmp/core_lac_correctness_reduced_js.glrm")
VOCAB_JSON = Path(
    "/Users/isaacbreen/Projects2/constraint-framework-analysis/.cache/vocab_cache/llama3_vocab.json"
)
PREFIXES = [
    ("empty", b""),
    ("function keyword", b"function"),
    ("function name", b"function f"),
    ("function args open", b"function f("),
    ("function one arg", b"function f(x"),
    ("function block", b"function f(x) {"),
    ("return expr", b"function f(x) { return x"),
    ("return call", b"function f(x) { return foo(x"),
    ("const assign", b"const x = "),
    ("let array", b"let a = ["),
    ("array items", b"let a = [1, 2"),
    ("object open", b"const o = {"),
    ("object key", b"const o = {foo"),
    ("object colon", b"const o = {foo: "),
    ("if condition", b"if (x"),
    ("if block", b"if (x) {"),
    ("if return", b"if (x) { return "),
    ("nested blocks", b"if (x) { if (y) { return "),
    ("paren expr", b"const y = (x + "),
    ("binary expr", b"x + y"),
    ("call chain", b"console.log("),
    ("double string", b'const s = "'),
    ("single string", b"const s = '"),
    ("string content", b'const s = "abc'),
    ("line comment", b"// hello"),
    ("block comment", b"/* hello"),
    ("regex-ish slash", b"const r = /"),
    ("invalid close paren", b")"),
    ("invalid close brace", b"}"),
    ("invalid close bracket", b"]"),
    ("invalid const equals", b"const ="),
    ("invalid function brace", b"function {"),
    ("unclosed array nested", b"function f(){ return [x, {y: "),
    ("template-ish", b"const t = `"),
]


def filtered_vocab() -> tuple[gm.Vocab, int]:
    raw_vocab = json.loads(VOCAB_JSON.read_text())
    token_to_id = {bytes.fromhex(hex_bytes): int(token_id) for token_id, hex_bytes in raw_vocab.items()}
    filtered = {
        token: token_id
        for token, token_id in token_to_id.items()
        if sum(1 for byte in token if byte in FILTER_BYTES) <= 1
    }
    return gm.Vocab.from_dict(filtered), len(filtered)


def build(grammar: str, vocab: gm.Vocab, flag: str):
    os.environ["GLRMASK_LAZY_NEGATIVE_PARSER_DWA"] = flag
    constraint = gm.Constraint.from_glrm_grammar(grammar, vocab)
    print(
        f"[build] lazy_negative={flag} parser_states={constraint.num_parser_states()} mask_len={constraint.mask_len()}",
        flush=True,
    )
    return constraint


def mask_for(constraint, prefix: bytes):
    state = constraint.start()
    try:
        state.commit_bytes(prefix)
    except Exception as exc:
        return ("reject", str(exc))
    mask = np.zeros(constraint.mask_len(), dtype=np.int32)
    state.fill_mask(mask)
    return ("ok", mask)


def main() -> int:
    grammar = GRAMMAR.read_text()
    vocab, vocab_size = filtered_vocab()
    print(
        f"[setup] grammar={GRAMMAR} glrm_bytes={len(grammar.encode())} vocab_size={vocab_size} module={gm.__file__}",
        flush=True,
    )

    baseline = build(grammar, vocab, "0")
    patched = build(grammar, vocab, "1")
    if baseline.mask_len() != patched.mask_len():
        print(
            f"[mismatch] mask_len baseline={baseline.mask_len()} patched={patched.mask_len()}",
            flush=True,
        )
        return 2

    checked = 0
    rejected = 0
    for label, prefix in PREFIXES:
        left_kind, left_value = mask_for(baseline, prefix)
        right_kind, right_value = mask_for(patched, prefix)
        if left_kind != right_kind:
            print(
                f"[mismatch] label={label!r} prefix={prefix!r} baseline={left_kind}:{left_value!r} patched={right_kind}:{right_value!r}",
                flush=True,
            )
            return 2
        if left_kind == "reject":
            rejected += 1
            print(f"[prefix] label={label!r} status=both_reject err={left_value!r}", flush=True)
            continue
        if not np.array_equal(left_value, right_value):
            diff_words = np.flatnonzero(left_value != right_value)
            first = int(diff_words[0])
            print(
                f"[mismatch] label={label!r} prefix={prefix!r} first_diff_word={first} baseline_word={int(left_value[first])} patched_word={int(right_value[first])}",
                flush=True,
            )
            return 2
        checked += 1
        print(
            f"[prefix] label={label!r} status=match nonzero={int(np.count_nonzero(left_value))}",
            flush=True,
        )

    print(f"[summary] result=ok checked={checked} both_rejected={rejected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())