import json
from types import SimpleNamespace

import pytest
from glrmask import Vocab, Grammar, Expr, TokenSet


def tokenizer_json():
    return json.dumps({
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [
            {"id": 2, "content": "<special>", "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 3, "content": "<ordinary>", "single_word": False,
             "lstrip": False, "rstrip": False, "normalized": False, "special": False},
        ],
        "normalizer": None, "pre_tokenizer": None, "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "",
                  "fuse_unk": False, "vocab": {"a": 0, "ÿ": 1}, "merges": []},
    })


def test_hf_vocab_duck_typed():
    backend = SimpleNamespace(to_str=tokenizer_json)
    for vocab in [Vocab.from_tokenizer_json(tokenizer_json(), 5),
                  Vocab.from_huggingface(backend, 5),
                  Vocab.from_huggingface(SimpleNamespace(backend_tokenizer=backend), 5)]:
        for token, spelling in [(0, b"a"), (1, b"\xff"), (3, b"<ordinary>")]:
            state = Grammar.from_expr(Expr.literal(spelling)).compile(vocab).start()
            assert state.commit_token(token)
            assert state.is_accepting()
        for token in (2, 4):
            state = Grammar.from_expr(Expr.sequence(TokenSet([token]))).compile(vocab).start()
            assert state.commit_token(token)
            assert state.is_accepting()


def test_hf_vocab_invalid():
    with pytest.raises(Exception, match="smaller"):
        Vocab.from_tokenizer_json(tokenizer_json(), 3)
    with pytest.raises(Exception):
        Vocab.from_tokenizer_json("not JSON")
