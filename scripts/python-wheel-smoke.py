#!/usr/bin/env python3
"""Minimal public Python API smoke test for an installed glrmask wheel/sdist."""

import ctypes
import sys
import types
import importlib.metadata
from pathlib import Path

import glrmask

# A release smoke must execute the fresh installed distribution.
assert Path(glrmask.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
distribution = importlib.metadata.distribution("glrmask")
metadata = distribution.metadata
assert metadata["Name"] == "glrmask"
expected_description = (Path(__file__).resolve().parents[1] / "python" / "PYTHON.md").read_text(encoding="utf-8")
# Compare the UTF-8 metadata body actually installed from the wheel, not an
# email.Message payload adapter whose decoding policy changes between Python
# releases. Normalize only platform newline spelling, preserving all content.
raw_metadata = distribution.read_text("METADATA")
assert raw_metadata is not None, "Installed distribution has no METADATA"
header, separator, description = raw_metadata.replace("\r\n", "\n").partition("\n\n")
assert separator, "Installed METADATA has no description body"
assert description.strip() == expected_description.strip(), "Installed package description differs from source"
if len(sys.argv) > 1:
    assert metadata["Version"] == sys.argv[1]


vocab = glrmask.Vocab.from_dict(
    {
        b"hello": 0,
        b" ": 1,
        b"world": 2,
    }
)
constraint = glrmask.Grammar.from_lark(
    'start: "hello" " " "world"'
).compile(vocab)
state = constraint.start()

assert state.mask().tolist() == [True, False, False]
assert state.commit_token(0)
assert state.mask().tolist() == [False, True, False]
assert state.commit_token(1)
assert state.mask().tolist() == [False, False, True]
assert state.commit_token(2)
assert state.is_accepting()


# Exercise the optional llama-cpp-python adapter without installing or loading a
# real model. The fake module mirrors the small low-level API surface consumed by
# Vocab.from_llama_cpp.
pieces = {
    0: b"a",
    1: b"a",  # unused: must be omitted
    2: b"a",  # control: must be omitted
    3: b"a",  # EOG: exact-token-only, not a byte token
    4: b"d",
    5: b"a",  # a second EOG token
}
fake_llama_cpp = types.ModuleType("llama_cpp")
fake_llama_cpp.LLAMA_TOKEN_ATTR_UNUSED = 1 << 1
fake_llama_cpp.LLAMA_TOKEN_ATTR_CONTROL = 1 << 3
fake_vocab_handle = object()


def fake_model_get_vocab(model):
    assert model is fake_llm.model
    return fake_vocab_handle


def fake_is_eog(vocab_handle, token_id):
    assert vocab_handle is fake_vocab_handle
    return token_id in (3, 5)


def fake_get_attr(vocab_handle, token_id):
    assert vocab_handle is fake_vocab_handle
    if token_id == 1:
        return fake_llama_cpp.LLAMA_TOKEN_ATTR_UNUSED
    if token_id == 2:
        return fake_llama_cpp.LLAMA_TOKEN_ATTR_CONTROL
    return 0


def fake_token_to_piece(vocab_handle, token_id, buffer, capacity, _lstrip, _special):
    assert vocab_handle is fake_vocab_handle
    piece = pieces[token_id]
    if buffer is None or capacity < len(piece):
        return -len(piece)
    ctypes.memmove(buffer, piece, len(piece))
    return len(piece)


class FakeLlama:
    model = object()

    @staticmethod
    def n_vocab():
        return len(pieces)


fake_llm = FakeLlama()
fake_llama_cpp.llama_model_get_vocab = fake_model_get_vocab
fake_llama_cpp.llama_vocab_is_eog = fake_is_eog
fake_llama_cpp.llama_vocab_get_attr = fake_get_attr
fake_llama_cpp.llama_token_to_piece = fake_token_to_piece
sys.modules["llama_cpp"] = fake_llama_cpp

llama_vocab = glrmask.Vocab.from_llama_cpp(fake_llm)
assert llama_vocab.llama_cpp_end_token_ids == [3, 5]
assert glrmask.TokenSet([2]) == glrmask.TokenSet([2])  # control token: no byte route
control_module = glrmask.Module(glrmask.Grammar.from_glrm(
    'glrm 1; start start; extern token CONTROL; nt start = CONTROL;'
))
assert control_module.bind("CONTROL", glrmask.TokenSet([2])) is None
control_constraint = control_module.compile(llama_vocab)
assert control_constraint.start().mask(size=6).tolist() == [False, False, True, False, False, False]

llama_constraint = glrmask.Grammar.from_expr(glrmask.Expr.sequence(
    glrmask.Grammar.from_lark('start: "a"'),
    glrmask.TokenSet(llama_vocab.llama_cpp_end_token_ids),
)).compile(llama_vocab)
llama_state = llama_constraint.start()
assert llama_state.mask(size=6).tolist() == [True, False, False, False, False, False]
assert llama_state.commit_token(0)
assert llama_state.mask(size=6).tolist() == [False, False, False, True, False, True]
assert llama_state.commit_token(3)
assert llama_state.is_accepting() and llama_state.is_terminated()
assert not llama_state.mask(size=6).any()

print("public Python API smoke test passed")
