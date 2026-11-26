#!/usr/bin/env python3
"""Minimal public Python API smoke test for an installed glrmask wheel/sdist."""

import sys
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


vocab = glrmask.Vocab({0: b"hello", 1: b" ", 2: b"world"})
constraint = glrmask.Grammar.from_lark(
    'start: "hello" " " "world"'
).compile(vocab)
state = constraint.start()

assert state.mask().tolist() == [1]
assert state.commit_token(0)
assert state.mask().tolist() == [2]
assert state.commit_token(1)
assert state.mask().tolist() == [4]
assert state.commit_token(2)
assert state.is_accepting()


# Tokens without byte spellings can still be bound explicitly.
llama_vocab = glrmask.Vocab({0: b"a", 4: b"d"})
control_module = glrmask.Module(glrmask.Grammar.from_glrm(
    'glrm 1; start start; extern token CONTROL; nt start = CONTROL;'
))
assert control_module.bind("CONTROL", glrmask.TokenSet([2])) is None
control_constraint = control_module.compile(llama_vocab)
assert control_constraint.start().mask().tolist() == [4]

# Token IDs depend on the model, so expressions containing them are bound on a Module.
llama_module = glrmask.Module(glrmask.Grammar.from_glrm(
    'glrm 1; start start; extern grammar VALUE; nt start = VALUE;'
))
llama_module.bind("VALUE", glrmask.Expr.sequence(
    glrmask.Grammar.from_lark('start: "a"'),
    glrmask.TokenSet([3, 5]),
))
llama_constraint = llama_module.compile(llama_vocab)
llama_state = llama_constraint.start()
assert llama_state.mask().tolist() == [1]
assert llama_state.commit_token(0)
assert llama_state.mask().tolist() == [40]
assert llama_state.commit_token(3)
assert llama_state.is_accepting() and llama_state.is_terminated()
assert not llama_state.mask().any()

print("public Python API smoke test passed")
