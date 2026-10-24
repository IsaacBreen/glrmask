"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark, and GLRMask's native format: GLRM.

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Module`: a grammar plus its extern bindings and any precompiled work. Can be saved and loaded.
- `Constraint`: a grammar compiled for a vocabulary, ready to use.
- `ConstraintState`: the state of one generated sequence.

## Quickstart

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({0: b"hello", 1: b" ", 2: b"world"})
grammar = glrmask.Grammar.from_ebnf('start ::= "hello" " " "world"')
constraint = grammar.compile(vocab)
state = constraint.start()
for token_id, expected in [(0, [True, False, False]),
                           (1, [False, True, False]),
                           (2, [False, False, True])]:
    assert state.mask().tolist() == expected
    assert state.commit_token(token_id)
assert state.is_accepting()
```

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from collections.abc import Mapping as _Mapping
from ._glrmask import (
    BoundaryMaskStrategy,
    ParserBackend,
    Constraint,
    ConstraintState,
    TokenSet,
    Expr,
    Extern,
    Externs,
    Grammar,
    MaskStrategy,
    Module,
    Vocab,
    _internal,
)

_Mapping.register(Externs)

__all__ = [
    "BoundaryMaskStrategy",
    "ParserBackend",
    "Constraint",
    "ConstraintState",
    "TokenSet",
    "Expr",
    "Extern",
    "Externs",
    "Grammar",
    "MaskStrategy",
    "Module",
    "Vocab",
    "_internal",
]
