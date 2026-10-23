"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark, and GLRMask's native format: GLRM.

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: the source-language definition, without compilation settings.
- `Module`: a reusable grammar with bindings and any retained compilation work. It may still have unresolved externs.
- `Constraint`: a fully runnable grammar compiled against a specific vocabulary.
- `ConstraintState`: the mutable state for a single generated sequence.

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
    state.commit_token(token_id)
assert state.is_accepting()
```

Python masks are NumPy Boolean arrays indexed by original model token ID.

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from collections.abc import Mapping
from ._glrmask import (
    BoundaryMaskStrategy,
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

Mapping.register(Externs)

__all__ = [
    "BoundaryMaskStrategy",
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
