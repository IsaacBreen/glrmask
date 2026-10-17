"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts GLRM (its native grammar format), JSON Schema, and Lark.

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: the language you want to generate.
- `PreparedGrammar`: vocabulary-specific reusable machinery, which can still have unresolved bindings.
- `Constraint`: a grammar compiled against a specific vocabulary.
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

from ._glrmask import *
