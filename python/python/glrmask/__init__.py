"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON schemas and Lark grammars. GLRM is its internal grammar format.

## Installation

```bash
python -m pip install glrmask
```

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

## Core concepts

GLRMask has four core types:

- `Grammar`: source grammar semantics and bindings.
- `PreparedGrammar`: reusable vocabulary-specific compiled machinery, which can still have unresolved bindings.
- `Constraint`: a closed, runnable grammar compiled against a vocabulary.
- `ConstraintState`: the mutable state for one generated sequence.

Compile a `Grammar` directly, or use `prepare()` and `link()` for reusable components. Create a state with `start()`, then compute the next-token mask, sample an allowed token, and commit it.

`Grammar` and `PreparedGrammar` keep reusable definitions with mutable bindings. `bind()` fills one declared slot on the handle. Clones and previously attached children retain independent binding snapshots; a failed bind leaves the handle unchanged.

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""

from ._glrmask import *
