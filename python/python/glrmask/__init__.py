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

The next compatibility release targets ordinary, GIL-enabled CPython 3.9
through 3.15. Its `cp39-abi3` wheels use Python's stable extension ABI, so one
wheel per operating-system/architecture combination can serve multiple Python
versions. This does not imply support for free-threaded Python (`cp314t`,
`cp315t` or `abi3t`): the extension explicitly retains the GIL requirement.

The expanded build matrix covers Windows x64/ARM64, macOS Intel/Apple Silicon,
and Linux x64/ARM64 with either glibc or musl. Linux glibc Python wheels retain
the manylinux2014 baseline; native C++/Node binaries use glibc 2.28 or newer.
See [the compatibility guide](https://github.com/IsaacBreen/glrmask/blob/main/docs/compatibility.md)
for runtime requirements and the distinction between source support and
published package availability.

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""

from ._glrmask import *
