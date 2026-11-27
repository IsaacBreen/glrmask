"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark and [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md).

## Installation

```bash
python -m pip install glrmask
```

For the release candidate: `pip install --pre glrmask` or `pip install glrmask==1.0.0rc1`.

## Usage

GLRMask has four core types:

- `Grammar`
- `Module`: a partially compiled grammar.
- `Constraint`: a compiled grammar, ready to use.
- `ConstraintState`: the state of one generated sequence.

## Quickstart

```python
import numpy as np
import glrmask

grammar = glrmask.Grammar.from_json_schema({
    "type": "object",
    "properties": {"color": {"enum": ["red", "green", "blue"]}},
    "required": ["color"],
    "additionalProperties": False,
})

# One token per byte, plus EOS. With a real model, build this from its tokenizer.
vocab = glrmask.Vocab({i: bytes([i]) for i in range(256)})
EOS = 256

state = grammar.compile(vocab).start()
rng = np.random.default_rng(0)
output = []
while True:
    logits = rng.standard_normal(257)  # stand-in for your model
    allowed = np.zeros(257, dtype=bool)
    m = state.mask()
    # NumPy boilerplate: unpack the bitmask into booleans to index the logits.
    allowed[:state.mask_bit_len] = np.unpackbits(m.astype('<u4').view(np.uint8), bitorder='little')[:state.mask_bit_len].astype(bool)
    allowed[EOS] = state.is_accepting()
    if not allowed.any():
        raise RuntimeError("no allowed token and cannot stop")
    token = int(np.argmax(np.where(allowed, logits, -np.inf)))
    if token == EOS:
        break
    if not state.commit_token(token):
        raise RuntimeError("allowed token rejected")
    output.append(token)

if not state.is_accepting():
    raise RuntimeError("incomplete output")
print(bytes(output).decode())  # e.g. {"color": "blue"}
```

## Learn more

[Full guide](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md) · [API reference](https://glrmask.com/api/python/)
"""
from collections.abc import Mapping as _Mapping
from ._glrmask import (
    Vocab,
    Grammar,
    Constraint,
    ConstraintState,
    Module,
    Expr,
    Extern,
    Externs,
    TokenSet,
    MaskStrategy,
    GLRMaskError,
    __version__,
    _internal,
)

_Mapping.register(Externs)

__all__ = [
    "Vocab",
    "Grammar",
    "Constraint",
    "ConstraintState",
    "Module",
    "Expr",
    "Extern",
    "Externs",
    "TokenSet",
    "MaskStrategy",
    "GLRMaskError",
    "__version__",
]
