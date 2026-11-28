"""GLRMask is a grammar-constrained decoding library for high-throughput LLM serving. It is optimized for fast mask computation with extremely low tail latency, even for complex grammars.

GLRMask supports general context-free grammars, including ambiguous ones. It accepts JSON Schema, Lark and [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md).

## Installation

```bash
python -m pip install glrmask
```

## Usage

GLRMask has four core types:

- `Grammar`: a grammar, written as source text or built from an expression.
- `Constraint`: a compiled grammar, ready to use.
- `ConstraintState`: the state of one generated sequence.
- `Module`: a partially compiled grammar ([see below](#composing-grammars)).

In the quickstart, supply the tokenizer's exact token bytes and the model's EOS ID. The model produces logits; the sampling routine must apply the packed mask and permit EOS only when the state is accepting.

## Quickstart

```python
grammar = Grammar.from_json_schema({
    "type": "object",
    "properties": {
        "color": {
            "enum": ["red", "green", "blue"]
        }
    },
    "required": ["color"],
    "additionalProperties": False,
})

# Token ID -> exact bytes from tokenizer
vocab = Vocab(token_bytes)
state = grammar.compile(vocab).start()

output = []
while True:
    logits = model(output)
    mask = state.mask()
    token = sample(
        logits, mask, state.is_accepting()
    )
    if token == EOS:
        break
    if not state.commit_token(token):
        raise RuntimeError("Rejected token")
    output.append(token)
```

## Composing grammars

GLRMask compiles grammars in parts and links them later. A `Module` is a grammar with named externs. Bind each extern to another grammar, a compiled constraint or a token set, then compile. Parts that are already compiled are reused, so linking avoids recompiling them from scratch.

```python
# Token ID -> exact bytes from tokenizer
vocab = Vocab(token_bytes)

weather = Grammar.from_json_schema({
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"],
    "additionalProperties": False,
}).compile(vocab)

parent = Grammar.from_glrm('''
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS
              "</weather>"
''')
call = Module(parent)
call.bind("ARGS", weather)
constraint = call.compile(vocab)
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
