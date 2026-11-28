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

For Hugging Face tokenizers, use `Vocab.from_huggingface(tokenizer)` in Python, or `Vocab::from_tokenizer_file(path, None)` in Rust (enable `hf` and import `VocabFromTokenizer`). Pass the model vocabulary size to include padded IDs. For other tokenizers, supply their exact token bytes. The quickstart also needs the model's EOS ID. The model produces logits; the sampling routine must apply the packed mask and permit EOS only when the state is accepting.

## Quickstart

```python
from glrmask import Grammar, Vocab
import numpy as np

# Replace with your tokenizer's token bytes
token_bytes = {i: bytes([i]) for i in range(256)}
EOS = 256
rng = np.random.default_rng(0)
def model(output):
    return rng.standard_normal(257)
def sample(logits, mask, accepting):
    allowed = np.zeros(len(logits), dtype=bool)
    bits = np.unpackbits(mask.astype('<u4').view(np.uint8), bitorder='little')
    allowed[:EOS] = bits[:EOS]
    allowed[EOS] = accepting
    if not allowed.any():
        raise RuntimeError('no allowed token and cannot stop')
    return int(np.argmax(np.where(allowed, logits, -np.inf)))
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
assert state.is_accepting()
print(bytes(output).decode())
```

## Composing grammars

In a typical serving setup, a grammar constraint is one main grammar with smaller grammars nested inside it, some of which, like tool schemas, change from request to request. GLRMask is built to make this kind of composition cheap. Like modules in a program, grammars can be written and compiled as separate parts and combined later.

A server compiles its main grammar once as a `Module` with named externs. Each tool's schema is compiled on its own, in parallel or on another machine. For each request, GLRMask composes the main grammar with the compiled schemas by filling in the externs.

Resolving the interactions between them requires some extra *boundary* work. However, this work is usually small compared with compiling the grammars themselves, so composition remains cheap even when the grammars are large.

```python
from glrmask import Grammar, Module, Vocab
# Replace with your tokenizer's token bytes
token_bytes = {i: bytes([i]) for i in range(256)}
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
