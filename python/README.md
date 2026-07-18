# GLRMask for Python

Extremely fast grammar-constrained decoding for LLMs.

The `glrmask` package compiles a grammar together with a model vocabulary and exposes an incremental next-token mask for use inside a decoding loop.

## Installation

```bash
python -m pip install glrmask
```

Published wheels include the native extension and support Python 3.9 through 3.13.

## Quickstart

```python
import glrmask

vocab = glrmask.Vocab.from_dict({
    b"hello": 0,
    b" ": 1,
    b"world": 2,
})
constraint = glrmask.Constraint.from_ebnf(
    'start ::= "hello" " " "world"',
    vocab,
)
state = constraint.start()

assert state.mask().tolist() == [True, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, False]
state.commit_token(1)
assert state.mask().tolist() == [False, False, True]
state.commit_token(2)
assert state.is_finished()
```

`state.mask()` returns a NumPy Boolean array indexed by model token ID. Pass `state.mask(size)` when the model's logits vector is larger than the highest token ID in the vocabulary.

## Core API

### Vocabulary

Create a vocabulary from either token bytes to token IDs or token IDs to bytes:

```python
vocab = glrmask.Vocab.from_dict({b"yes": 0, b"no": 1})
vocab = glrmask.Vocab.from_id_to_bytes({0: b"yes", 1: b"no"})
```

Tokens are matched by bytes, not decoded Unicode strings.

For `llama-cpp-python`, construct the vocabulary directly from a `Llama` instance:

```python
from llama_cpp import Llama

llm = Llama(model_path="model.gguf", logits_all=True)
vocab = glrmask.Vocab.from_llama_cpp(llm)
end_token_ids = vocab.llama_cpp_end_token_ids
```

The constructor excludes EOG, control, unused, and empty-piece tokens from the byte vocabulary. Pass `end_token_ids` to the constraint constructor when those tokens should terminate generation.

### Compile a constraint

`Constraint` supports JSON Schema, GLRM, Lark, and EBNF:

```python
constraint = glrmask.Constraint.from_json_schema(schema, vocab)
constraint = glrmask.Constraint.from_glrm_grammar(grammar, vocab)
constraint = glrmask.Constraint.from_lark(grammar, vocab)
constraint = glrmask.Constraint.from_ebnf(grammar, vocab)
```

Each constructor accepts an optional `end_token_ids=[...]` argument.

### Decode

Create one state per generated sequence:

```python
state = constraint.start()

while generating:
    mask = state.mask(model_vocab_size)
    token_id = sample_with_mask(logits, mask)
    state.commit_token(token_id)
```

The main state operations are:

- `mask(size=None)`: return the allowed-token mask.
- `commit_token(token_id)`: advance by one model token.
- `commit_tokens(token_ids)`: advance by several model tokens.
- `commit_bytes(data)`: advance by raw bytes.
- `forced()`: return a forced token sequence when one can be determined.
- `is_complete()` and `is_finished()`: report whether the grammar has completed.
- `is_failed()`: report whether no valid parser state remains.

Enable bounded token rollback with `constraint.start(max_rollback_tokens=N)`, then call `state.rollback(count)`. `state.validate_tokens(token_ids)` returns the longest valid prefix without modifying the state.

## Cache compiled constraints

`Constraint` objects are immutable and reusable across requests. Serialize them with `save()` and restore them with `load()`:

```python
artifact = constraint.save()
constraint = glrmask.Constraint.load(artifact, vocab)
```

For complex constraints, compilation typically takes a few hundred milliseconds. To minimize cold-start latency on cache miss, use `DynamicConstraint`. It has the same grammar constructors and produces identical masks, but compiles much faster at the cost of higher mask-generation latency. Compile and cache the corresponding `Constraint` separately for subsequent requests.

```python
constraint = glrmask.DynamicConstraint.from_json_schema(schema, vocab)
state = constraint.start()
```

`DynamicConstraint` and its state also support `save()`, `load()`, `mask()`, `commit_token()`, `commit_tokens()`, `commit_bytes()`, `forced()`, `is_complete()`, and `is_finished()`.

## Grammar formats

GLRM is GLRMask's native EBNF-like syntax and supports exact model-token terminals with `@token(<id>)`. The package also accepts Lark and EBNF grammars, plus JSON Schema through `from_json_schema()`.

See the [root README](../README.md#grammar-formats) for the format overview and special-token examples.

## Source builds

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install ./python
```

Building from source requires a Rust toolchain and the platform's native linker and build tools. On Windows, activate the environment with `.venv\Scripts\activate`.
