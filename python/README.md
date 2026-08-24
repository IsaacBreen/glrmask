# GLRMask for Python

Extremely fast grammar-constrained decoding for LLMs.

The `glrmask` package compiles a grammar together with a model vocabulary and exposes an incremental next-token mask for use inside a decoding loop.

## Allocator policy

The Python extension uses mimalloc with delayed automatic purging enabled. It
does not override `MIMALLOC_PURGE_DELAY`, whose mimalloc v3 default is 1000 ms.

GLRMask defaults each purge to a memory **reset** (`MADV_FREE` on supported Unix
systems and `MEM_RESET` on Windows) rather than a synchronous decommit. Reset
pages remain reclaimable by the operating system and reusable by mimalloc, but
process RSS may not decrease immediately. This avoids charging immediate page
decommit work to an otherwise unrelated runtime allocation.

Most runtime work also uses bounded preallocated parser, tokenizer, accumulator,
and mask storage. Ordinary applications therefore need no allocator lifecycle
calls or manual trimming. Set `MIMALLOC_PURGE_DECOMMITS=1` before importing
GLRMask when immediate RSS reduction is more important than allocator tail
latency.

The unstable `glrmask._internal.mimalloc_purge_delay()`,
`glrmask._internal.mimalloc_purge_decommits()`, and
`glrmask._internal.collect_allocator(force=True)` helpers remain available for
diagnostics and controlled experiments.

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
assert state.is_accepting()
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

`Vocab.from_llama_cpp()` excludes EOG, control, unused, and empty-piece tokens from the byte vocabulary. Their IDs are available in `llama_cpp_end_token_ids` for the decoder.

### Compile a constraint

`Constraint` supports JSON Schema, GLRM, Lark, and EBNF:

```python
constraint = glrmask.Constraint.from_json_schema(schema, vocab)
constraint = glrmask.Constraint.from_glrm_grammar(grammar, vocab)
constraint = glrmask.Constraint.from_lark(grammar, vocab)
constraint = glrmask.Constraint.from_ebnf(grammar, vocab)
```

`from_glrm_grammar(...)` accepts compiled child constraints in `subgrammars` and exact token IDs in `bindings`. `DynamicConstraint.from_glrm_grammar(...)` accepts the same arguments.

A compiled child is bound by name:

```python
payload = glrmask.Constraint.from_json_schema(payload_schema, vocab)

constraint = glrmask.Constraint.from_glrm_grammar(
    '''
    glrm 1;
    start document;
    extern grammar payload;
    nt document = "{" payload "}";
    ''',
    vocab,
    subgrammars={"payload": payload},
)
```

One model token may span the parent and child grammars. Every constraint must use the same vocabulary.

Parent and child constraints may use different `ignore` terminals. Equal ignore languages are shared. Different ones stay scoped to their grammar, including within tokens that span the boundary.

### Decode

Create one state per generated sequence:

```python
state = constraint.start()

while generating:
    mask = state.mask(model_vocab_size)
    if state.is_accepting():
        mask[end_token_ids] = True

    token_id = sample_with_mask(logits, mask)
    if token_id in end_tokens:
        break
    state.commit_token(token_id)
```

The main state operations are:

- `mask(size=None)`: return the allowed-token mask.
- `commit_token(token_id)`: advance by one model token.
- `commit_bytes(data)`: advance by raw bytes.
- `forced()`: return a forced token sequence when one can be determined.
- `is_accepting()`: report whether the current prefix may validly end here.
- `is_rejected()`: report whether the current prefix is irrecoverably invalid.

## Cache compiled constraints

`Constraint` objects are immutable and reusable across requests. Serialize them with `save()` and restore them with `load()`:

```python
artifact = constraint.save()
constraint = glrmask.Constraint.load(artifact, vocab)
```

`DynamicConstraint` compiles faster and produces masks more slowly. Use it for one-off constraints or cache misses while compiling a `Constraint` for later requests.

```python
constraint = glrmask.DynamicConstraint.from_json_schema(schema, vocab)
state = constraint.start()
```

`DynamicConstraintState` has the same decoding methods as `ConstraintState`.

## Grammar formats

GLRM is GLRMask's native grammar format. Its source starts with the literal `glrm 1;` header. Declarations use `=`, regexes use full-match semantics, and unsupported or non-regular constructs are rejected:

```glrm
glrm 1;
start value;
t NUMBER = /-?(0|[1-9][0-9]*)/;
nt value = NUMBER | "null";
```

Exact model-token terminals are named in GLRM and bound outside the grammar:

```python
grammar = '''
glrm 1;
start message;
extern token END_TURN;
nt message = "hello" END_TURN;
'''
constraint = glrmask.Constraint.from_glrm_grammar(
    grammar,
    vocab,
    bindings={"END_TURN": end_turn_id},
)
```

Bindings accept one token ID or a list of interchangeable IDs. Lark and EBNF use `@token(<id>)` for exact IDs.

See the [root README](../README.md#grammar-formats) for the fuller format overview.

## Source builds

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install ./python
```

Building from source requires a Rust toolchain and the platform's native linker and build tools. On Windows, activate the environment with `.venv\Scripts\activate`.
