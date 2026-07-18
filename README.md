# GLRMask

GLRMask is a grammar-constrained generation library for high-throughput LLM decoding. It is optimized for extremely low next-token mask latency across the distribution, even for complex grammars.

## How it works

GLRMask maintains a GLR parser state for the generated prefix, updating it as tokens are committed. To compute the next-token mask, a precomputed deterministic weighted automaton reads each parser stack one symbol at a time.

Each transition carries a Boolean mask over the model vocabulary. These masks are intersected along each stack traversal and unioned across alternative paths.

## Performance

Measured with MaskBench on the JSONSchemaBench corpus, using the Llama 3 vocabulary on an Intel Core i7-13620H under Ubuntu 24.04/WSL2.

> **Preliminary:** these timings are not yet accurate and should not be relied on.

### Mask-generation latency

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-mask-tail-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-mask-tail-2026-07-16.webp">
    <img src="docs/assets/benchmark-mask-tail-2026-07-16.webp" alt="Mask-generation latency tail curves for GLRMask and LLGuidance, with higher exceedance probabilities on the left and rarer events on the right" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-mask-cfa-bars-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-mask-cfa-bars-2026-07-16.webp">
    <img src="docs/assets/benchmark-mask-cfa-bars-2026-07-16.webp" alt="Mask-generation latency percentiles for GLRMask and LLGuidance" width="92%">
  </picture>
</p>

### Compilation time

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-compilation-cfa-bars-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-compilation-cfa-bars-2026-07-16.webp">
    <img src="docs/assets/benchmark-compilation-cfa-bars-2026-07-16.webp" alt="Compilation-time percentiles for GLRMask and LLGuidance" width="92%">
  </picture>
</p>

See the [full benchmark report](docs/benchmark-full-corpus-2026-07-16.md) for methodology.

## Installation

### Python

```bash
python -m pip install glrmask==0.1.0
```

Published wheels include the native extension. Building from source requires a Rust toolchain and the platform's native build tools.

### Rust

```bash
cargo add glrmask@0.1.0
```

## Usage

GLRMask takes a grammar and vocabulary and produces a `Constraint`, which provides a `ConstraintState` via `constraint.start()`.

In the decoding loop, call `state.mask()` to generate the next-token mask. This should be done in parallel with the LLM's forward pass so that the mask is ready before the logits. Apply the mask to the logits before sampling, then call `state.commit_token(token_id)` to advance the state with the sampled token.

```text
state = constraint.start()

while generating:
    in parallel:
        logits = llm.forward(...)
        mask = state.mask()

    logits = apply_mask(logits, mask)
    token_id = sample(logits)
    state.commit_token(token_id)
```

## Python quickstart

```bash
python -m pip install glrmask==0.1.0 llama-cpp-python torch
```

```python
import ctypes
import numpy as np
import llama_cpp
from torch import from_numpy
from torch.distributions import Categorical

import glrmask


llm = llama_cpp.Llama(model_path="model.gguf", logits_all=True)
llama_vocab = llama_cpp.llama_model_get_vocab(llm.model)
tokens = range(llm.n_vocab())
end_token_ids = [
    token for token in tokens
    if llama_cpp.llama_vocab_is_eog(llama_vocab, token)
]
end_tokens = set(end_token_ids)

def token_bytes(token):
    size = -llama_cpp.llama_token_to_piece(llama_vocab, token, None, 0, 0, False)
    buffer = ctypes.create_string_buffer(size)
    length = llama_cpp.llama_token_to_piece(llama_vocab, token, buffer, size, 0, False)
    return buffer.raw[:length]

get_logits = lambda: llm.scores[llm.n_tokens - 1]
sample = lambda logits: Categorical(logits=from_numpy(logits)).sample().item()

prompt = "Classify this review: The story dragged badly. Sentiment: "
input_tokens = llm.tokenize(prompt.encode())

MAX_OUTPUT_TOKENS = 64
```

### Without constraints

```python
llm.reset()
llm.eval(input_tokens)

generated = []

for _ in range(MAX_OUTPUT_TOKENS):
    logits = get_logits()
    token = sample(logits)
    llm.eval([token])
    generated.append(token)

    if token in end_tokens:
        break

print(llm.detokenize(generated).decode())
```

### With GLRMask

```python
vocab = glrmask.Vocab.from_id_to_bytes({
    token: token_bytes(token)
    for token in tokens
    if token not in end_tokens
    and not (
        llama_cpp.llama_vocab_get_attr(llama_vocab, token)
        & (llama_cpp.LLAMA_TOKEN_ATTR_CONTROL | llama_cpp.LLAMA_TOKEN_ATTR_UNUSED)
    )
})

schema = '{"type":"string","enum":["positive","negative","neutral"]}'
constraint = glrmask.Constraint.from_json_schema(
    schema,
    vocab,
    end_token_ids=end_token_ids,
)

llm.reset()
llm.eval(input_tokens)

state = constraint.start()
generated = []

for _ in range(MAX_OUTPUT_TOKENS):
    logits = get_logits()
    mask = state.mask(llm.n_vocab())
    logits[~mask] = -np.inf

    token = sample(logits)
    llm.eval([token])
    state.commit_token(token)
    generated.append(token)

    if token in end_tokens:
        break

print(llm.detokenize(generated).decode())
```

## Compilation and caching

A `Constraint` can be serialized and cached for reuse across requests.

`DynamicConstraint` has the same interface and produces identical masks, but compiles much faster than `Constraint`, at the cost of higher mask-generation latency.

To minimize cold-start latency, `DynamicConstraint` can be used on a cache miss while the corresponding `Constraint` is compiled in parallel and cached for subsequent requests.

```text
grammar + vocabulary
        │
        ▼
  Constraint cache
    ├─ hit  → Constraint ───────────────────→ generate
    └─ miss
         ├─ current request → DynamicConstraint → generate
         └─ parallel build  → compile Constraint → cache
```

<p align="center"><em>Cold-start architecture</em></p>

## Grammar formats

[Unfortunately, there is no universally accepted EBNF dialect.](https://dwheeler.com/essays/dont-use-iso-14977-ebnf.html) In keeping with this tradition, GLRMask includes its own.

GLRM is GLRMask's native, EBNF-like grammar syntax. It supports exact model-token terminals with `@token(<id>)`. GLRMask also accepts Lark and EBNF grammars.

## Special tokens

Use `@token(<id>)` in GLRM, Lark, or EBNF to match an exact model token:

```text
start ::= "hello" @token(128009)
```

Use `end_token_ids` to require one of the specified model tokens after the grammar completes:

```python
constraint = glrmask.Constraint.from_json_schema(
    schema,
    vocab,
    end_token_ids=[128009],
)
```

The state becomes complete only after one of those tokens is committed.

## License

Licensed under either the MIT License or the Apache License, Version 2.0, at your option.
