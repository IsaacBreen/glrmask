# GLRMask

GLRMask is a grammar-constrained generation library for high-throughput LLM decoding. It is optimized for extremely low next-token mask latency across the distribution, even for complex grammars.

## Performance

Measured with MaskBench on the JSONSchemaBench corpus, using the Llama 3 vocabulary on an Intel Core i7-13620H under Ubuntu 24.04/WSL2.

> **Preliminary:** these timings are not yet accurate and should not be relied on.

### Mask-generation latency

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-mask-tail-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-mask-tail-2026-07-16.webp">
    <img src="docs/assets/benchmark-mask-tail-2026-07-16.webp" alt="Mask-generation latency tail curves for GLRMask and LLGuidance, with GLRMask speedup by exceedance probability" width="100%">
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-mask-summary-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-mask-summary-2026-07-16.webp">
    <img src="docs/assets/benchmark-mask-summary-2026-07-16.webp" alt="Mask-generation latency percentile summary for GLRMask and LLGuidance" width="88%">
  </picture>
</p>

| Latency | GLRMask | LLGuidance |
|---|---:|---:|
| Mean | **1.743 µs** | 24.179 µs |
| Median | **1.440 µs** | 12.205 µs |
| p99 | **5.171 µs** | 247.306 µs |
| p99.9 | **7.673 µs** | 950.700 µs |
| p99.99 | **11.556 µs** | 2,771.304 µs |
| Maximum | **28.565 µs** | 8,041.301 µs |

### Compilation time

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-compilation-summary-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-compilation-summary-2026-07-16.webp">
    <img src="docs/assets/benchmark-compilation-summary-2026-07-16.webp" alt="Compilation-time percentile summary for GLRMask and LLGuidance" width="88%">
  </picture>
</p>

| Compilation time | GLRMask | LLGuidance |
|---|---:|---:|
| Mean | 86.698 ms | **1.527 ms** |
| Median | 50.963 ms | **0.904 ms** |
| p99 | 565.257 ms | **11.711 ms** |
| p99.9 | 2,221.152 ms | **42.989 ms** |
| Maximum | 6,440.287 ms | **239.964 ms** |

See the [full benchmark report](docs/benchmark-full-corpus-2026-07-16.md) for methodology.

## Installation

### Python

```bash
python -m pip install glrmask==0.1.0
```

Published wheels contain the native extension. Building from source requires Python, a Rust toolchain, and the platform's native linker and build tools.

### Rust

```bash
cargo add glrmask@0.1.0
```

or add the dependency directly:

```toml
[dependencies]
glrmask = "0.1.0"
```

## Python quickstart

```bash
python -m pip install glrmask==0.1.0 llama-cpp-python torch
```

```python
import numpy as np
from llama_cpp import Llama
from torch import from_numpy
from torch.distributions import Categorical

import glrmask


llm = Llama(model_path="model.gguf", logits_all=True)

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

    if token == llm.token_eos():
        break

print(llm.detokenize(generated).decode())
```

### With GLRMask

```python
vocab = glrmask.Vocab.from_llama_cpp(llm)

schema = '{"type":"string","enum":["positive","negative","neutral"]}'
constraint = glrmask.Constraint.from_json_schema(schema, vocab)

llm.reset()
llm.eval(input_tokens)

state = constraint.start()
generated = []

for _ in range(MAX_OUTPUT_TOKENS):
    logits = get_logits()
    mask = state.mask()
    logits[~mask] = -np.inf

    token = sample(logits)
    llm.eval([token])
    state.commit_token(token)
    generated.append(token)

    if token == llm.token_eos():
        break

print(llm.detokenize(generated).decode())
```

### With forced tokens

`forced()` returns the token IDs that can be emitted without sampling. It does not advance the state.

```python
llm.reset()
llm.eval(input_tokens)

state = constraint.start()
generated = []

for _ in range(MAX_OUTPUT_TOKENS):
    logits = get_logits()
    mask = state.mask()
    logits[~mask] = -np.inf

    token = sample(logits)
    llm.eval([token])
    state.commit_token(token)
    generated.append(token)

    if token == llm.token_eos():
        break

    for token in state.forced():
        llm.eval([token])
        state.commit_token(token)
        generated.append(token)

        if token == llm.token_eos():
            break

print(llm.detokenize(generated).decode())
```

## Compilation modes

`Constraint` and `DynamicConstraint` have the same interface and produce identical masks. `Constraint` is optimized for per-token latency; `DynamicConstraint` is optimized for cold-start latency.

On a cache miss, use `DynamicConstraint` immediately while a builder compiles and caches `Constraint`. To hot-swap an active request, start a state from the compiled `Constraint` and replay the generated token IDs.

| Mode | Median compilation | p99 TBM | Maximum TBM |
|---|---:|---:|---:|
| `Constraint` | 50.963 ms | **10.521 µs** | **49.539 µs** |
| `DynamicConstraint` | **4.550 ms** | 23.122 ms | 323.609 ms |

## JSON Schema

GLRMask implements a pragmatic subset of JSON Schema. Unsupported constructs may be rejected. See [JSON Schema semantic deviations](docs/json-schema-semantic-deviations.md).

## Other grammar formats

Lark grammars use the corresponding constructor:

```python
constraint = glrmask.Constraint.from_lark(lark_source, vocab)
```

## Special tokens

Use `@token(<id>)` in EBNF, Lark, or GLRM grammars to match a model token by ID:

```text
start ::= "hello" @token(128009)
```

GLRMask normally masks the vocabulary's EOS token until the constraint is complete. If EOS is referenced explicitly with `@token(...)`, the grammar controls when it is allowed.

## Serialization

Compiled constraints can be serialized and restored:

```python
artifact = constraint.save()
restored = glrmask.Constraint.load(artifact, vocab)
state = restored.start()
```

Rust provides `Constraint::save()` and `Constraint::load(...)`.

## License

Licensed under either the MIT License or the Apache License, Version 2.0, at your option.
