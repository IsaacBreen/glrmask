# GLRMask

GLRMask is a grammar-constrained generation library for high-throughput LLM decoding. It is optimized for extremely low next-token mask latency across the distribution, even for complex grammars.

## Performance

Measured with MaskBench on the JSONSchemaBench corpus, using the Llama 3 vocabulary on an Intel Core i7-13620H under Ubuntu 24.04/WSL2.

> **Preliminary:** this engineering run spans three GLRMask revisions. The comparisons below use only coverage-matched GLRMask and LLGuidance observations; see the [full benchmark report](docs/benchmark-full-corpus-2026-07-16.md) for methodology and limitations.

GLRMask shifts work into ahead-of-time compilation. In this run, that made compilation substantially slower than LLGuidance, while keeping mask generation in the low-microsecond range across the measured distribution and widening the latency advantage in the tail.

### Mask-generation latency

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-mask-tail-2026-07-16-dark.webp">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-mask-tail-2026-07-16.webp">
    <img src="docs/assets/benchmark-mask-tail-2026-07-16.webp" alt="Mask-generation latency tail curves for GLRMask and LLGuidance, with GLRMask speedup by exceedance probability" width="100%">
  </picture>
</p>

<p align="center"><em>Full paired tail over 2,122,307 shared finite token positions. The lower panel is LLGuidance latency divided by GLRMask latency.</em></p>

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

The figure gives a fuller percentile view; the table preserves the headline values exactly. Lower is better.

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

Compilation values use the 8,956 problems on which both frameworks built successfully. Lower is better.

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

This example also requires llama-cpp-python and PyTorch:

```bash
python -m pip install llama-cpp-python torch
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

`forced()` returns a deterministic continuation without advancing the constraint. Each returned token is committed to both the model and the constraint.

```python
llm.reset()
llm.eval(input_tokens)

state = constraint.start()
generated = []
token = None

while token != llm.token_eos() and len(generated) < MAX_OUTPUT_TOKENS:
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

`Constraint` performs grammar and vocabulary analysis ahead of time and is intended for constraints reused across requests.

`DynamicConstraint` compiles faster but performs more work during each mask query. It is intended for one-off constraints where startup latency matters more than runtime latency.

| Mode | Median compilation | p99 TBM | Maximum TBM |
|---|---:|---:|---:|
| `Constraint` | 50.963 ms | **10.521 µs** | **49.539 µs** |
| `DynamicConstraint` | **4.550 ms** | 23.122 ms | 323.609 ms |

Dynamic mode was measured on a smaller cohort.

```python
static = glrmask.Constraint.from_json_schema(schema, vocab)
dynamic = glrmask.DynamicConstraint.from_json_schema(schema, vocab)
```

## JSON Schema

GLRMask implements a pragmatic subset of JSON Schema. Unsupported constructs may be rejected, and some documented cases broaden or restrict the accepted instance language. See [JSON Schema semantic deviations](docs/json-schema-semantic-deviations.md).

## Rust quickstart

The Rust API returns a packed `u32` bitmask. Bit `token_id % 32` in word `token_id / 32` indicates whether a token is admitted.

```rust
use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], token_id: usize) -> bool {
    let word = token_id / 32;
    word < mask.len() && ((mask[word] >> (token_id % 32)) & 1) != 0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = Vocab::new(
        vec![
            (0, b"hello".to_vec()),
            (1, b" ".to_vec()),
            (2, b"world".to_vec()),
        ],
        None,
    );

    let constraint = Constraint::from_ebnf(
        r#"start ::= "hello" " " "world""#,
        &vocab,
    )?;

    let mut state = constraint.start();

    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0)?;

    assert!(token_allowed(&state.mask(), 1));
    state.commit_token(1)?;

    assert!(token_allowed(&state.mask(), 2));
    state.commit_token(2)?;

    assert!(state.is_finished());

    Ok(())
}
```

Run the repository example with:

```bash
cargo run --example ebnf
```

## Other inputs and serialization

Lark grammars use the corresponding constructor:

```python
constraint = glrmask.Constraint.from_lark(lark_source, vocab)
```

EBNF, Lark, and GLRM grammars can match an exact model token ID with `@token(<id>)`:

```text
start ::= "hello" @token(128009)
```

Compiled constraints can be serialized and restored:

```python
artifact = constraint.save()
restored = glrmask.Constraint.load(artifact, vocab)
state = restored.start()
```

Rust provides `Constraint::save()` and `Constraint::load(...)`.

The [`glrmask-runtime`](glrmask-runtime) crate loads versioned runtime artifacts without including the grammar import and compilation pipeline.

## Limitations

- GLRMask implements a pragmatic subset of JSON Schema with [documented semantic deviations](docs/json-schema-semantic-deviations.md).
- Compilation time and mask latency depend on the grammar, schema, vocabulary, hardware, build configuration, and cache state.
- The full benchmark spans three GLRMask revisions, and dynamic mode was measured on a smaller cohort.
- Direct integrations with serving frameworks are not included in v0.1.

## License

Licensed under either the MIT License or the Apache License, Version 2.0, at your option.
