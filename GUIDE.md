GLRMask is a grammar-constrained generation library for high-throughput LLM decoding. It is optimized for extremely low next-token mask latency across the distribution, even for complex grammars.

> **Interim benchmark:** the figures below are the corrected official-9,558-schema view of the 20 August 2026 engineering run. They supersede the July figures, but are not the final native publication benchmark.

<p align="center">
  <img src="https://raw.githubusercontent.com/IsaacBreen/glrmask/2f8b1505d0cba2467a458eb8b45c4879710468dd/docs/assets/benchmark-tbm-tail-2026-08-21.webp" alt="TBM latency tail curves for GLRMask and LLGuidance in the corrected 20 August 2026 engineering run" width="100%">
</p>

## Installation

### Python

```bash
python -m pip install glrmask
```

Published wheels include the native extension. Building from source requires a Rust toolchain and the platform's native build tools.

### Rust

```bash
cargo add glrmask
```

**Documentation:** [Python](https://github.com/IsaacBreen/glrmask/blob/main/python/README.md) · [Rust](https://docs.rs/glrmask)

## Usage

GLRMask compiles a grammar and vocabulary into a `Constraint`. The resulting `Constraint` can be serialized and cached for reuse across requests.

At runtime, call `constraint.start()` to initialize a `ConstraintState`. In the decoding loop, run `state.mask()` in parallel with the model’s forward pass so the mask is ready in time for sampling. Then apply the mask to the logits, sample a token, and call `state.commit_token(token_id)` to advance the state.

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

For constraints that will not be reused enough to justify static compilation, `DynamicConstraint` leaves more work in the token loop and avoids the full static build. The corresponding `Constraint` can be compiled separately and cached for later requests.

## Python quickstart

```bash
python -m pip install glrmask llama-cpp-python torch
```

```python
import numpy as np
from llama_cpp import Llama
from torch import from_numpy
from torch.distributions import Categorical

import glrmask


llm = Llama(model_path="model.gguf", logits_all=True)
vocab = glrmask.Vocab.from_llama_cpp(llm)
end_token_ids = vocab.llama_cpp_end_token_ids
end_tokens = set(end_token_ids)

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

## Grammar formats

Unfortunately, [there is no universally accepted EBNF dialect.](https://dwheeler.com/essays/dont-use-iso-14977-ebnf.html) In keeping with this tradition, GLRMask includes its own.

GLRM is GLRMask's native, EBNF-like grammar syntax. It supports exact model-token terminals with `@token(<id>)`. GLRMask also accepts Lark and EBNF grammars.

### Reusing compiled subgrammars

Declare an external grammar with `extern g name;`, then bind an independently
compiled constraint by name. Hidden call terminals and cross-boundary token
paths are handled automatically:

```python
payload = glrmask.Constraint.from_json_schema(payload_schema, vocab)

document = glrmask.Constraint.from_glrm_grammar(
    '''
    start document;
    extern g payload;
    nt document ::= "{" payload "}";
    ''',
    vocab,
    subgrammars={"payload": payload},
)
```

Inline `g name ::= { ... };` and externally bound `extern g name;` have the
same language semantics, including scope-local ignores and model tokens that
cross parent/child boundaries.

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

## Saving compiled constraints

A compiled `Constraint` can be serialized and loaded again:

```python
blob = constraint.save()
constraint = glrmask.Constraint.load(blob, vocab)
```

Load an artifact only with the exact vocabulary it was compiled against. `Constraint.load()` currently does not verify the supplied vocabulary. Composed constraints are saved as one artifact, including their child constraints.

`DynamicConstraint` supports the same source formats but leaves more work for mask generation. It is useful for constraints that are unlikely to be reused enough to justify static compilation.

## How it works

GLRMask maintains a GLR parser state for the generated prefix, updating it as tokens are committed. To compute the next-token mask, a precomputed deterministic weighted automaton reads each parser stack one symbol at a time.

Each transition carries a Boolean mask over the model vocabulary. These masks are intersected along each stack traversal and unioned across alternative paths.

## Performance

Latest corrected engineering result: the **9,558 official JSONSchemaBench schemas**, using their corresponding MaskBench replay payloads. The historical run originally contained 705 additional MaskBench-only cases; those are excluded from every number and graph shown here. The original full sweep used AWS M8azn, and the corrected GLRMask runtime tail was refreshed on the same CPU family after fixing a deterministic post-deserialization first-commit bug. This is intentionally not presented as the final native publication run.

| TBM | GLRMask | LLGuidance |
|---|---:|---:|
| p50 | **3 µs** | 10 µs |
| p99 | **10 µs** | 223 µs |
| p99.9 | **18 µs** | 788 µs |
| p99.99 | **24 µs** | 2,290 µs |
| maximum | **70 µs** | 14,426 µs |

| TTFM | GLRMask | LLGuidance |
|---|---:|---:|
| p50 | 10,421 µs | **1,109 µs** |
| p90 | 59,350 µs | **2,738 µs** |
| p99 | 229,809 µs | **12,023 µs** |
| maximum | 779,588 µs | **81,104 µs** |

<p align="center">
  <img src="https://raw.githubusercontent.com/IsaacBreen/glrmask/fbc60288c7d86701a03a3100fa7acfa3dc8fc5fd/docs/assets/benchmark-tbm-2026-08-21-v3.webp" alt="TBM latency comparison for GLRMask and LLGuidance in the corrected 20 August 2026 engineering run" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/IsaacBreen/glrmask/fbc60288c7d86701a03a3100fa7acfa3dc8fc5fd/docs/assets/benchmark-ttfm-2026-08-21-v3.webp" alt="TTFM comparison for GLRMask and LLGuidance in the corrected 20 August 2026 engineering run" width="100%">
</p>

The old CFA runner used llguidance 1.6.1 and Linux thread-CPU timing; the final native runner uses a different, stricter methodology. See the [20 August engineering benchmark report](https://github.com/IsaacBreen/glrmask/blob/main/docs/benchmark-cfa-full-2026-08-20.md) for the exact corpus, hardware, fix/rerun provenance, build numbers, and interpretation limits.
