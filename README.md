# GLRMask

GLRMask is a grammar-constrained generation library designed for high-throughput decoding. It moves grammar and vocabulary analysis ahead of time wherever possible, minimizing work during generation and keeping next-token mask computation fast and predictable, especially at the tail and for complex grammars.

## Performance

Measured with MaskBench on the JSONSchemaBench corpus, using the Llama 3 vocabulary on an Intel Core i7-13620H under Ubuntu 24.04/WSL2. Engines ran single-threaded; each token timing is the minimum of 20 traversals.

### Mask generation latency

| Latency | GLRMask | LLGuidance |
|---|---:|---:|
| Median | **1.440 µs** | 12.197 µs |
| p99 | **5.172 µs** | 246.788 µs |
| p99.9 | **7.675 µs** | 950.309 µs |
| Maximum | **28.565 µs** | 8,041.301 µs |

![Mask generation latency for GLRMask and LLGuidance on a logarithmic scale](docs/assets/mask-generation-latency-2026-07-16.svg)

### Compilation time

| Latency | GLRMask | LLGuidance |
|---|---:|---:|
| Median | 50.963 ms | **0.905 ms** |
| p99 | 565.006 ms | **11.810 ms** |
| p99.9 | 2,217.617 ms | **42.986 ms** |
| Maximum | 6,440.287 ms | **239.964 ms** |

See the [full benchmark report](docs/benchmark-full-corpus-2026-07-16.md) for the complete setup and results.

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

This example also requires PyTorch and Transformers:

```bash
python -m pip install torch transformers
```

```python
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import glrmask


MODEL_ID = "openai-community/gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

vocab = glrmask.Vocab.from_id_to_bytes({
    token_id: bytes(tokenizer.byte_decoder[c] for c in token)
    for token, token_id in tokenizer.get_vocab().items()
})

schema = r'''
{
  "type": "object",
  "properties": {
    "sentiment": {"enum": ["positive", "negative", "neutral"]}
  },
  "required": ["sentiment"],
  "additionalProperties": false
}
'''

constraint = glrmask.Constraint.from_json_schema(schema, vocab)
state = constraint.start()

prompt = """Classify the sentiment of this review:

The performances were excellent, but the story dragged badly.

Return only a JSON object with a sentiment field."""

model_input = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
past_key_values = None
generated = []


@torch.inference_mode()
def model_step(input_ids, cache):
    return model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
    )


with ThreadPoolExecutor(max_workers=1) as executor:
    for _ in range(64):
        model_future = executor.submit(
            model_step,
            model_input,
            past_key_values,
        )
        allowed = torch.from_numpy(state.mask()).to(DEVICE)

        output = model_future.result()
        logits = output.logits[0, -1].float()
        logits.masked_fill_(~allowed, -torch.inf)

        if torch.isneginf(logits).all():
            raise RuntimeError("the constraint rejected every token")

        token_id = torch.multinomial(
            torch.softmax(logits / 0.8, dim=-1),
            num_samples=1,
        )
        token = int(token_id)

        generated.append(token)
        state.commit_token(token)

        model_input = token_id.view(1, 1)
        past_key_values = output.past_key_values

        if token == tokenizer.eos_token_id:
            break
    else:
        raise RuntimeError("generation did not finish within 64 tokens")

print(tokenizer.decode(generated, skip_special_tokens=True))
```

The model forward pass and mask generation run concurrently. The sampled token advances the constraint and becomes the model's next input.

The complete example is in [`examples/python_quickstart.py`](examples/python_quickstart.py).

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
