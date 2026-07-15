# glrmask

`glrmask` compiles grammars into immutable decoding constraints for tokenized LLM
generation. A compiled `Constraint` can produce token masks, accept committed
tokens incrementally, and serialize to bytes.

## Supported Inputs

- EBNF
- Lark
- JSON Schema (partial support; not full specification conformance)

JSON Schema support is a pragmatic subset, not a claim of full JSON Schema
conformance. Unsupported constructs may be rejected, and some cases deliberately
broaden or restrict the accepted instances for tractability. See
[JSON Schema semantic deviations](docs/json-schema-semantic-deviations.md) before
relying on exact equivalence to the source schema.

## Quick Start

```rust
use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], token_id: usize) -> bool {
    let word = token_id / 32;
    word < mask.len() && ((mask[word] >> (token_id % 32)) & 1) != 0
}

fn main() {
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
    )
    .unwrap();

    let mut state = constraint.start();
    assert!(token_allowed(&state.mask(), 0));

    state.commit_token(0).unwrap();
    assert!(token_allowed(&state.mask(), 1));

    state.commit_token(1).unwrap();
    assert!(token_allowed(&state.mask(), 2));

    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}
```

## Serialization

Compiled constraints can be cached and reloaded without recompilation.

```rust
let bytes = constraint.save();
let restored = Constraint::load(&bytes).unwrap();
assert_eq!(constraint.mask_len(), restored.mask_len());
```

## Exact LLM Token IDs

EBNF, Lark, and GLRM grammars can match an exact LLM token ID with
`@token(<id>)`:

```text
start ::= "hello" @token(128009)
```

A special-token atom is matched only by `commit_token` with that exact token
ID. Its vocabulary bytes, when present, do not match the atom through
`commit_bytes` and cannot partially match it. Token IDs absent from the byte
vocabulary are supported, and an EOS token explicitly used as `@token(...)` is
controlled by the grammar like any other special token.

## State Helpers

- `state.mask()` returns the packed token bitmask.
- `state.commit_token(token_id)` advances with one token.
- `state.commit_tokens(&token_ids)` advances with a token slice.
- `state.commit_bytes(bytes)` advances with raw bytes.
- `state.force()` returns the currently forced token IDs.

## Examples

```bash
cargo run --example ebnf
cargo run --example json_schema
```

## License

MIT OR Apache-2.0
