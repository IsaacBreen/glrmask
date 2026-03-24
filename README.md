# glrmask

`glrmask` compiles grammars into immutable decoding constraints for tokenized LLM
generation. A compiled `Constraint` can produce token masks, accept committed
tokens incrementally, serialize to bytes, and expose explicit diagnostics when
you need to inspect compiler or runtime behavior.

## Supported Inputs

- EBNF
- Lark
- JSON Schema

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

## Diagnostics

Diagnostics are explicit. The normal compilation path is quiet. If you want the
compiler bundle or runtime metrics, call the diagnostics APIs directly.

```rust
use glrmask::{Constraint, Vocab};

let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"b".to_vec())], None);

let (constraint, diagnostics) =
    Constraint::from_ebnf_with_diagnostics(r#"start ::= "a" "b""#, &vocab).unwrap();

let state = constraint.start();
let mask_metrics = state.mask_metrics();
let commit_metrics = state.commit_token_metrics(0).unwrap();

assert!(diagnostics.glr_table.num_states > 0);
assert!(mask_metrics.mask_words > 0);
assert_eq!(commit_metrics.bytes_len, 1);
```

## Serialization

Compiled constraints can be cached and reloaded without recompilation.

```rust
let bytes = constraint.save();
let restored = Constraint::load(&bytes).unwrap();
assert_eq!(constraint.mask_len(), restored.mask_len());
```

## State Helpers

- `state.mask()` returns the packed token bitmask.
- `state.commit_token(token_id)` advances with one token.
- `state.commit_tokens(&token_ids)` advances with a token slice.
- `state.commit_bytes(bytes)` advances with raw bytes.
- `state.force()` returns the currently forced token IDs.
- `state.summary()` returns structural state statistics.

## Examples

```bash
cargo run --example ebnf
cargo run --example json_schema
cargo run --example diagnostics
```

## License

MIT OR Apache-2.0
