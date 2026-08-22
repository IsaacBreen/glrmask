# Constraint-state lifecycle and end tokens

`ConstraintState` and `DynamicConstraintState` use the same small interface. A state is cheap to drive and explicitly cloneable when speculative decoding needs a checkpoint.

## Rust

```rust
use glrmask::{CompileOptions, Constraint, Grammar, Vocab};

let vocab = Vocab::new(vec![
    (0, b"a".to_vec()),
    (1, b"b".to_vec()),
    (2, b"x".to_vec()),
]);
let options = CompileOptions::default().end_token_ids(&[64]);
let constraint = Constraint::compile(
    Grammar::ebnf(r#"start ::= "a" "b""#),
    &vocab,
    &options,
)?;
let mut state = constraint.start();

let checkpoint = state.clone();
state.commit_token(0)?;
state.commit_token(1)?;
assert!(!state.is_accepting());
state.commit_token(64)?;
assert!(state.is_accepting());

state = checkpoint;
assert!(!state.is_accepting());
assert!(!state.is_rejected());
# Ok::<(), glrmask::Error>(())
```

There is no built-in rollback history or validation API. For speculative work, clone the state and restore the clone if needed. To validate a token sequence without mutating the live state, clone it and call `commit_token` on the clone one token at a time.

## State predicates

- `is_accepting()` means the current prefix may validly end here. An accepting prefix may still allow additional tokens.
- `is_rejected()` means no valid parser/tokenizer state remains. The prefix is irrecoverably invalid.
- If both are false, the prefix is valid but not currently accepted as a complete output.

## End-token semantics

End tokens belong to the compiled grammar:

- pass IDs with `CompileOptions::end_tokens`;
- before the byte-language portion accepts, the end-token mask bit is clear;
- once it accepts, the end token is admitted;
- committing the admitted end token makes the augmented grammar accepting;
- multiple end-token IDs form a choice;
- a token may simultaneously have ordinary byte semantics and exact end-token semantics.

Do not invent empty bytes for EOS. Size packed buffers from `constraint.mask_len()` or the serving model vocabulary, whichever is larger.

The Python state API follows the same acceptance/rejection terminology and the same core commit/mask operations.
