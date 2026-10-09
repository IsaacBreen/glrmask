# Constraint-state lifecycle and end tokens

`Constraint` is closed and immutable/shareable. Call `start()` to create one independent mutable `ConstraintState` per generated sequence. State creation performs initialization only; it never hides deferred grammar compilation.

## Rust

```rust
use glrmask::{BuildOptions, Grammar, Vocab};

let vocab = Vocab::new(vec![
    (0, b"a".to_vec()),
    (1, b"b".to_vec()),
    (2, b"<eos>".to_vec()),
]);
let constraint = Grammar::from_ebnf(r#"start ::= "a" "b""#).compile_with(
    &vocab,
    BuildOptions::default().end_tokens([2]),
)?;
let mut state = constraint.start();

let checkpoint = state.clone();
state.commit_token(0)?;
state.commit_token(1)?;
assert!(state.is_accepting());
assert!(!state.is_rejected());

// End token 2 is now allowed because the grammar body accepts:
state.commit_token(2)?;
assert!(state.is_accepting());
assert!(!state.is_rejected());
assert!(state.mask().iter().all(|word| *word == 0));
assert!(state.commit_token(0).is_err());

state = checkpoint;
assert!(!state.is_accepting());
assert!(!state.is_rejected());
# Ok::<(), glrmask::Error>(())
```

There is no built-in rollback history. For speculative decoding, clone the state and restore the clone if needed. To validate a token sequence without mutating the live state, clone it and commit the candidate tokens to the clone.

## State predicates and stopping policy

- `is_accepting()` means the grammar body is complete at the current prefix. An accepting body may still admit continuation tokens.
- `is_rejected()` means no valid parser/tokenizer state remains.

### Stopping choices

Callers choose how to stop generation depending on whether continuation past acceptance is desired:

1. **Stop at first complete match**: Check `state.is_accepting()` and break as soon as it becomes true.
2. **Permit continuation until end token**: Sample and commit tokens, stopping when the chosen `token_id` belongs to your configured `end_token_ids`. Always commit the chosen token before checking for the break.

Acceptance alone does not mean an end token was consumed. When an allowed end token is committed, the next-token mask becomes empty, the state remains accepting and non-rejected, and further commits are rejected. An empty mask does not prove an end token was consumed: it can also describe an accepting body with no available continuation, or a rejected prefix.

### Caller tracking example

When configured end-token IDs are known, track them explicitly in your generation loop:

```python
end_token_ids = {eos_id}
state = constraint.start()

while True:
    mask = state.mask(vocab_size)
    token_id = sample_with_mask(logits, mask)
    state.commit_token(token_id)
    if token_id in end_token_ids:
        break
```

In Rust:

```rust
let end_token_ids = [2];
let mut state = constraint.start();

while let Some(token_id) = sample(&state.mask()) {
    state.commit_token(token_id)?;
    if end_token_ids.contains(&token_id) {
        break;
    }
}
```

## End-token semantics

End tokens are supplied only when producing the final runnable constraint:

```rust
# use glrmask::{BuildOptions, Grammar, Vocab};
# fn demo(grammar: Grammar<'_>, vocab: &Vocab, eos: u32) -> glrmask::Result<()> {
let constraint = grammar.compile_with(
    vocab,
    BuildOptions::default().end_tokens([eos]),
)?;
# let _ = constraint;
# Ok(())
# }
```

They are generation-root policy rather than embedded grammar terminals. An end token is absent from the mask before body acceptance, becomes allowed once the body accepts, and when committed empties the next-token mask while keeping the state accepting and rejecting further commits. If that `Constraint` is later embedded as a child, its previous root end-token policy is not inherited; only its compiled grammar body is imported.

End-token IDs do not need byte spellings in the ordinary vocabulary. GLRMask sizes the public mask coordinate to include configured end-token IDs.

The Python API uses the same semantics through `grammar.compile(..., end_tokens=[...])` or `module.link(..., end_tokens=[...])`.
