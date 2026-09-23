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
assert!(!state.is_terminated());

state.commit_token(2)?;
assert!(state.is_terminated());
assert!(state.mask().iter().all(|word| *word == 0));

state = checkpoint;
assert!(!state.is_accepting());
assert!(!state.is_rejected());
# Ok::<(), glrmask::Error>(())
```

There is no built-in rollback history. For speculative decoding, clone the state and restore the clone if needed. To validate a token sequence without mutating the live state, clone it and commit the candidate tokens to the clone.

## State predicates

- `is_accepting()` means the grammar body is complete at the current prefix. An accepting body may still allow non-end continuation tokens.
- `is_rejected()` means no valid parser/tokenizer state remains.
- `is_terminated()` means one of the final constraint's configured end-token IDs was committed while the grammar body was accepting. A terminated state has an empty next-token mask and rejects further commits.

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

They are generation-root policy rather than embedded grammar terminals. An end token is absent from the mask before body acceptance, becomes allowed once the body accepts, and terminates the state when committed. If that `Constraint` is later embedded as a child, its previous root end-token policy is not inherited; only its compiled grammar body is imported.

End-token IDs do not need byte spellings in the ordinary vocabulary. GLRMask sizes the public mask coordinate to include configured end-token IDs.

The Python API uses the same semantics through `grammar.compile(..., end_tokens=[...])` or `module.link(..., end_tokens=[...])` and exposes `state.is_terminated()`.
