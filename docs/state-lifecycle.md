# Constraint-state lifecycle and end-token APIs

These APIs support serving engines that validate speculative proposals, temporarily advance a constraint state, and restore an earlier state without retaining unbounded history.

## Python

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({
    0: b"a",
    1: b"b",
    2: b"x",
})
constraint = glrmask.Constraint.from_ebnf(
    'start ::= "a" "b"',
    vocab,
    end_token_ids=[64],
)
state = constraint.start(max_rollback_tokens=4)

# Longest admissible prefix, without mutating `state`.
assert state.validate_tokens([0, 1, 64, 2]) == [0, 1, 64]

state.commit_tokens([0, 1])
assert not state.is_complete()
assert state.mask(size=65)[64]

state.commit_token(64)
assert state.is_complete()

state.rollback(3)
assert not state.is_complete()
assert not state.is_failed()
```

The JSON Schema, EBNF, Lark, and GLRM constructors all accept the optional `end_token_ids` argument.

Public Python lifecycle surfaces:

```text
Constraint.start(max_rollback_tokens=0)
ConstraintState.rollback(num_tokens)
ConstraintState.validate_tokens(token_ids)
ConstraintState.is_failed()
Constraint.mask_len()
ConstraintState.fill_mask(bitmask)
```

## Rust

```rust
use glrmask::{Constraint, Vocab};

let vocab = Vocab::new(vec![
    (0, b"a".to_vec()),
    (1, b"b".to_vec()),
    (2, b"x".to_vec()),
]);
let constraint = Constraint::from_ebnf_with_end_tokens(
    r#"start ::= "a" "b""#,
    &vocab,
    &[64],
)?;
let mut state = constraint.start_with_rollback(4);

assert_eq!(state.validate_tokens(&[0, 1, 64, 2]), vec![0, 1, 64]);
state.commit_tokens(&[0, 1])?;
assert!(!state.is_complete());
state.commit_token(64)?;
assert!(state.is_complete());
state.rollback(3)?;
assert!(!state.is_failed());
# Ok::<(), String>(())
```

Rust constructors with explicit end-token variants:

```text
Constraint::from_json_schema_with_end_tokens(...)
Constraint::from_ebnf_with_end_tokens(...)
Constraint::from_lark_with_end_tokens(...)
Constraint::from_glrm_grammar_with_end_tokens(...)
```

`DynamicConstraint` exposes the corresponding four constructors.

## Rollback semantics

- A capacity of zero is the default and retains no rollback snapshots.
- A positive capacity retains at most that many pre-commit token snapshots.
- `rollback(n)` restores the state before the last `n` recorded token commits and consumes those snapshots.
- `rollback(0)` is a no-op.
- A request beyond available history fails atomically.
- An invalid known-token commit can enter the failed state; `rollback(1)` restores its pre-commit snapshot when history is enabled.
- An unknown token ID fails before mutation and does not consume history.
- Rollback clears state-dependent mask caches and scratch data.

## Proposal validation

`validate_tokens(tokens)` evaluates a history-free copy of the current state and returns the longest prefix that commits without error or failure. It does not mutate parser state, completion, rollback history, or the current mask.

## Failed state

`is_failed()` means the live parser/tokenizer state set is empty. It is distinct from completion:

- complete: the full grammar, including any required end token, has been consumed;
- failed: no valid continuation remains.

## End-token semantics

End tokens belong to the grammar, not to `Vocab` metadata:

- pass IDs through `end_token_ids` when compiling a constraint;
- each ID is appended as an exact special-token terminal after the original start language;
- before the byte-language portion completes, the end-token mask bit is clear;
- after it completes, the end-token bit is admitted;
- committing an admitted end token completes the augmented grammar;
- multiple end-token IDs form a choice;
- a token may simultaneously have ordinary byte semantics and exact end-token semantics.

Do not invent empty bytes for EOS. Size packed buffers from `constraint.mask_len()` or the serving model vocabulary, whichever is larger.
