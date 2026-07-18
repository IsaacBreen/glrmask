# Constraint-state lifecycle APIs for the follow-up release

> **Status:** release-preparation contract for `<NEXT_VERSION>`. Public `glrmask 0.1.0` does not provide the rollback, proposal-validation, failed-state, or explicit-EOS constructor surfaces described here. Merge this document with the integrated implementation, not by itself.

These APIs support consumers that temporarily advance a constraint state, validate a speculative token sequence, and restore the exact earlier state without retaining an unbounded generated-token history.

## Python

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes(
    {
        0: b"a",
        1: b"b",
        2: b"x",
    },
    eos_token_id=64,
)
constraint = glrmask.Constraint.from_ebnf('start ::= "a" "b"', vocab)
state = constraint.start(max_rollback_tokens=4)

# Returns the longest admissible prefix and does not mutate state.
assert state.validate_tokens([0, 1, 2]) == [0, 1]

state.commit_tokens([0, 1])
assert state.is_complete()
assert state.mask()[64]  # EOS is admissible at completion.

state.rollback(2)
assert not state.is_complete()
assert not state.is_failed()
```

Public Python signatures required by the frozen vLLM backend:

```text
Vocab.from_dict(token_to_id, eos_token_id=None)
Vocab.from_id_to_bytes(id_to_bytes, eos_token_id=None)
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

let vocab = Vocab::new(
    vec![
        (0, b"a".to_vec()),
        (1, b"b".to_vec()),
        (2, b"x".to_vec()),
    ],
    Some(64),
);
let constraint = Constraint::from_ebnf(r#"start ::= "a" "b""#, &vocab)?;
let mut state = constraint.start_with_rollback(4);

assert_eq!(state.validate_tokens(&[0, 1, 2]), vec![0, 1]);
state.commit_tokens(&[0, 1])?;
assert!(state.is_complete());
state.rollback(2)?;
assert!(!state.is_failed());
# Ok::<(), String>(())
```

Required Rust surfaces:

```text
Constraint::start_with_rollback(max_rollback_tokens)
ConstraintState::rollback(num_tokens)
ConstraintState::validate_tokens(tokens)
ConstraintState::is_failed()
Vocab::new(entries, eos_token_id)
Constraint::mask_len()
ConstraintState::fill_mask(buf)
```

## Rollback semantics

- `max_rollback_tokens=0` is the ordinary default. It retains no rollback snapshots and preserves the existing zero-history behavior.
- A positive capacity stores at most that many pre-commit snapshots. Older snapshots are discarded when the capacity is reached.
- `rollback(n)` restores the state immediately before the last `n` recorded token commits and consumes those snapshots.
- `rollback(0)` is a no-op.
- A request beyond the available history fails before mutation. The current mask and state remain unchanged.
- An invalid known-token commit may leave the state failed; if a pre-commit snapshot was recorded, `rollback(1)` restores the prior state.
- A token ID unknown to the constraint fails without consuming an additional rollback slot.
- Rollback clears state-dependent mask caches and scratch data so the restored mask is recomputed from the restored state.

## Proposal validation

`validate_tokens(tokens)` evaluates a copy of the current state with rollback history disabled. It returns the longest prefix that can be committed without an error or failed state.

It does not mutate:

- parser or lexer state;
- completion state;
- rollback history;
- the current mask.

The method returns token IDs rather than a count because that is the public contract consumed by the frozen vLLM backend.

## Failed state

`is_failed()` reports that the internal live-state set is empty. It is distinct from completion:

- complete: the grammar can terminate now;
- failed: no live parser/tokenizer state remains.

Consumers using rollback should check failure after a temporary commit and restore the previous snapshot before continuing.

## Explicit EOS

EOS is supplied separately from the byte-token mapping. It is a lifecycle sentinel, not ordinary bytes:

- do not insert EOS into `id_to_bytes` with an invented or empty byte spelling;
- pass its token ID through `eos_token_id` / `Vocab::new(..., Some(id))`;
- `mask_len()` includes the EOS ID even when it is above every byte-token ID;
- the EOS mask bit is clear before completion and set when the grammar is complete;
- serving adapters using this lifecycle EOS should consume it as a generation-lifecycle event rather than commit it as ordinary parser bytes.

An explicit grammar atom such as `@token(<eos_id>)` is a separate exact-token grammar feature. The lifecycle EOS behavior described here does not replace that grammar-level mechanism.

The packed mask must therefore be sized from `constraint.mask_len()`, not from the largest byte-token ID alone.
