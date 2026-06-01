# Runtime invariant catalogue

## State invariants

- `ConstraintState::constraint` points to the immutable artifact for the entire
  state lifetime.
- `ConstraintState::state` is the only semantic frontier map.
- `generation` is incremented after every Commit attempt.
- `mask_cache` is valid only when its generation equals `self.generation`.
- cloning preserves `state` and `generation` but not cache/scratch contents.

## Mask invariants

- Mask never mutates the semantic frontier.
- Mask may mutate mask cache and scratch.
- Mask output is in original vocabulary token-id space.
- Dense accumulator token ids are internal token ids.
- Parser DWA weights and scan-relation weights are already reconciled to the
  same internal token space before runtime.
- EOS handling is a final materialization step, not a Parser-DWA transition.

## Commit invariants

- Commit is the only public operation that mutates the frontier map.
- Commit consumes bytes, not abstract terminals; terminals are emitted by the
  tokenizer scan.
- Template-DFA advance must agree with GLR-table advance.
- Optional validation may assert that agreement but must not change the result.
- Commit success/failure should agree with pre-commit Mask membership for token
  commits when the debug oracle is enabled.

## Cache/scratch invariants

- `CommitBuffers::clear_all` leaves no pending state from a previous call.
- Scratch maps must not leak entries between semantically distinct commits.
- Dense mask scratch can be reused only after being overwritten or cleared.
- Cache hit paths must copy a full materialized mask, not a borrowed slice that
  can be invalidated by later mutation.

## Publication invariant

Every named fast path should be justifiable as a refinement of one mathematical
operation.  If a path cannot be described as Mask or Commit, it probably belongs
in diagnostics or should be removed.
