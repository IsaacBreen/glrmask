# Proof obligations by module

## `acceptance.rs`

Show that normalized matches are exactly unique `(width, terminal)` pairs that are either ignored or actionable.

## `initial_scan.rs`

Show that the first-offset scan summary is extensionally equivalent to executing the tokenizer separately for each initial live tokenizer state.

## `pruning.rs`

Show that pruning removes exactly branches whose delayed exclusions are contradicted by accepted terminals, and remaps surviving exclusions to residual tokenizer states.

## `queue.rs`

Show that merging and finalization preserve represented parser stack sets modulo GSS compaction.

## `single_top.rs`

Show that each shortcut result equals the reference GLR advance for its precondition shape.

## `fast_path.rs`

For every fast path, prove: if it returns `Some(result)`, then `result` equals the result of `general.rs`; if the preconditions fail, it returns `None`.

## `profiled.rs`

Show that profiled state updates are identical to unprofiled state updates for the same inputs, ignoring observation fields.

## `api.rs`

Show that generation increments exactly once per public commit call and that token lookup errors do not mutate state.
