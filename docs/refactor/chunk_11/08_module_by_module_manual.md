# Module-by-module implementation manual

This file describes exactly how to read and maintain the new runtime boundary.

## `runtime/mod.rs`

This is the facade for runtime types.  It should export `Constraint`,
`ConstraintState`, `MaskProfile`, `CommitProfile`, and GLR advance diagnostics.
It should not contain algorithmic code.

## `runtime/state/mod.rs`

Keep only the struct definition and clone/debug semantics here.  If a method
answers a question about the frontier, put it in `inspect.rs`.  If a method is a
derived convenience routine, put it in `force.rs` or another named derived-file.
If a field is a cache or scratch buffer, define its type elsewhere.

## `runtime/state/inspect.rs`

Only read from the frontier.  Do not mutate `generation`, `buffers`, or caches.
Methods here should be safe to call between every generated token for debugging.

## `runtime/state/force.rs`

May call Mask and Commit on cloned states.  Should not mutate `self`.  This file
is allowed to know about token bytes and first-byte forcing because it is a
convenience feature.

## `runtime/mask/mod.rs`

This remains the Mask phase graph.  Keep traversal structure here.  When a type
becomes reusable or purely representational, move it out.  Dense accumulator
logic is already extracted.

## `runtime/mask/dense_acc.rs`

Owns `DenseMaskAcc` and associated dense-token-set operations.  It should not
know about public original-vocabulary bitset layout.  It speaks in internal
token ids and Parser-DWA weights.

## `runtime/mask/bitset.rs`

Owns operations on the caller-visible packed `Vec<u32>` layout.  Do not put
internal-token dense `u64` operations here.

## `runtime/commit/mod.rs`

This remains the Commit phase graph and fast-path implementation.  The next
cleanup should split it by algorithmic path: initial-token fast path,
small-queue path, direct-linear path, profiled path, and reference path.  This
chunk intentionally avoids that risky split.

## `runtime/commit/parser_advance.rs`

The only job is: given a parser stack frontier and terminal id, produce the
successor parser stack frontier.  It may select a template-DFA acceleration, but
it must preserve the GLR-table advance relation.

## `runtime/commit/mask_assert.rs`

Debug-only oracle.  Never use it to define Commit behavior.

## `runtime/commit/options.rs`

All environment reads for Commit-local choices live here.  If a new Commit env
var appears, place it here and document whether it changes semantics or only
validation/performance.
