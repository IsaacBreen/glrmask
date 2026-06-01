# Compile repair strategy for Chunk 06

This chunk intentionally avoided compilation.  When compilation begins, follow
this order so the architecture is preserved.

## 1. Fix missing imports locally

If a split file cannot see a type, add the narrowest possible import to that
file.  Do not add a giant import to `mod.rs` unless multiple files truly need the
same name.

Examples:

```rust
use super::types::RuntimeCanMatchByTerminal;
use super::ordered_vocab::OrderedVocab;
```

## 2. Fix visibility with `pub(super)`, not `pub(crate)`

Most helpers are shared only inside `compile::scan_relation`.  Use:

```rust
pub(super) fn helper(...)
```

Only pipeline entry points should be `pub(crate)`.

## 3. Extract common helpers instead of making cyclic imports worse

If `legacy_materialize.rs` and `vocab_materialize.rs` complain about sibling
imports, create:

```text
src/compile/scan_relation/materialize_common.rs
```

Move only genuinely shared helpers there.

## 4. Address unused imports without deleting the boundary

Because `#![deny(warnings)]` is active, unused imports will fail.  Remove imports
from individual files.  Avoid silencing broad categories except in the private
prelude during the transition.

## 5. Do not collapse files to fix privacy

If a function in one file needs a helper from another, that means the boundary
needs an explicit edge.  It does not mean the files should merge.

## 6. Run rustfmt after import repair

The line-range split preserves old formatting.  Rustfmt should be run only after
basic compile issues are resolved so it does not obscure semantic repairs.

## 7. Then run tests targeting partial-boundary semantics

The first correctness tests should target:

- token ending at boundary;
- token ending in partial lexer state;
- token blocked by lexer;
- parser rejecting all terminals in `CanMatch(q')`;
- duplicate byte strings; and
- serialization roundtrip.
