# Exact edit log

This log describes the mechanical edits in enough detail for a basic implementer
to reproduce them.

## 1. Split `src/compile/scan_relation/mod.rs`

Original state: one file of roughly 1850 lines.

New state:

```text
src/compile/scan_relation/mod.rs
src/compile/scan_relation/types.rs
src/compile/scan_relation/ordered_vocab.rs
src/compile/scan_relation/vocab_equivalence.rs
src/compile/scan_relation/vocab_materialize.rs
src/compile/scan_relation/legacy_materialize.rs
src/compile/scan_relation/root_collect.rs
src/compile/scan_relation/compute.rs
```

Line-range moves from the old file:

| Old range | New file |
| --- | --- |
| interface type block | `types.rs` |
| ordered vocab/cache functions | `ordered_vocab.rs` |
| CanMatch token equivalence functions | `vocab_equivalence.rs` |
| grouped sweep functions | `vocab_materialize.rs` |
| expanded legacy sweep functions | `legacy_materialize.rs` |
| sparse root collection helpers | `root_collect.rs` |
| public compute functions | `compute.rs` |

## 2. Rewrite `mod.rs`

The new `mod.rs` contains:

- module-level mathematical docs;
- a private prelude for shared imports;
- module declarations; and
- reexports for only the compile-pipeline API.

Do not add construction logic back into `mod.rs`.

## 3. Add `src/scan/`

Created:

```text
src/scan/mod.rs
src/scan/relation.rs
src/scan/execution.rs
```

`relation.rs` names paper-level scan concepts.  `execution.rs` contains the
runtime primitive formerly implemented directly inside commit.

## 4. Update `src/lib.rs`

Added:

```rust
pub(crate) mod scan;
```

The module remains crate-private.  It is shared internally; it is not a public
API promise yet.

## 5. Rewrite runtime commit scanner wrapper

`src/runtime/commit/tokenizer_scan.rs` now keeps `InitialCommitScan` but delegates
byte scanning to `crate::scan::execution::execute_tokenizer_from_state`.

## 6. Add local README

`src/compile/scan_relation/README.md` describes denotation and reading order.

## 7. Add this documentation set

All Chunk 06 docs live under:

```text
docs/refactor/chunk_06/
```
