# Chunk 12 changeset — Commit transition subsystem split

## Summary

This chunk continues from the Chunk 11 patched tree and focuses on the largest remaining runtime knot: `src/runtime/commit/mod.rs`.

The old file mixed all of the following concerns in one place:

- public `ConstraintState::commit_*` API methods,
- scanner match normalization,
- delayed longest-match pruning,
- parser stack-effect advancement,
- queue-based reference Commit,
- unprofiled fast paths,
- profiled transition logic,
- per-advance diagnostic construction,
- GSS merge/fusion helpers,
- single-stack parser shortcuts.

Chunk 12 splits these into named files under `src/runtime/commit/` so the source tree mirrors the mathematical Commit relation.

## Mathematical target

Commit is the runtime transition relation on live constraint states.

A live state is a finite map:

```text
lexer_state -> parser_frontier
```

A byte fragment is scanned from each live lexer state. Completed terminal matches advance the parser frontier. Ignored terminals advance only the scanner. Residual lexer states are retained only when the parser frontier can still accept a terminal that may complete from that residual state. Delayed longest-match exclusions are pruned/remapped at the boundary where they become observable.

## Source changes

### New or newly populated files

- `src/runtime/commit/api.rs`
- `src/runtime/commit/acceptance.rs`
- `src/runtime/commit/fast_path.rs`
- `src/runtime/commit/general.rs`
- `src/runtime/commit/initial_scan.rs`
- `src/runtime/commit/profiled.rs`
- `src/runtime/commit/pruning.rs`
- `src/runtime/commit/queue.rs`
- `src/runtime/commit/single_top.rs`
- `src/runtime/commit/terminal_advance.rs`
- `src/runtime/commit/types.rs`

### Updated files

- `src/runtime/commit/mod.rs` is now a routing layer and module-level mathematical description.
- `src/runtime/commit/README.md` now documents the Commit relation and the new file map.

## Major symbol moves

- Public commit methods moved to `api.rs`.
- `commit_bytes_impl` moved to `general.rs`.
- `commit_bytes_impl_profiled`, `record_per_advance_entry`, and `final_stacks` moved to `profiled.rs`.
- Fast paths moved to `fast_path.rs`.
- `ActionableTerminals` and normalized-match filtering moved to `acceptance.rs`.
- Initial scanner summary methods moved to `initial_scan.rs`.
- Pruning and delayed longest-match helpers moved to `pruning.rs`.
- Queue merge/finalization helpers moved to `queue.rs`.
- Single-top action shortcuts moved to `single_top.rs`.
- Cached terminal advance helper moved to `terminal_advance.rs`.
- Local aliases and small records moved to `types.rs`.

## Documentation added

Added 40 self-contained Markdown files under:

```text
docs/refactor/chunk_12/
```

These cover:

- Commit denotation,
- state-space model,
- scanner match normalization,
- actionable-terminal filtering,
- delayed longest-match pruning,
- parser stack-effect semantics,
- queue semantics,
- fast-path contracts,
- profiling semantics,
- API boundary,
- function move ledger,
- invariant catalogue,
- proof obligations,
- testing strategy,
- compile-repair strategy,
- review checklist,
- deferred follow-up work.

## Static checks

See `glrmask_chunk_12/glrmask_chunk_12_CHECKS.md`.

Highlights:

- expected Commit submodules exist: PASS
- `src/runtime/commit/mod.rs` is 76 lines: PASS
- naive brace balance for Commit Rust files: PASS
- stale `mask_game` terminology absent from Commit sources: PASS
- stale `end_state_may_advance` name absent: PASS
- 40 Chunk 12 docs present: PASS

## Patch stats

```text
files_changed=57
insertions=4753
deletions=3196
```

## Important limitations

Per instruction, I did not compile, run tests, benchmark, or rustfmt this chunk.

The most likely later compile-repair work is mechanical:

- replacing broad `use super::*` imports with explicit imports,
- reducing `pub(super)` visibility where helpers are now file-local,
- repairing any import/visibility fallout from the source split.

## Largest remaining Commit files after this chunk

```text
    76 src/runtime/commit/mod.rs
    90 src/runtime/commit/parser_advance.rs
   114 src/runtime/commit/acceptance.rs
   129 src/runtime/commit/api.rs
   142 src/runtime/commit/profile.rs
   183 src/runtime/commit/pruning.rs
   210 src/runtime/commit/single_top.rs
   320 src/runtime/commit/template_advance.rs
   403 src/runtime/commit/general.rs
   990 src/runtime/commit/fast_path.rs
   998 src/runtime/commit/profiled.rs
  3987 total
```

`fast_path.rs` and `profiled.rs` remain large by design in this chunk. They are now isolated and documented as follow-up targets rather than mixed into the top-level transition blob.
