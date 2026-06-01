# Deep dive: runtime/state/scratch.rs

This file documents `25_state_scratch_deep_dive.md`'s target role after Chunk 11.  The focus is
commit scratch-buffer ownership and lifetime.  It is intentionally more detailed than ordinary inline comments
because these notes are meant to guide the later all-at-once implementation
and compile-repair pass.

## What belongs here

- Code whose primary responsibility is commit scratch-buffer ownership and lifetime.
- Local helper functions that cannot be named independently without losing
  clarity.
- Comments that state invariants, not historical accident.
- Small private types when those types exist only to make this file's local
  algorithm readable.

## What does not belong here

- Importer or compile-time construction logic.
- Paper-independent benchmarking hacks unless isolated behind a profile or
  options module.
- Environment-variable reads outside an `options.rs` file.
- Public API decisions that are not specific to this runtime operation.
- Serialization compatibility logic, unless the file is in `artifact/`.

## Mathematical contract

The file must be explainable as a part of the equation:

```text
runtime = immutable compiled constraint + live frontier + Mask query + Commit transition
```

If a future change cannot be placed in one of those four categories, it
should not be added here.  It may belong in diagnostics, compile-time
construction, or a benchmark harness.

## Specific review questions

1. Can a reader tell whether this code mutates the semantic frontier?
2. If it mutates only caches or scratch, is that explicit?
3. If it returns a mask, is the token-id space clearly original rather than
   internal?
4. If it consumes a terminal id, is that terminal id a grammar terminal and
   not a tokenizer state or vocabulary token?
5. If it handles template DFAs, is the reference GLR-table relation still
   visible?
6. Are all panic/assert sites either internal invariant checks or debug
   validation oracles?
7. Would moving this code break the conceptual diagram in
   `18_runtime_transition_diagram.md`?

## Maintenance rule

Prefer adding a new small file over adding another unrelated section to a
monolithic file.  The strongest signal that this file needs another split is
when it begins to answer two different mathematical questions.

## Definition of local done

- The file's first comment names the concept it implements.
- The imports correspond to that concept.
- No unrelated environment reads are present.
- No new public API surface appears accidentally.
- Fast paths document their preconditions and fallback relation.
