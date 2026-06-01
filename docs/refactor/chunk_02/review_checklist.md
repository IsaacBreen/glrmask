# Chunk 02 review checklist

Use this checklist to review the paper-terminology alignment pass.  It is
organized from most important to least important.  A failed high-priority check
means the chunk should be revised before moving on to deeper refactors.

## A. High-priority checks

### A1. Paper objects have top-level compile homes

Expected tree:

```text
src/compile/
  mod.rs
  terminal_dwa/
  scan_relation/
  parser_dwa/
```

Pass criteria:

- `src/lib.rs` declares `pub(crate) mod compile;`.
- `src/compile/mod.rs` documents the denotational boundary.
- Terminal-DWA code is under `src/compile/terminal_dwa/`.
- CanMatch/scan-relation code is under `src/compile/scan_relation/`.
- Parser-DWA builder code is under `src/compile/parser_dwa/`.

### A2. Historical path names are gone from source paths

Run:

```bash
find src -path '*id_map_and_terminal_dwa*' -o -path '*constraint_possible_matches*' -o -path '*l2p*' -o -path '*l1*'
```

Expected output: none.

### A3. Historical source identifiers are gone

Run:

```bash
rg -n 'id_map_and_terminal_dwa|constraint_possible_matches|mask_game|possible_matches|PossibleMatches|possible_match|pmv|PMV|L2P|l2p' src bindings README.md Cargo.toml
```

Expected output: none.

Docs may mention old names in migration tables; source should not.

### A4. Runtime API uses internal-token language

Search in `src/runtime`, `src/api`, and `bindings/python`.

Expected public-ish names:

```text
internal_to_original_token_ids
original_to_internal_token_ids
fill_mask_and_internal_token_ids
```

Forbidden names:

```text
mask_game_mapping
mask_game_token_ids
fill_mask_and_mask_game_token_ids
```

### A5. Scan relation retains the equivalence warning

Inspect `src/compile/scan_relation/mod.rs`.

It must still include the invariant that Terminal-DWA equivalence maps must not
be reused for CanMatch equivalence.  This warning is a correctness property, not
just a comment.

## B. Medium-priority checks

### B1. Pipeline imports tell the mathematical story

Inspect `src/compiler/pipeline.rs`.

The compile artifact imports should come from:

```rust
crate::compile::terminal_dwa
crate::compile::scan_relation
crate::compile::parser_dwa
```

The pipeline should not import Terminal-DWA or Parser-DWA builders from
`compiler::stages` anymore.

### B2. Environment variables match the new concepts

Search for:

```bash
rg -n 'GLRMASK_L2P|GLRMASK_PM|GLRMASK_DWA_PM_MODE|GLRMASK_PARSER_DWA_PM_COMPACTION|GLRMASK_COMPACT_POSSIBLE_MATCHES' src bindings README.md Cargo.toml docs
```

Expected source output: none.  Docs may mention the old variables in migration
tables.

Expected new families:

```text
GLRMASK_PAIR_PARTITION_*
GLRMASK_SCAN_RELATION_*
GLRMASK_DWA_CAN_MATCH_MODE
GLRMASK_PARSER_DWA_CAN_MATCH_COMPACTION
GLRMASK_COMPACT_CAN_MATCH_BEFORE_RECONCILE
```

### B3. Direct and pair partition docs are comprehensible

Inspect:

```text
src/compile/terminal_dwa/direct_partition/mod.rs
src/compile/terminal_dwa/pair_partition/mod.rs
```

Pass criteria:

- `direct_partition` is described as the direct/single-step path.
- `pair_partition` is described as the pair/multi-step path.
- No comments refer to historical commit hashes or historical pipeline-shape notes.

### B4. Parser DWA docs do not over-emphasize LR mechanics

Inspect:

```text
src/compile/parser_dwa/mod.rs
src/compile/parser_dwa/builder.rs
src/runtime/mask/mod.rs
src/runtime/commit/mod.rs
```

Pass criteria:

- Parser DWA is described as an automaton over parser stack prefixes.
- Mask is described as walking active stacks through Parser DWA.
- Commit/template DFA comments emphasize stack effects rather than LR-specific
  details.

## C. Low-priority checks

### C1. Documentation is self-contained

The following documents should exist:

```text
docs/terminology.md
docs/scan_relation.md
docs/terminal_dwa.md
docs/parser_dwa.md
docs/chunk_02_terminology_alignment.md
docs/refactor/chunk_02/implementation_manual.md
docs/refactor/chunk_02/mathematical_contracts.md
docs/refactor/chunk_02/review_checklist.md
```

Together they should explain both what changed and why.

### C2. Public API boundary docs mention the alias removal

Inspect `docs/api_boundary.md`.

It should say that benchmark-era `mask_game_*` aliases are removed and replaced
by internal-token quotient terminology.

### C3. README does not use old names

Run:

```bash
rg -n 'mask_game|possible_matches|id_map_and_terminal_dwa|l2p|L2P|pmv|PMV' README.md
```

Expected output: none.

## D. Explicit non-checks

Do not fail this chunk because of any of the following:

- The crate has not been compiled.
- Rustfmt has not been run.
- Large files are still large.
- GLR internals still live under `src/compiler/glr`.
- Template compile code still lives under `src/compiler/stages/templates`.
- Environment-variable sprawl still exists.

Those are real issues, but they belong to later chunks.

## E. Reviewer questions

A reviewer should answer these questions after reading the chunk:

1. Can I explain the difference between Terminal DWA and Parser DWA from the file
   tree alone?
2. Can I explain why CanMatch is not the same thing as completed terminal
   scanning?
3. Can I find the runtime Mask implementation without knowing historical names?
4. Can I find the runtime Commit implementation without knowing historical names?
5. Can I tell which token quotient belongs to which compiled object?
6. Are old names confined to documentation migration tables?
7. Is there any public-ish API name that still sounds like a benchmark harness?
8. Does the source tree now give us a stable basis for splitting large modules?

The desired answer to every question is yes.
