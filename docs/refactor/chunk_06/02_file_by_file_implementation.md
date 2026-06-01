# File-by-file implementation guide

This is the practical guide for applying, reviewing, or continuing Chunk 06.

## `src/scan/mod.rs`

Purpose: create a neutral scan namespace.

Rules:

1. Do not import parser, GSS, DWA, or compile-pipeline code here.
2. This module can name concepts shared by compile-time and runtime code.
3. Runtime-specific execution helpers are allowed only when they are primitive
   byte scans, not parser commits.
4. Compile-time global relation construction belongs in `compile::scan_relation`,
   not here.

## `src/scan/relation.rs`

Purpose: name the paper-level scan relation.

Definitions added:

- `CompletedTerminals`
- `BoundaryState`
- `PartialLexerState`
- `CanMatchSet`
- `ScanOutcome`

These are intentionally lightweight.  They do not replace every concrete runtime
representation in one pass.  Their purpose is to keep the language of the code
honest.

Future work:

- Replace ad-hoc comments in mask/commit with these names.
- Consider implementing conversion helpers from `TokenizerExecResult` into
  `ScanOutcome` once the exact boundary semantics are unified.

## `src/scan/execution.rs`

Purpose: move primitive runtime byte scanning out of `runtime::commit`.

The old function was physically inside `runtime/commit/tokenizer_scan.rs`, which
made commit look like the owner of lexer scanning.  Commit should own parser
advancement, stack state, and transactional mutation.  It should not own the
basic scan primitive.

The new helper takes:

- a `Tokenizer`;
- preflattened transitions;
- byte slice;
- starting state.

It returns the historical `TokenizerExecResult`, so callers do not need to be
rewritten yet.

## `src/runtime/commit/tokenizer_scan.rs`

Purpose after this chunk: commit-local bookkeeping only.

It still defines `InitialCommitScan`, because that structure is about commit
state and accepted terminal bookkeeping.  But `execute_tokenizer_from_state_small`
now delegates to `crate::scan::execution`.

Review rule: if this file grows new lexer-walking code, that code should be
moved back to `src/scan/execution.rs` unless it depends directly on commit
transaction state.

## `src/compile/scan_relation/mod.rs`

Purpose: subsystem facade.

This file should remain small.  It declares the modules, reexports only the
compile-pipeline entry points and artifact types, and states the mathematical
warning about Terminal-DWA equivalence not implying CanMatch equivalence.

Do not add algorithmic code here.

## `src/compile/scan_relation/types.rs`

Purpose: public/internal type boundary.

Keep only types that define the interface between scan-relation phases or between
this subsystem and the compile pipeline.  Do not put sweep algorithms here.

## `src/compile/scan_relation/ordered_vocab.rs`

Purpose: byte-sorted vocabulary artifacts.

Responsibilities:

- build the byte-sorted ordered vocabulary;
- build the ordered vocabulary prefix tree;
- cache ordered vocab/trie artifacts;
- expose token-byte reconstruction for grouped internal ids; and
- keep dense helper conversions used by weight materialization.

Non-responsibilities:

- deciding CanMatch equivalence;
- collecting CanMatch intervals;
- materializing runtime weights; or
- parser/DWA behavior.

## `src/compile/scan_relation/vocab_equivalence.rs`

Purpose: CanMatch-specific token quotient.

This file is the mathematical firewall against accidentally reusing Terminal-DWA
classes.  It computes token classes by scanning token bytes against the lexer and
observing future terminal completion behavior.

Review rule: if a change here imports Terminal-DWA ID maps or terminal-sequence
weights, stop and re-check soundness.

## `src/compile/scan_relation/vocab_materialize.rs`

Purpose: turn grouped interval maps into runtime weights.

This module owns the sweep-line over ordered token ids.  Its core invariant is
that active groups determine the token signature.  A token's scan-relation
internal id is exactly the id of that active signature.

## `src/compile/scan_relation/legacy_materialize.rs`

Purpose: validation oracle and fallback.

This file is not the main algorithm.  It exists because the old expanded sweep
is easier to compare against.  Isolating it makes the main code simpler and
keeps publication readers away from old incidental structure.

## `src/compile/scan_relation/root_collect.rs`

Purpose: small-root sparse collection.

This path avoids the full grouped interval collector for tiny root signatures.
It should remain guarded by explicit thresholds.  It is compile-time only.

## `src/compile/scan_relation/compute.rs`

Purpose: pipeline entry points.

This is the only module that should be called by `compile::pipeline`.  It wires
together all phases and returns `ScanRelationComputation`.

Keep it orchestration-only.  If a helper has more than a few lines or exposes a
new sub-invariant, move it into one of the specific modules above.
