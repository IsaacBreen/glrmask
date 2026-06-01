# Invariant catalogue

This file lists invariants reviewers should preserve.

## I1: Terminal-DWA classes are not CanMatch classes

Never feed Terminal-DWA token equivalence into `compute_scan_relation` as if it
were CanMatch equivalence.  The safe path is either no pre-quotient or a quotient
computed by `vocab_equivalence.rs`.

## I2: Ordered token ids are local

`OrderedVocab` ids are local to the scan-relation trie.  They must not escape as
runtime token ids.  Conversion must pass through `ScanRelationVocabMap` and then
through the pipeline's shared ID-space reconciliation.

## I3: Duplicate byte strings remain represented

If multiple original token ids have the same byte string, `ordered_to_originals`
must preserve all of them.  The scan relation may quotient them, but the mapping
must still reconstruct original ids for final mask materialization.

## I4: A range is inclusive

Interval ranges in `IntervalCanMatchMap` are inclusive `(lo, hi)`.  Sweep events
therefore add at `lo` and remove at `hi + 1`, with saturation and vocab-length
clamping.

## I5: Empty terminal sets do not create runtime weights

A `TerminalRangeGroup` with no terminals is meaningless and should be dropped.
A runtime weight with no token set should not be emitted.

## I6: Sparse root collection is compile-time only

`root_collect.rs` may inspect the vocabulary trie and all lexer states.  Runtime
commit may not call it.

## I7: Legacy materialization is not the main path

`legacy_materialize.rs` may be used for validation or emergency fallback.  New
algorithmic improvements belong in `vocab_materialize.rs`.

## I8: Runtime scanning is single-fragment execution

`scan::execution::execute_tokenizer_from_state` scans exactly one chosen byte
fragment from one lexer state.  It must not inspect the whole vocabulary or try
to compute CanMatch.

## I9: CanMatch weights are by terminal

`RuntimeCanMatchByTerminal` must remain keyed by `TerminalID`.  This matches mask
logic, which asks whether some terminal completion accepted by the parser can be
witnessed by a token from the current lexer state.

## I10: The partial-boundary case is mandatory

Any explanation of masking that says only “scan token bytes and accept completed
terminals” is incomplete.  It must also handle the case where token bytes end in
a non-boundary lexer state and require a future completion terminal.

## I11: Cache hits must be verified

Ordered vocab cache entries are keyed by a fingerprint but still verify the
source mapping.  A hash match alone is not enough.

## I12: Profile flags must not leak into algorithm modules unnecessarily

This chunk did not complete the configuration migration, but it localizes most
scan-relation profile flag reads.  Future work should move them into typed
options rather than adding more direct `std::env` reads.
