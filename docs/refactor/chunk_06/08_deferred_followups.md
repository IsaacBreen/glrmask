# Deferred follow-ups after Chunk 06

Chunk 06 creates the right boundaries but intentionally does not finish every
possible cleanup.

## F1: Extract materialization common helpers

Current split has `legacy_materialize.rs` importing helper functions from
`vocab_materialize.rs`.  If this feels aesthetically wrong during compile
repair, extract:

```text
used_state_class_ids
intern_state_terminal_label
next_nonzero_stamp
range helpers if needed
```

into `materialize_common.rs`.

Priority: medium.

## F2: Rename `terminal_sequences.rs` to `sparse_can_match.rs`

The file computes sparse CanMatch maps, not arbitrary terminal sequences.  The
current name came from Chunk 02 and is acceptable, but `sparse_can_match.rs` may
be clearer after the scan-relation split.

Priority: medium.

## F3: Move scan-relation env vars into typed options

Current env vars include:

- `GLRMASK_SCAN_RELATION_ORDERED_VOCAB_CACHE`
- `GLRMASK_SCAN_RELATION_ORDERED_VOCAB_CACHE_CAPACITY`
- `GLRMASK_SCAN_RELATION_VOCAB_EQUIV`
- `GLRMASK_SCAN_RELATION_VOCAB_EQUIV_NAIVE`
- `GLRMASK_SCAN_RELATION_SPARSE_ROOT_COLLECT`
- `GLRMASK_SCAN_RELATION_SPARSE_ROOT_MAX_STATES`
- `GLRMASK_SCAN_RELATION_SPARSE_ROOT_MAX_TERMINALS`
- `GLRMASK_VALIDATE_GROUP_SCAN_RELATION_VOCAB`
- `GLRMASK_SCAN_RELATION_USE_LEGACY_VOCAB_SWEEP`

These should eventually become fields under compile options.

Priority: high, but belongs to the configuration/profiling chunk.

## F4: Add actual unit tests for partial-boundary cases

This chunk documents the test obligation.  Once compile/test work resumes, add a
small grammar and vocab where one token ends inside a terminal and another byte
can complete it.

Priority: high.

## F5: Replace runtime comments with `ScanOutcome` vocabulary

Runtime mask/commit code still contains older comments.  Later chunks should
rewrite them to use `Complete`/`Partial` scan terminology consistently.

Priority: medium.

## F6: Verify serialization compatibility

The runtime artifact field `can_match` was not structurally changed in this
chunk.  Still, when tests resume, serialization/load roundtrips should be rerun.

Priority: medium.
