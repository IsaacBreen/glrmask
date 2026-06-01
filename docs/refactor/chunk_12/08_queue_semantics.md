# Queue semantics and state fusion

## Queue entries

A queue entry is `(offset, tokenizer_state, parser_frontier)`. It means: after consuming `offset` bytes of the current byte fragment, lexical scanning would resume in `tokenizer_state` and the parser frontier is `parser_frontier`.

## Pending state

When `offset == bytes.len()`, the entry is not processed further during this Commit call. It is added to pending state. At the end, pending frontiers with the same tokenizer state are merged and then fused.

## Source boundary

`queue.rs` owns:

- `merge_parser_state`
- `queue_parser_state`
- `merge_small_parser_state`
- `finalize_pending_state`

## Fusion invariant

Fusion is an implementation-level compaction. It must preserve the represented set of stacks and annotations. The transition relation is defined before fusion; fusion only chooses a smaller representation of the same frontier.
