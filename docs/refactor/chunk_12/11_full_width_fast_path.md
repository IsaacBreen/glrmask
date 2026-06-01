# Full-width fast path

The full-width fast path covers the case where each live tokenizer state has exactly one non-ignored actionable terminal match consuming the whole byte fragment. Then there is no intermediate offset queue to explore.

This optimization is mathematically a degenerate queue with only offsets `0` and `len(bytes)`. Because there are no interior offsets, all matches can be advanced independently and merged into the pending final state.

The correctness proof is a queue proof: show that the general queue would enqueue exactly the same final entries and no interior entries.
