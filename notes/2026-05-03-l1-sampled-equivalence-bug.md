# 2026-05-03 - L1 Sampled Equivalence Bug

## Summary

We hit order-sensitive mask/commit mismatches in the L1 terminal-DWA path. The
bug looked absurdly sensitive: duplicate vocab tokens, token ordering, tiny
regex edits, and changing a token byte could make the MRE appear or disappear.

The root cause was not those surface details. They only changed whether an
unsound equivalence merge became observable.

## What Went Wrong

L1 had a sampled post-processing step after max-length tokenizer-state
equivalence. It checked a small sample of LLM tokens and merged tokenizer
states that looked equivalent on that sample.

That was not a refinement. It was approximate coarsening.

For L1, TSID merging is only sound when every original tokenizer state in the
TSID has the same whole-token terminal result for every LLM token in the
vocabulary. Sampling cannot prove that. An unsampled token can distinguish the
states later, and then the terminal DWA weight says one thing while
`commit_token` follows the concrete tokenizer state and says another.

The o1052 failures exposed both directions:

- false positive: token present in `mask()`, but `commit_token` rejected it;
- false negative: token absent from `mask()`, but `commit_token` accepted it.

The false negative around `b"T"` was especially useful. Direct concrete walking
from the tokenizer state reached an end-state terminal signature containing the
needed terminal, while the built L1 transition weight had been derived from a
sample-merged representative whose signature did not contain that terminal.

## Correct L1 Invariant

For each original tokenizer start state and LLM token bytes:

1. Run the whole token bytes from that start state.
2. Discard terminal matches that do not consume the whole token.
3. Add active terminals matched at full width.
4. Add active possible-future terminals from the concrete full-token end state.

Two tokenizer states may share an L1 TSID only if this result is identical for
every token in the vocab.

Optimizations may change how this is computed, but not the relation being
computed.

## Lessons

- Do not use sampled evidence to merge equivalence classes that affect masks.
- If a sampled pass is used as a proposal, it must be followed by an exact
  finalize pass before the result influences id maps or terminal DWA weights.
- Representative states are dangerous in L1 suffix walks. Whole-token
  equivalence from a start state is not necessarily suffix-closed after the
  first byte. Walk suffixes from concrete DFA states, then map only the final
  result back into compressed spaces.
- Weird MRE sensitivity often means internal ordering changed whether an
  unsound merge became visible. Treat that as a sign to inspect id maps,
  representatives, and cross-partition remapping, not as evidence that the
  surface token or regex detail is semantically special.

## Current Direction

L1 now keeps max-length TSID merging, but the sampled post-merge has been
removed. A second exact post-max-length pass compares compressed whole-vocab
terminal-signature profiles and only merges states whose profiles compare
exactly. This is intended to recover useful L1 TSID coarsening without relying
on sampling.

## Performance Follow-Up

Two tempting shortcuts were tested and rejected:

- A cheap hash-based L1 max-length proposal, followed by exact refinement of
  every non-singleton proposal bucket, was correct in shape but too coarse in
  practice. It forced exact whole-vocab profiles for thousands of states and
  made o1052 substantially slower.
- Simplifying the tokenizer separately for L1 active terminals looked safe, but
  it cost seconds and often reduced the DFA only modestly. On o1052 it made the
  critical L1 partitions slower rather than faster.

`GLRMASK_PARTITION_SERIAL=1` is useful diagnostically because it removes outer
partition contention and shows how fast individual analyses can be when inner
Rayon work has the machine to itself. It is not a compile-time fix by itself:
the serial sum is still worse than parallel wall time for o1052.

The remaining L1 bottleneck is the exact max-length prepass on large original
DFAs, especially long-token partitions. A better fix needs either a faster exact
bounded refinement or a vocab-trie/token-specific exact L1 equivalence analysis
that does not fall back to sampled or hash-defined merging.
