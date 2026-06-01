# Chunk 12 overview: Commit as a decomposed transition relation

This chunk restructures the Commit runtime after Chunk 11 separated runtime state, Mask, and Commit at the top level. The specific problem addressed here was that `src/runtime/commit/mod.rs` still contained nearly all of the operational semantics and most of the performance special cases in one 3220-line file.


## Commit denotation

A Commit step is not a parser operation alone and not a lexer operation alone. It is a relation over live constrained-decoding states. A live state is modeled as a finite map

```text
S : LexerState -> ParserFrontier
```

where each parser frontier is a graph-structured stack whose branches carry delayed longest-match exclusions. Given a byte fragment `b`, Commit enumerates scanner outcomes from every live lexer state, advances the parser frontier by every completed non-ignored terminal that is semantically actionable, carries residual lexer states only when the parser can still accept a terminal that completes that residual scan, and fuses equivalent parser-frontier branches at the end.

The central invariant is:

```text
Commit(S, b) = union over all lexer states q and parser frontiers G in S
               of every pair (q', G') reachable by scanning b from q
               and interpreting completed terminals as parser stack effects.
```

This chunk does not change that denotation. It changes the source layout so each component of the relation is represented by a separate file with a narrow contract.


## Concrete source result

`src/runtime/commit/mod.rs` is now a routing module. The actual work is divided as follows:

```text
src/runtime/commit/
  mod.rs               module boundary and shared imports
  api.rs               public ConstraintState commit methods
  general.rs           reference queue-based transition
  fast_path.rs         unprofiled fast paths
  profiled.rs          profiled transition and per-advance diagnostics
  acceptance.rs        actionable terminal filtering and normalized matches
  initial_scan.rs      first-offset scanner summary
  pruning.rs           delayed longest-match pruning and future exclusions
  queue.rs             frontier enqueue/merge/finalize helpers
  single_top.rs        single-top stack-effect shortcuts
  terminal_advance.rs  cached terminal advance helper
  types.rs             local aliases and small records
```

## Why this is the right next boundary

The paper distinguishes Mask and Commit. Earlier chunks made that distinction visible at the runtime directory level, but Commit itself was still an undifferentiated implementation blob. That blob obscured a deeper mathematical split:

1. scanner relation,
2. match normalization,
3. parser stack-effect interpretation,
4. residual lexer-state preservation,
5. delayed exclusion pruning,
6. state fusion,
7. diagnostics/profiling.

A publication-facing implementation should let a reader check each of those independently.
