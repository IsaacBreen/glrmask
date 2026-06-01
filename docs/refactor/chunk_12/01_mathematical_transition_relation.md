# Mathematical transition relation

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


## Objects

Let `Q_L` be the set of tokenizer states, `T` the set of grammar terminals, and `F` the set of parser frontiers represented by GSS values. Runtime state is a finite partial map:

```text
S : Q_L ⇀ F
```

A tokenizer execution from `q` over bytes `b` returns two kinds of information:

```text
Exec(q, b) = (M, r)
```

where `M` is a finite set of terminal matches `(terminal, width)` and `r` is either absent, meaning the bytes cannot be continued as an in-progress token from that offset, or a residual tokenizer state.

A parser frontier advance is:

```text
Advance(G, t) -> G'
```

where `G'` is empty exactly when no active parser stack can accept terminal `t`.

## Commit step

For every queue entry `(offset, q, G)`, Commit executes `Exec(q, b[offset..])`. For every normalized match `(t, w)`, it considers a new offset `offset + w`.

- If `t` is the ignored terminal, the parser frontier is unchanged and only the lexer/tokenizer position changes.
- If `t` is not ignored, Commit applies `Advance(G, t)`.
- If the scanner leaves an end state `r`, Commit preserves `(r, G)` only if the parser can still advance on some terminal that can complete from `r`.

The final map is then fused by tokenizer state.

## Why a queue is necessary

A byte fragment can contain several terminal boundaries. It can also have several possible tokenizations when the tokenizer/language relation branches. A simple loop over matches is not sufficient: choosing one match changes the remaining byte suffix and therefore the scanner state from which future matches are interpreted. The queue represents the dynamic programming frontier over byte offsets.
