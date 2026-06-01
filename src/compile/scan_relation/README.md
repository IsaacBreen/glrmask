# Scan relation subsystem

This directory owns the compile-time construction of the paper's scan relation
and the runtime `CanMatch` artifact.

## Denotation

For a lexer state `q` and byte fragment `b`, scanning has two logically separate
outputs:

1. `completed(q, b)`: terminals completed wholly inside `b`; and
2. `end(q, b)`: the lexer state left at the fragment boundary, if the scan did
   not block.

If `end(q, b)` is a boundary state, the completed terminals are enough.  If it
is a non-boundary lexer state `q'`, the token is admissible only when the parser
can accept at least one terminal in `CanMatch(q')`.

The Terminal DWA answers a different question: which token/lexer-state pairs can
produce a completed terminal sequence.  That quotient is not sound for
`CanMatch`, because a token can agree on completed terminals while disagreeing
on the partial state it leaves behind.

## Reading order

1. `mod.rs` — subsystem contract and public entry points.
2. `types.rs` — runtime artifact and construction types.
3. `terminal_sequences.rs` — sparse trie walker used by exact pair partitioning.
4. `collector.rs` — grouped interval collection from tokenizer states.
5. `ordered_vocab.rs` — byte-sorted vocab/trie and cache.
6. `vocab_equivalence.rs` — CanMatch-specific token quotient.
7. `vocab_materialize.rs` — interval maps to runtime weights.
8. `root_collect.rs` — sparse root fast path.
9. `legacy_materialize.rs` — old expanded sweep kept as validation oracle.
10. `compute.rs` — compile-pipeline entry points.
