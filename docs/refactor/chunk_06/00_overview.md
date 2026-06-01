# Chunk 06 overview: scan relation and CanMatch split

This chunk makes the scan-relation subsystem readable as a mathematical object
rather than as a bag of historical "possible matches" utilities.

The key idea is simple but easy to lose in code: a vocabulary token is a byte
fragment, and the lexer can finish zero or more grammar terminals while scanning
that fragment.  The lexer can also stop inside a terminal.  If it stops inside a
terminal, accepting the token is not justified merely because the completed
prefix was compatible with the parser.  The parser must also be compatible with
at least one terminal that could complete the partial lexer state.

That second condition is what this code stores as `CanMatch`.

Before this chunk, the branch already used paper-aligned names, but the central
file `src/compile/scan_relation/mod.rs` was still about 1850 lines and mixed at
least seven independent responsibilities:

1. ordered vocabulary construction;
2. ordered vocabulary cache policy;
3. CanMatch-specific vocabulary equivalence;
4. grouped interval collection;
5. sweep-line materialization of runtime weights;
6. legacy expanded-sweep validation;
7. root sparse collection; and
8. public compile-pipeline entry points.

After this chunk, `mod.rs` is a small boundary module.  Each responsibility has a
named file.  Runtime commit no longer owns the primitive byte-scan helper; it
calls into `crate::scan::execution`, while compile-time global CanMatch
construction remains in `crate::compile::scan_relation`.

## What changed

Source files added or rewritten:

| File | Purpose |
| --- | --- |
| `src/scan/mod.rs` | Shared scan-domain vocabulary module. |
| `src/scan/relation.rs` | Names `ScanOutcome`, `CompletedTerminals`, `PartialLexerState`, and `CanMatchSet`. |
| `src/scan/execution.rs` | Runtime byte-scan helper used by commit. |
| `src/compile/scan_relation/mod.rs` | Small subsystem facade and mathematical contract. |
| `src/compile/scan_relation/types.rs` | Runtime artifact and construction interface types. |
| `src/compile/scan_relation/ordered_vocab.rs` | Byte-sorted vocab/trie and cache. |
| `src/compile/scan_relation/vocab_equivalence.rs` | CanMatch-specific vocabulary quotient. |
| `src/compile/scan_relation/vocab_materialize.rs` | Sweep-line conversion from interval maps to weights. |
| `src/compile/scan_relation/legacy_materialize.rs` | Legacy expanded sweep isolated as validation oracle. |
| `src/compile/scan_relation/root_collect.rs` | Sparse root collection path. |
| `src/compile/scan_relation/compute.rs` | Compile-pipeline entry points. |
| `src/compile/scan_relation/README.md` | Local reading guide and denotation. |

No compile/test/benchmark run was attempted in this chunk.

## What did not change

The algorithmic intent did not change.  This chunk is structural.  It aims to
make the existing mathematics visible:

- the byte-sorted trie still drives scan-relation construction;
- the grouped interval collector still builds `IntervalCanMatchMap`s;
- the sweep-line materializer still quotients tokens by active `(state,terminal)`
  labels;
- the legacy expanded path still exists only as a fallback/validation oracle;
- the optional CanMatch vocabulary equivalence path remains separate from
  Terminal-DWA equivalence; and
- runtime commit still receives a `TokenizerExecResult` from the byte scan.

## Why this is mathematically necessary

Terminal-DWA equivalence is not CanMatch equivalence.

A Terminal DWA class says: for all completed terminal sequences of interest,
these token/lexer-state pairs behave the same.  CanMatch equivalence says: for
all lexer states, after scanning the token bytes, the set of possible future
terminal completions is the same.  These are different quotient relations.

The old layout made it too easy to pass an ID map from one relation into the
other.  The new layout forces the reader to walk through a named CanMatch
quotient path: `ordered_vocab -> vocab_equivalence -> collector -> vocab_materialize`.
