# Scan relation and CanMatch

The scan relation is the boundary between byte-level tokenization and
terminal-level parser reasoning.  It should be kept distinct from both the
Terminal DWA and the Parser DWA.

## Definitions

Let `q` be a lexer/tokenizer state and `b` a byte fragment.  Scanning `b` from
`q` may complete zero or more grammar terminals and may end either at a terminal
boundary or inside a terminal match.

`Scan(q, b)` records:

1. the sequence of completed grammar terminals emitted inside `b`; and
2. the lexer state left at the byte-fragment boundary.

If scanning ends at a boundary state, the completed terminal sequence is already
closed.  If scanning ends at a non-boundary state `q'`, then the runtime still
needs to know which terminals can complete that partial match.  That second
relation is `CanMatch(q')`.

## Why this is not the Terminal DWA

The Terminal DWA is indexed by completed terminal sequences.  It cannot by
itself answer partial-byte questions, because a token may end in the middle of a
terminal.  The scan relation exists because runtime commit and runtime mask need
to treat “completed terminals inside this fragment” and “a partial terminal that
can complete later” differently.

## Code ownership

```text
src/compile/scan_relation/
  mod.rs                 build the runtime CanMatch weights and token quotient
  collector.rs           dense interval collector over the vocab trie
  terminal_sequences.rs  sparse CanMatchComputer over tokenizer states and vocab-prefix nodes
  profile.rs             scan-relation profiling helpers
```

Runtime byte scanning remains in:

```text
src/runtime/commit/tokenizer_scan.rs
```

This split is intentional.  Compile-time scan-relation construction asks what
could happen for every relevant lexer state and token prefix.  Runtime commit
asks what did happen for the bytes the model actually chose.

## Critical invariant

Terminal-DWA equivalence must not be reused as Scan/CanMatch equivalence.

Two tokens may be equivalent when viewed only as completed terminal sequences
and still differ when a partial lexer state is involved.  Therefore scan-relation
vocabulary equivalence has its own builder and validation logic.

## Naming decisions in chunk 02

| Old name | New name | Reason |
| --- | --- | --- |
| `constraint_possible_matches` | `scan_relation` | Names the relation instead of a vague cache. |
| `PossibleMatchesComputer` | `CanMatchComputer` | Computes terminals that can complete from a state/prefix. |
| `PossibleMatchesProfile` | `CanMatchProfile` | Profile belongs to CanMatch computation. |
| `RuntimePossibleMatchesByTerminal` | `RuntimeCanMatchByTerminal` | Runtime artifact stores CanMatch weights by terminal. |
| `pmv` | `scan_relation_vocab` | The vocabulary quotient belongs to the scan relation. |

## Future split

`scan_relation/mod.rs` is still too large.  A later chunk should split it into:

```text
scan_relation/
  mod.rs
  ordered_vocab.rs
  vocab_equivalence.rs
  interval_sweep.rs
  materialize.rs
  validate.rs
  collector.rs
  terminal_sequences.rs
  profile.rs
```

Chunk 02 only establishes the correct terminology and module home before deeper
surgery.

## Chunk 06 structural update


Chunk 06 split the scan-relation implementation into explicit mathematical
responsibilities.  The old monolithic implementation mixed ordered vocabulary
construction, CanMatch-specific vocabulary quotienting, grouped interval
collection, sweep-line materialization, legacy validation, sparse root
collection, and public compute entry points.  These now live in separate files
under `src/compile/scan_relation/`.

It also introduced `src/scan/` for shared scan-domain vocabulary.  Runtime commit
now delegates primitive byte scanning to `scan::execution`, while global
CanMatch construction remains compile-time only.
