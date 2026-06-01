# Chunk 02 mathematical contracts

This document states the contracts that motivated the Chunk 02 rename/move pass.
It is not a proof of the implementation.  It is the mathematical target the
implementation should be organized around.

## 1. Completed scanning and partial scanning are different relations

A vocabulary token is a byte string.  A grammar terminal is a parser-level symbol
emitted by the lexer.  Scanning a byte string from lexer state `q` can do two
things at once:

1. emit zero or more completed grammar terminals;
2. stop either at a terminal boundary or inside a partial terminal match.

Therefore the correct object is not simply “the terminals for a token”.  It is a
branching scan relation:

```text
Scan(q, b) = set of (r, q')
```

where:

- `q` is the lexer state before scanning;
- `b` is the byte fragment being scanned;
- `r` is the completed terminal sequence emitted inside `b`;
- `q'` is the lexer state left at the fragment boundary.

If `q'` is a terminal boundary state, then `r` is already a complete description
of what the parser should see.  If `q'` is non-boundary, then a future byte must
complete the partial terminal.  The parser side must know which terminals could
complete from `q'`.

That second object is CanMatch:

```text
CanMatch(q') = set of terminals that can complete from lexer state q'
```

The old name `possible_matches` did not express this distinction.  It sounded
like a generic cache of terminals.  The new name `CanMatch` is intentionally
verb-like: from a partial lexer state, which terminals can match next?

## 2. Terminal DWA contract

The Terminal DWA reads completed grammar-terminal sequences.  Its weights are
sets, bitsets, or range sets denoting lexer-state/token pairs.

The denotation is:

```text
[[TerminalDWA]](r) = { (q, v) : r is a completed terminal sequence emitted while scanning bytes(v) from q }
```

This says nothing about parser stacks.  It is a lexer/vocabulary object.

Implementation consequence:

- Terminal-DWA code belongs under `compile::terminal_dwa`.
- Token id maps inside this module are internal quotients for this object.
- The module should not be named after the quotient (`id_map_and_terminal_dwa`).
- Partition names should describe their mathematical role, not historical levels.

### Direct partition

The direct partition handles the part of Terminal-DWA construction where terminal
behavior can be represented by direct/single-step terminal paths.  The code may
still use optimized tables and local equivalence maps, but the conceptual role is
direct terminal behavior.

### Pair partition

The pair partition handles the part of Terminal-DWA construction where token byte
strings may interact with pairs or multi-step terminal paths.  The old name `l2p`
encoded implementation history.  The new name says what the partition is about:
pair/multi-step terminal behavior.

## 3. CanMatch/scan-relation contract

The scan-relation compile artifact is not the Terminal DWA.  It uses lexer state,
terminal reachability, and token bytes to answer a different question:

```text
for a partial lexer state q', which internal tokens/terminals remain compatible
with completing a terminal match?
```

Implementation consequence:

- `RuntimeCanMatchByTerminal` is a runtime map from terminal to a weight/mask.
- `ScanRelationVocabMap` is a quotient built for this relation.
- Terminal-DWA equivalence must never be assumed sufficient here.

This is the source of a critical invariant:

```text
TerminalDWA-equivalent tokens need not be CanMatch-equivalent tokens.
```

The implementation already carried a warning to that effect.  Chunk 02 preserves
and foregrounds it.

## 4. Parser DWA contract

The Parser DWA reads parser stack prefixes.  Its weights are again masks over the
same kind of token/lexer-state pairs after reconciliation.

The intended denotation is:

```text
[[ParserDWA]](rho)[q, v] = 1 iff rho belongs to the stack-prefix language that permits token v from lexer state q
```

This makes the Parser DWA a parser-side object whose weights are expressed in the
same token/lexer vocabulary as the Terminal DWA and scan relation.

Implementation consequence:

- Parser-DWA code belongs under `compile::parser_dwa`.
- Runtime mask should describe itself as walking active stacks through Parser DWA.
- Runtime commit should not be described as an LR-specific operation; template
  DFAs recognize stack effects independent of parser presentation.

## 5. Mask contract

Mask is the runtime operation that produces a vocabulary mask for the LLM step.
It observes the current `ConstraintState`, especially active parser stacks and
lexer state, and produces allowed vocabulary tokens.

Conceptually:

```text
Mask(state) = combine weights encountered by walking active stack prefixes through ParserDWA
```

“Combine” here means intersect/accumulate the relevant weights along a DWA walk,
then project through the runtime token-space mapping to materialize a mask in the
original vocabulary ID space.

Implementation consequence:

- Runtime mask code belongs under `runtime::mask`.
- The API should say `fill_mask` and `fill_mask_and_internal_token_ids`.
- It should not say `mask_game`.

## 6. Commit contract

Commit is the runtime operation that accepts bytes or token IDs and updates the
constraint state.  It scans bytes, emits completed terminals, updates the lexer
state, and advances parser stacks through the GLR transition relation or an
equivalent stack-effect recognizer.

Conceptually:

```text
Commit(state, bytes) = advance lexer + parser state according to Scan and parser stack effects
```

Template DFAs belong here as an acceleration.  They are not a different parser
semantics.  They recognize precomputed stack effects used by commit.

Implementation consequence:

- Runtime commit code belongs under `runtime::commit`.
- Template advance code remains under `runtime::commit/template_advance.rs` for
  now, because its runtime role is commit acceleration.
- Compile-time template construction can remain under `compiler::stages::templates`
  until a later chunk gives template DFAs their own paper-level compile home.

## 7. Token quotient contract

The crate has at least three reasons to quotient tokens:

1. Terminal-DWA equivalence.
2. Scan-relation / CanMatch equivalence.
3. Final runtime reconciliation across compiled artifacts.

These quotients should not be conflated.  The names should say which object a
quotient belongs to:

- Terminal-DWA local id maps live inside `compile::terminal_dwa`.
- Scan-relation vocab maps live inside `compile::scan_relation`.
- Runtime internal/original mappings live inside `runtime::*` and public APIs use
  `internal_to_original_token_ids` / `original_to_internal_token_ids`.

The old `mask_game_mapping` name was especially misleading because it named the
consumer/harness rather than the quotient.

## 8. Why this chunk must come before deeper file splitting

Large-file decomposition without terminology cleanup would be unstable.  If a
file is split while still named after historical accidents, the new files inherit
that confusion.  Chunk 02 establishes the nouns first:

- Terminal DWA;
- scan relation;
- CanMatch;
- Parser DWA;
- direct partition;
- pair partition;
- Mask;
- Commit.

After this, later chunks can split large files by subresponsibility while keeping
the mathematical denotation intact.
