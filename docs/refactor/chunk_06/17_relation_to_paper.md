# Relation to the paper exposition

The paper's explanation should now be able to mirror the code more directly.

## Paper object: Scan relation

Code locations:

```text
src/scan/relation.rs
src/compile/scan_relation/mod.rs
src/compile/scan_relation/collector.rs
```

Suggested paper phrasing:

> Scanning a token byte string from lexer state `q` returns the terminals
> completed within the token and the lexer state left at the token boundary.

The code now has names for both parts: `CompletedTerminals` and boundary/partial
lexer state.

## Paper object: CanMatch

Code locations:

```text
src/compile/scan_relation/types.rs
src/compile/scan_relation/terminal_sequences.rs
src/compile/scan_relation/vocab_equivalence.rs
src/compile/scan_relation/vocab_materialize.rs
```

Suggested paper phrasing:

> If the token boundary falls inside a terminal match, we use `CanMatch(q')` to
> restrict tokens to those whose partial lexer state can still complete a parser-
> admissible terminal.

The code now stores runtime CanMatch as `RuntimeCanMatchByTerminal`.

## Paper object: Terminal DWA

Code locations:

```text
src/compile/terminal_dwa/**
```

Connection:

The Terminal DWA concerns completed terminal sequences.  It is upstream of the
Parser DWA and adjacent to the scan relation, but its equivalence classes are not
CanMatch equivalence classes.

## Paper object: Parser DWA

Code locations:

```text
src/compile/parser_dwa/**
```

Connection:

The Parser DWA returns token masks for parser stack prefixes.  CanMatch weights
must be reconciled into the same final token space so runtime mask generation can
combine parser admissibility and lexer continuation constraints.

## Suggested paper/code naming convention

Use:

```text
Scan
CanMatch
Terminal DWA
Parser DWA
Mask
Commit
```

Avoid:

```text
possible matches
PM/PMV
L1/L2P
mask game
```

Those old names obscure which mathematical relation is being discussed.
