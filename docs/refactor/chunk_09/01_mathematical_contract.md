# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Denotational contract

The parser subsystem supplies a relation, not merely code.  The central relation is:

```text
Advance(T, S, t) = S'
```

where:

- `T` is a GLR table over terminals, nonterminals, states, actions, and gotos;
- `S` is a persistent parser graph-structured stack (GSS);
- `t` is a completed grammar terminal; and
- `S'` is the set of parser stack paths reachable after consuming `t`.

This relation is used in two different ways.

### Compile-time use

The compiler asks what stack effects terminals induce.  Template DFA construction and Parser-DWA construction need a representation of terminal stack effects that is independent of the concrete generated prefix.  GLR table construction and analysis are compile-time producers of those effects.

### Runtime use

Commit receives concrete bytes, converts them to completed terminal sequences, and must update the active parser stacks.  Runtime therefore executes the same parser-stack relation over a concrete GSS.

The old `compiler::glr` namespace hid this dual role.  `parser::glr` expresses it.

## Analysis contract

`parser::glr::analysis` maps a flat grammar to an `AnalyzedGrammar`:

```text
GrammarDef -> AnalyzedGrammar
```

The output records:

- augmented start production;
- normalized productions;
- terminal/nonterminal display names;
- nullable nonterminals;
- FIRST sets;
- FOLLOW sets;
- rules indexed by left-hand side.

The critical invariant is that grammar normalization must preserve the generated terminal language while making table construction finite and tractable.

## Table contract

`parser::glr::table` maps `AnalyzedGrammar` to `GLRTable`:

```text
AnalyzedGrammar -> GLRTable
```

The table has two layers:

1. **Admission support**, represented by `advance` rows.  This answers whether a terminal is even supported by a top state.
2. **Execution action**, represented by `action` rows.  This may contain optimized actions such as stack-shift bundles and guarded stack shifts.

A table optimization may change execution actions, merge states, or add synthetic states, but it must preserve admission/execution consistency.

## Advance contract

`parser::glr::advance` maps a concrete GSS through one terminal:

```text
(GLRTable, ParserGSS, TerminalID) -> ParserGSS
```

The predicate `stack_can_advance_on(table, stack, terminal)` is exact: it is not a heuristic.  It returns true exactly when at least one concrete parser path inside the GSS can legally consume the terminal under the optimized action rows and guarded stack predicates.

`stack_can_advance_on_any` is the lifted set-valued predicate:

```text
exists t in terminals. stack_can_advance_on(table, stack, t)
```

but it uses admission rows to avoid scanning impossible terminals unnecessarily.

## Compatibility contract

`compiler::glr` remains a hidden compatibility shim.  It must not become a second implementation.  It may re-export from `parser::glr`, but all new imports should point at `parser::glr`.
