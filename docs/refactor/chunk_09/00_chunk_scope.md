# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Scope statement

Chunk 09 is about the parser substrate below the paper automata.  It is not about JSON Schema, grammar syntax, Terminal DWA determinization, Parser DWA determinization, runtime mask materialization, or Python bindings except where those areas import GLR parser types.

The chunk's single-sentence purpose is:

> Make the GLR parser machinery visibly shared between compile-time construction and runtime stack advancement.

## In scope

- Namespace move from `compiler::glr` to `parser::glr`.
- Source-tree distinction between grammar analysis, table construction/optimization, and stack advancement.
- Naming correction from `parser` to `advance` for the old execution module.
- Naming correction from speculative `may_advance` to exact `can_advance` predicates.
- Typed local options for GLR table and stack-advance policy.
- Readability splits for the largest GLR files using textual includes.
- Self-contained docs and ledgers for the next compile-repair phase.

## Out of scope

- Changing GLR recognition semantics.
- Changing generated table rows.
- Changing stack-effect recognizer semantics.
- Removing all compatibility shims.
- Proving or benchmarking performance equivalence.
- Turning textual include fragments into fully independent Rust submodules.

## Why textual includes are acceptable in this chunk

A perfect cleanup would turn every fragment into a true Rust module.  Doing that prematurely forces visibility decisions before the actual mathematical proof obligations are audited.  This chunk therefore splits the source physically while preserving the old single-module privacy model.  That is the right intermediate state: reviewers can inspect smaller files immediately, and a later compile-repair pass can make helper visibility explicit after tests identify the precise dependency surface.
