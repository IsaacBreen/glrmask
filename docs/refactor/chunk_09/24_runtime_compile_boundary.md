# Chunk 09: GLR parser/table cleanup

This document belongs to the Chunk 09 package.  Chunk 09 promotes GLR parser machinery out of the historical `compiler::glr` namespace and into the parser-domain namespace `parser::glr`.  It also splits the largest GLR files into source-reading fragments without changing the mathematical table/advance semantics.


## Runtime/compile boundary

Chunk 09 draws a sharper line:

```text
compile pipeline -> builds GLR table and Parser DWA
runtime commit   -> executes parser-stack advance relation
runtime mask     -> walks compiled Parser DWA
```

The GLR table is constructed at compile time but retained in the runtime artifact because commit needs it.  Therefore, `GLRTable` cannot honestly live in a compile-only namespace.

## Important runtime users

- `runtime/artifact.rs` stores the table.
- `runtime/commit/mod.rs` executes parser advancement and exact applicability.
- `runtime/mask/mod.rs` uses parser accumulator/label conventions.
- `runtime/state.rs` stores parser GSS state.

## Important compile users

- `compile/pipeline/analysis.rs` creates `AnalyzedGrammar` and `GLRTable`.
- `compile/terminal_dwa/*` reads analyzed grammar/table helpers.
- `compile/parser_dwa/*` reads GLR labels and grammar analysis.
- `compiler/stages/templates/*` still uses GLR table/labels until template DFA gets its own cleanup chunk.
