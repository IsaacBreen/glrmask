# Architecture

This document is the publication-facing map from the paper to the implementation. It is intentionally skeletal in the baseline cleanup chunk and should be expanded as modules are renamed and split.

## Core paper objects

| Paper term | Intended code home | Responsibility |
| --- | --- | --- |
| Terminal DWA | `src/compile/terminal_dwa/` | Compile terminal-sequence acceptance into weights over lexer-state/token pairs. |
| Parser DWA | `src/compile/parser_dwa/` | Compile parser-stack-prefix acceptance into weights over lexer-state/token pairs. |
| Mask | `src/runtime/mask/` | Walk active stacks through the Parser DWA and materialize a vocabulary mask. |
| Commit | `src/runtime/commit/` | Scan token bytes, complete terminals, and advance parser state. |
| Template DFA | `src/compiler/stages/templates/` and `src/runtime/commit/template_advance.rs` | Precompute repeated-template scanner/parser behavior used by commit fast paths. |
| GLR parser domain | `src/parser/glr/` | Analyze flat grammars, build GLR tables, and execute parser stack effects shared by compile and runtime. |

## Current top-level source layout

```text
src/
  automata/    reusable lexer, weighted, and unweighted automata machinery
  compile/     Terminal DWA, Parser DWA, and scan-relation construction
  parser/      GLR parser analysis, table construction, and stack advancement
  compiler/    remaining legacy compiler infrastructure and compatibility shims
  ds/          shared data structures: weights, bitsets, GSS, vocab prefix trees
  grammar/     grammar ASTs, named grammar utilities, GLRM rendering, simplification
  import/      EBNF, Lark, JSON Schema, and GLRM frontends
  api/         publication facade: Constraint, ConstraintState, Vocab, errors, profiles
  diagnostics/ diagnostics/cache/front-end helpers outside the core API
  runtime/     Constraint, ConstraintState, Mask, Commit, serialization
  error.rs     crate error type
  vocab.rs     vocabulary representation
```

## Target cleanup direction

Later chunks should make the paper operations obvious in the file tree:

```text
src/
  api/
  frontend/
  grammar/
  compile/
    terminal_dwa/
    parser_dwa/
    scan_relation/
    phase_graph/
  runtime/
    mask/
    commit/
    artifact/
  automata/
  parser/
    glr/
  weights/
  diagnostics/
```

The exact target tree may change during implementation, but every rename should improve one of three things: paper alignment, public API clarity, or separation of concerns.
