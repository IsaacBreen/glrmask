# Grammar IR

This directory is the named-grammar boundary of the crate. It is deliberately
not the parser table, not the lexer, and not the runtime.

The mathematical pipeline is:

```text
source frontends
  -> NamedGrammar / GrammarExpr          (this directory: ast.rs)
  -> denotation-preserving transforms    (transforms/)
  -> flat GrammarDef productions         (lower/)
  -> GLR table + Terminal DWA + Parser DWA
```

Rules for this directory:

1. `ast.rs` owns syntax only.
2. `transforms/` rewrites `NamedGrammar` to another `NamedGrammar`.
3. `lower/` changes representation from named syntax to flat productions.
4. `render/` observes or serializes; it never allocates compiler IDs.
5. `glrm/` parses the GLRM exchange format; the renderer is under `render/`.
