# Import migration backlog

Chunk 07 intentionally keeps compatibility shims. This backlog lists the migration that should happen later, preferably after at least one compile-repair pass.

## Rule

New implementation code should import from `crate::grammar_ir`. Old call sites may continue to import from `crate::grammar` until touched for another reason.

## High-priority migrations

### Import frontends

Files under `src/import/` should eventually use `grammar_ir` directly because they produce `NamedGrammar` and apply transforms.

Likely replacements:

```text
crate::grammar::ast -> crate::grammar_ir::ast
crate::grammar::flat -> crate::grammar_ir::flat
crate::grammar::glrm -> crate::grammar_ir::glrm or frontend::glrm later
crate::grammar::factoring -> crate::grammar_ir::transforms::factor
crate::grammar::named_simplify -> crate::grammar_ir::transforms::simplify
crate::grammar::terminal_choice_promotion -> crate::grammar_ir::transforms::terminal_choice
crate::grammar::exact_subtraction_lowering -> crate::grammar_ir::transforms::exact_subtraction
```

### Compile pipeline

The compile pipeline consumes `GrammarDef`, so it should eventually import `crate::grammar_ir::flat::GrammarDef`.

### GLR table construction

GLR analysis and table construction consume flat grammar types. They should import `grammar_ir::flat` once the compatibility period ends.

## Medium-priority migrations

### Runtime

Runtime mostly needs `TerminalID`; it can import from `grammar_ir::flat` or, later, from a dedicated `ids` module.

### Tests

Tests should use new paths if they are edited. Do not churn every test solely for import paths unless a compile error requires it.

## Low-priority migrations

### Comments and docs

Search for old textual references to `grammar/ast.rs` and update them when relevant. Avoid misleading future readers into thinking `src/grammar/ast.rs` is still implementation.

## Eventually remove shims

Remove `src/grammar/*` shims only when:

1. all internal imports use `grammar_ir`;
2. no public API promises old internal paths;
3. tests and benches have been migrated;
4. changelog notes the internal module rename.
