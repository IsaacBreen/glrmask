# Visibility policy

Visibility should express the mathematical boundary.

## `pub`

Use `pub` only for items intended to be reachable through a public or compatibility-facing module path.

Examples:

- `GrammarExpr`
- `NamedGrammar`
- `NamedRule`
- `lower`
- `expr_to_grammar_expr`
- `from_glrm`
- `to_glrm`

## `pub(crate)`

Use `pub(crate)` for helpers shared across crate subsystems but not public API.

Examples:

- renderer escape helpers used by lowering;
- compatibility shim functions inside crate-private modules;
- diagnostic-only internals.

## `pub(super)`

Use `pub(super)` for methods implemented in child modules but called by their parent.

Examples:

- `Lowerer::emit_repeat_range`
- `Lowerer::lower_separated_sequence_inner`
- `Lowerer::emit_expr_nfa`
- `Lowerer::exact_nonterminal_subtraction_expr`

## private

Use private visibility for implementation details inside one file.

Examples:

- GLRM lexer tokens;
- transform helper structs;
- local recursive walkers;
- exact-subtraction partition helpers.

## Compatibility shims

Old `src/grammar/*` modules may use `pub use` for new items because the whole `grammar` module is crate-private. Do not add logic to these shims. If a shim needs logic to repair visibility, it should be a one-line wrapper that forwards to `grammar_ir`.
