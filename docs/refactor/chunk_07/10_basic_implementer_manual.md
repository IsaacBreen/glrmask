# Basic implementer manual

This file assumes the reader knows Rust basics but not this crate.

## Adding a new grammar expression

Suppose you want to add a new source syntax construct.

1. Add the variant to `src/grammar_ir/ast.rs` in `GrammarExpr`.
2. Add traversal handling to `NamedGrammar::prune_unreachable` if the variant may contain references.
3. Add rendering in `src/grammar_ir/render/lark.rs`.
4. Add GLRM rendering in `src/grammar_ir/render/glrm.rs` if it must round-trip through GLRM.
5. Add GLRM parsing in `src/grammar_ir/glrm/mod.rs` if it is part of the GLRM format.
6. Add nullability behavior in `src/grammar_ir/lower/terminal_expr.rs` if the construct can affect emptiness.
7. Add lowering behavior in `src/grammar_ir/lower/mod.rs` or a more specific lowering submodule.
8. Add transform behavior in `src/grammar_ir/transforms/*` if any transform recursively walks expressions.
9. Add tests near the code most responsible for the invariant.

Do not add lowering code to `ast.rs`.

## Adding a new transform

1. Create `src/grammar_ir/transforms/my_transform.rs`.
2. Add `pub mod my_transform;` to `transforms/mod.rs`.
3. Make the function accept `NamedGrammar` or `&mut NamedGrammar`.
4. Do not accept or return `GrammarDef`.
5. Add a short mathematical doc comment saying what language is preserved.
6. Add it to the importer or transform pipeline only after deciding order relative to existing transforms.

## Adding a new renderer

1. Create `src/grammar_ir/render/my_format.rs`.
2. Add it to `render/mod.rs`.
3. Accept `&NamedGrammar` or `&GrammarExpr`.
4. Return text or a formatting result.
5. Do not mutate the grammar.
6. Do not allocate terminal/nonterminal ids.

## Debugging lowering

Start in `lower/mod.rs::lower`.

Then inspect:

- general expression lowering: `lower_expr` and `lower_expr_terminalish`;
- repeat behavior: `lower/repeat.rs`;
- separated sequence behavior: `lower/separated_sequence.rs`;
- terminal regex conversion: `lower/terminal_expr.rs`;
- ExprNFA behavior: `lower/expr_nfa_lower.rs`.

When a bug appears downstream in GLR or DWA construction, first ask whether the flat `GrammarDef` is already wrong. If so, the bug belongs in `grammar_ir::lower` or `grammar_ir::transforms`, not in the DWA code.
