# Chunk 07 overview: Grammar IR and lowering split

This chunk turns the old `src/grammar/ast.rs` / `src/grammar/*.rs` area into an explicit **grammar intermediate representation** subsystem.

The core mathematical distinction is now visible in the file tree:

```text
source frontends
  -> grammar_ir::ast::NamedGrammar
  -> grammar_ir::transforms::*
  -> grammar_ir::lower::lower
  -> grammar_ir::flat::GrammarDef
  -> GLR / Terminal DWA / scan relation / Parser DWA / runtime
```

The old `src/grammar/*` namespace remains, but only as a compatibility shim. New code should import from `crate::grammar_ir::*`.

## Why this is the next chunk

Chunks 03 through 06 clarified the compile-side automata objects: Terminal DWA, Parser DWA, and scan relation. The next upstream ambiguity was the grammar layer. Previously, one file mixed all of the following concerns:

1. named grammar syntax;
2. Lark-like printing;
3. terminal expression conversion;
4. nullability analysis;
5. bounded-repeat lowering;
6. separated-sequence lowering;
7. ExprNFA lowering;
8. exact alternative subtraction during lowering;
9. final conversion to `GrammarDef`;
10. tests.

That was mathematically misleading. The parser/lexer/compiler pipeline should not look as if syntax, denotation-preserving rewrites, and representation-changing lowering are the same object.

## Source shape after this chunk

```text
src/grammar/ast.rs
src/grammar/exact_subtraction_lowering.rs
src/grammar/expr_nfa.rs
src/grammar/factoring.rs
src/grammar/flat.rs
src/grammar/glrm.rs
src/grammar/mod.rs
src/grammar/named_simplify.rs
src/grammar/terminal_choice_promotion.rs
src/grammar_ir/README.md
src/grammar_ir/ast.rs
src/grammar_ir/expr_nfa.rs
src/grammar_ir/flat.rs
src/grammar_ir/glrm/mod.rs
src/grammar_ir/glrm/tests.rs
src/grammar_ir/lower/README.md
src/grammar_ir/lower/exact_subtraction.rs
src/grammar_ir/lower/expr_nfa_lower.rs
src/grammar_ir/lower/mod.rs
src/grammar_ir/lower/repeat.rs
src/grammar_ir/lower/separated_sequence.rs
src/grammar_ir/lower/terminal_expr.rs
src/grammar_ir/lower/tests.rs
src/grammar_ir/mod.rs
src/grammar_ir/render/README.md
src/grammar_ir/render/glrm.rs
src/grammar_ir/render/lark.rs
src/grammar_ir/render/mod.rs
src/grammar_ir/transforms/README.md
src/grammar_ir/transforms/exact_subtraction/mod.rs
src/grammar_ir/transforms/exact_subtraction/tests.rs
src/grammar_ir/transforms/factor.rs
src/grammar_ir/transforms/mod.rs
src/grammar_ir/transforms/simplify.rs
src/grammar_ir/transforms/terminal_choice.rs
```

## File metrics

| file | LOC | fn | struct | enum |
| --- | ---: | ---: | ---: | ---: |
| `src/grammar_ir/ast.rs` | 198 | 4 | 2 | 2 |
| `src/grammar_ir/expr_nfa.rs` | 311 | 25 | 2 | 0 |
| `src/grammar_ir/flat.rs` | 121 | 9 | 2 | 2 |
| `src/grammar_ir/glrm/mod.rs` | 854 | 35 | 2 | 1 |
| `src/grammar_ir/glrm/tests.rs` | 225 | 9 | 0 | 0 |
| `src/grammar_ir/lower/exact_subtraction.rs` | 192 | 4 | 0 | 0 |
| `src/grammar_ir/lower/expr_nfa_lower.rs` | 184 | 5 | 0 | 0 |
| `src/grammar_ir/lower/mod.rs` | 725 | 19 | 1 | 0 |
| `src/grammar_ir/lower/repeat.rs` | 424 | 14 | 0 | 1 |
| `src/grammar_ir/lower/separated_sequence.rs` | 203 | 4 | 0 | 0 |
| `src/grammar_ir/lower/terminal_expr.rs` | 346 | 7 | 0 | 0 |
| `src/grammar_ir/lower/tests.rs` | 156 | 8 | 0 | 0 |
| `src/grammar_ir/mod.rs` | 17 | 0 | 0 | 0 |
| `src/grammar_ir/render/glrm.rs` | 258 | 9 | 0 | 0 |
| `src/grammar_ir/render/lark.rs` | 257 | 7 | 0 | 0 |
| `src/grammar_ir/render/mod.rs` | 7 | 0 | 0 | 0 |
| `src/grammar_ir/transforms/exact_subtraction/mod.rs` | 688 | 19 | 9 | 0 |
| `src/grammar_ir/transforms/exact_subtraction/tests.rs` | 363 | 16 | 1 | 0 |
| `src/grammar_ir/transforms/factor.rs` | 460 | 20 | 1 | 0 |
| `src/grammar_ir/transforms/mod.rs` | 15 | 0 | 0 | 0 |
| `src/grammar_ir/transforms/simplify.rs` | 577 | 24 | 1 | 0 |
| `src/grammar_ir/transforms/terminal_choice.rs` | 566 | 24 | 5 | 1 |
| `src/grammar/ast.rs` | 5 | 0 | 0 | 0 |
| `src/grammar/exact_subtraction_lowering.rs` | 3 | 0 | 0 | 0 |
| `src/grammar/expr_nfa.rs` | 3 | 0 | 0 | 0 |
| `src/grammar/factoring.rs` | 3 | 0 | 0 | 0 |
| `src/grammar/flat.rs` | 3 | 0 | 0 | 0 |
| `src/grammar/glrm.rs` | 3 | 0 | 0 | 0 |
| `src/grammar/mod.rs` | 14 | 0 | 0 | 0 |
| `src/grammar/named_simplify.rs` | 3 | 0 | 0 | 0 |
| `src/grammar/terminal_choice_promotion.rs` | 3 | 0 | 0 | 0 |

## Acceptance highlights

- `src/grammar_ir/ast.rs` is now the syntax data model only.
- `src/grammar_ir/lower/` owns the only conversion from `NamedGrammar` to `GrammarDef`.
- `src/grammar_ir/transforms/` owns named-grammar-to-named-grammar rewrites.
- `src/grammar_ir/render/` owns observational formatting.
- `src/grammar_ir/glrm/` parses GLRM; `src/grammar_ir/render/glrm.rs` renders GLRM.
- No `src/grammar_ir/*.rs` implementation file exceeds 900 LOC.
