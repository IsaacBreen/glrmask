# Review checklist

Use this checklist before continuing to the next chunk.

## File tree

- [ ] `src/grammar_ir/mod.rs` exists and declares `ast`, `flat`, `expr_nfa`, `lower`, `transforms`, `render`, and `glrm`.
- [ ] `src/grammar/mod.rs` is only a compatibility namespace.
- [ ] Old `src/grammar/*.rs` files contain only re-exports.
- [ ] `src/lib.rs` declares `pub(crate) mod grammar_ir;`.

## Line count

- [ ] No `src/grammar_ir` Rust implementation file exceeds 900 LOC.
- [ ] `src/grammar_ir/ast.rs` is small enough to read top-to-bottom.
- [ ] `src/grammar_ir/lower/mod.rs` is under 900 LOC.
- [ ] `src/grammar_ir/glrm/mod.rs` is under 900 LOC after renderer/test extraction.
- [ ] `src/grammar_ir/transforms/exact_subtraction/mod.rs` is under 900 LOC after test extraction.

## Mathematical boundaries

- [ ] `ast.rs` does not contain `Lowerer`.
- [ ] `ast.rs` does not contain GLRM lexer/parser code.
- [ ] `render/lark.rs` does not construct `GrammarDef`.
- [ ] `render/glrm.rs` does not parse.
- [ ] `lower/mod.rs` owns `lower`.
- [ ] `transforms/` functions return/modify `NamedGrammar`, not `GrammarDef`.

## Compatibility

- [ ] `crate::grammar::ast::GrammarExpr` still resolves.
- [ ] `crate::grammar::ast::lower` still resolves.
- [ ] `crate::grammar::glrm::from_glrm` still resolves.
- [ ] `crate::grammar::glrm::to_glrm` still resolves.
- [ ] Existing importers can keep old imports until a later migration chunk.

## Later compile repair candidates

Because compilation is intentionally deferred, review these first once compile work starts:

1. any private/public re-export mismatch in `grammar::ast::comma_sep_shape`;
2. unused imports introduced by split files;
3. misplaced test imports after extracting test modules;
4. `pub(super)` visibility on methods implemented in child modules;
5. renderer helper visibility for `regex_escape_byte` and `u8set_to_class_def`.
