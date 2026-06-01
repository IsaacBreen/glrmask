# Open questions after Chunk 07

## Should `CommaSepShape` live in `ast.rs`?

It currently remains in `ast.rs` because it is part of how `SeparatedSequence` lowering is configured and because old code treated it as grammar-adjacent. A cleaner future may move all lowering policy enums to `compile/options.rs` or `grammar_ir/lower/options.rs`.

## Should renderer escape helpers live under `render/lark.rs`?

`regex_escape_byte` is useful to lowering when constructing terminal regex patterns. That makes it slightly more general than a renderer helper. A future `grammar_ir::escape` module may be cleaner.

## Should `expr_to_grammar_expr` be in lowering?

It converts from lexer `Expr` back to grammar syntax. It is currently in `lower::terminal_expr` because it is tightly related to terminal expression conversion. It could move to `grammar_ir::conversion` later.

## Should old `src/grammar` shims remain long term?

Probably not. They are useful during refactor. Once internal imports are migrated, they should be removed or left as private aliases only if benches/tests still need them.

## Should GLRM parsing be under `frontend`?

Probably yes in the final publication tree. For now, it stays under `grammar_ir::glrm` because it is an exchange-format parser tightly coupled to grammar IR.
