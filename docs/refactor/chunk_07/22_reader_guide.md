# Reader guide for the new grammar layer

## I want to understand source grammar syntax

Read:

1. `src/grammar_ir/README.md`
2. `src/grammar_ir/ast.rs`
3. `src/grammar_ir/flat.rs`

## I want to understand EBNF/Lark/JSON Schema import

Start in `src/import/` for now. Those modules still use old shim imports in places. Follow calls until they return `NamedGrammar`.

## I want to understand how source syntax becomes productions

Read:

1. `src/grammar_ir/lower/README.md`
2. `src/grammar_ir/lower/mod.rs`
3. `src/grammar_ir/lower/repeat.rs`
4. `src/grammar_ir/lower/separated_sequence.rs`
5. `src/grammar_ir/lower/terminal_expr.rs`

## I want to understand exact subtraction

Read both:

1. `src/grammar_ir/transforms/exact_subtraction/mod.rs`
2. `src/grammar_ir/lower/exact_subtraction.rs`

The first is a transform. The second is a local lowering helper.

## I want to understand GLRM

Read:

1. `src/grammar_ir/render/glrm.rs` for output;
2. `src/grammar_ir/glrm/mod.rs` for parser;
3. `src/grammar_ir/glrm/tests.rs` for round-trip expectations.

## I want to make a small bug fix

Prefer editing the module whose mathematical contract owns the bug. Do not put convenience code in `ast.rs` just because the type is defined there.
