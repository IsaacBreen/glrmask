# File-by-file surgery ledger

This ledger describes the concrete work applied in Chunk 07 and the work it deliberately leaves for later chunks.

## Old files replaced by compatibility shims

### `src/grammar/ast.rs`

Now re-exports:

- `crate::grammar_ir::ast::*`
- `crate::grammar_ir::lower::{lower, expr_to_grammar_expr}`
- `crate::grammar_ir::lower::separated_sequence::comma_sep_shape` with crate visibility

Reason: many call sites still import `crate::grammar::ast::{lower, GrammarExpr, NamedGrammar}`. Breaking all call sites in the same chunk would obscure the actual conceptual split.

### `src/grammar/flat.rs`

Now re-exports `crate::grammar_ir::flat::*`.

### `src/grammar/expr_nfa.rs`

Now re-exports `crate::grammar_ir::expr_nfa::*`.

### `src/grammar/factoring.rs`

Now re-exports `crate::grammar_ir::transforms::factor::*`.

### `src/grammar/named_simplify.rs`

Now re-exports `crate::grammar_ir::transforms::simplify::*`.

### `src/grammar/terminal_choice_promotion.rs`

Now re-exports `crate::grammar_ir::transforms::terminal_choice::*`.

### `src/grammar/exact_subtraction_lowering.rs`

Now re-exports `crate::grammar_ir::transforms::exact_subtraction::*`.

### `src/grammar/glrm.rs`

Now re-exports `crate::grammar_ir::glrm::*`.

## New files

### `src/grammar_ir/ast.rs`

Contains:

- `GrammarExpr`
- `CommaSepShape`
- `NamedRule`
- `NamedGrammar`
- `NamedGrammar::terminal_names_set`
- `NamedGrammar::prune_unreachable`
- `NamedGrammar::to_lark` as a renderer delegate only

Does not contain:

- lowerer state;
- regex parsing;
- flat grammar construction;
- Lark formatting internals;
- repeat lowering;
- GLRM parsing.

### `src/grammar_ir/lower/mod.rs`

Contains the lowering orchestrator and the `Lowerer` state. It owns the public `lower` function and the common helpers needed by all lowering submodules.

### `src/grammar_ir/lower/repeat.rs`

Contains bounded and unbounded repeat tree construction. This is where `RepeatRange` becomes helper nonterminals and productions.

### `src/grammar_ir/lower/separated_sequence.rs`

Contains the separator-placement algorithm. The key invariant is that optional absence and nullable derivation are not the same fact.

### `src/grammar_ir/lower/expr_nfa_lower.rs`

Contains `ExprNFA` lowering. It turns automaton states into nonterminals and transitions into productions.

### `src/grammar_ir/lower/terminal_expr.rs`

Contains terminal-expression conversion, nullability, ExprNFA placement validation, and conversion back from lexer expressions for diagnostics.

### `src/grammar_ir/lower/exact_subtraction.rs`

Contains local exact-alternative subtraction during lowering. This is distinct from the whole-grammar exact-subtraction transform.

### `src/grammar_ir/transforms/*`

Contains named-grammar-to-named-grammar transforms.

### `src/grammar_ir/render/lark.rs`

Contains human-readable Lark-like formatting and character-class escape helpers.

### `src/grammar_ir/render/glrm.rs`

Contains GLRM rendering.

### `src/grammar_ir/glrm/mod.rs`

Contains GLRM lexing and parsing.

## Deliberately deferred

1. Global import migration from `crate::grammar::*` to `crate::grammar_ir::*`.
2. JSON Schema frontend restructuring.
3. More aggressive transform unification.
4. Rewriting lowerer methods to use smaller context structs instead of one shared `Lowerer`.
5. Replacing env-var reads with typed compile options.
