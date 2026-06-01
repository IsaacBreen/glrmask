# Invariant catalogue

## AST invariants

1. `GrammarExpr::ExprNFA` is only valid as a complete nonterminal rule body.
2. `NamedRule::is_internal` only has semantic force for terminal rules.
3. `NamedGrammar::ignore` names a terminal rule body used as ignore pattern.
4. `NamedGrammar::rules` order remains stable across syntax-only operations unless a transform explicitly documents helper insertion.

## Transform invariants

1. Transform output must be another `NamedGrammar`.
2. Transforms may create named helper rules.
3. Transforms must preserve `start` and `ignore` unless they intentionally rewrite those bindings.
4. Transforms must not create `TerminalID` or `NonterminalID` values.

## Lowering invariants

1. Every non-internal terminal rule becomes exactly one terminal recognizer and one production binding the terminal to a parser nonterminal.
2. Internal-only terminal rules are available for terminal-body resolution but are not parser productions.
3. Generated helper nonterminals use `__` prefixes.
4. `nonterminal_names` excludes helper names beginning with `__`.
5. Duplicate emitted rules are removed after lowering while preserving first occurrence.
6. Terminal expression references are resolved through `Expr::Shared` and the terminal expression cache.

## Repeat invariants

1. Repeat helper caches are keyed by symbol and numeric bounds.
2. Exact-repeat helpers for the same symbol/count pair are reused.
3. Range-repeat helpers for the same symbol/min/max pair are reused where applicable.
4. Tree shape affects helper topology only, not language.

## Separated-sequence invariants

1. A required nullable item is not equivalent to an absent optional item.
2. A subtree's `can_be_empty` flag is a parent-composition signal, not an instruction to emit epsilon locally.
3. Separator symbols are emitted only between present subtrees.
4. Repetition items inside separated sequences thread the separator through the repeated pair.

## Renderer invariants

1. Renderers do not mutate grammars.
2. Renderers do not allocate flat ids.
3. Renderers may include comments for constructs with no native target-language surface syntax.
4. `NamedGrammar::to_lark` is a convenience delegate, not an implementation site.

## Compatibility invariants

1. Existing `crate::grammar::*` paths still resolve.
2. New implementation code should prefer `crate::grammar_ir::*`.
3. Compatibility shims should not grow new logic.
