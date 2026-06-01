# Grammar IR lowering

Lowering turns `NamedGrammar` into `GrammarDef`. It is the first point where
nonterminal ids, terminal ids, and generated helper rules are allocated.

The split is by semantic obligation:

- `mod.rs`: orchestration, terminal id allocation, general expression emission.
- `repeat.rs`: bounded/unbounded cardinality lowering.
- `separated_sequence.rs`: separator placement and optional-item absence.
- `expr_nfa_lower.rs`: automata-shaped expression lowering.
- `terminal_expr.rs`: conversion to lexer expressions and nullability.
- `exact_subtraction.rs`: local exact alternative subtraction during emission.
