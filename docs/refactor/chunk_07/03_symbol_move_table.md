# Symbol move table

## Syntax objects

| symbol | old location | new location | reason |
| --- | --- | --- | --- |
| `GrammarExpr` | `grammar::ast` | `grammar_ir::ast` | source-level syntax |
| `CommaSepShape` | `grammar::ast` | `grammar_ir::ast` | policy enum attached to separated-sequence syntax/lowering |
| `NamedRule` | `grammar::ast` | `grammar_ir::ast` | named syntax binding |
| `NamedGrammar` | `grammar::ast` | `grammar_ir::ast` | frontend output object |
| `terminal_names_set` | `grammar::ast` | `grammar_ir::ast` | local query on named syntax |
| `prune_unreachable` | `grammar::ast` | `grammar_ir::ast` | local graph query on named syntax |

## Rendering

| symbol | old location | new location | reason |
| --- | --- | --- | --- |
| `to_lark` implementation | `NamedGrammar` impl in `ast.rs` | `render::lark::to_lark` with delegate method | observation, not syntax |
| `grammar_expr_to_lark` | `ast.rs` | `render::lark.rs` | expression formatting |
| `u8set_to_class_def` | `ast.rs` | `render::lark.rs` | character-class formatting |
| `escape_byte` | `ast.rs` | `render::lark.rs` | byte escaping for renderer/lowerer display names |
| `regex_escape_byte` | `ast.rs` | `render::lark.rs` | regex literal escape helper |
| `to_glrm` | `grammar::glrm` | `render::glrm` and re-exported by `grammar_ir::glrm` | GLRM rendering is separate from parsing |

## Lowering

| symbol | old location | new location | reason |
| --- | --- | --- | --- |
| `Lowerer` | `grammar::ast` | `grammar_ir::lower` | lowering context |
| `lower` | `grammar::ast` | `grammar_ir::lower` | representation-changing morphism |
| `grammar_expr_to_expr` | `grammar::ast` | `lower::terminal_expr` | terminal-language conversion |
| `grammar_expr_is_nullable` | `grammar::ast` | `lower::terminal_expr` | semantic predicate used by lowering |
| `compute_rule_nullability` | `grammar::ast` | `lower::terminal_expr` | fixed point over named rule syntax |
| `validate_expr_nfa_placement` | `grammar::ast` | `lower::terminal_expr` | source-form precondition |
| `expr_to_grammar_expr` | `grammar::ast` | `lower::terminal_expr` | diagnostic/reconstruction helper |
| `repeat_tree_shape` | `grammar::ast` | `lower::repeat` | repeat lowering policy |
| `repeat_exact_nonterminal` | `grammar::ast` | `lower::repeat` | repeat helper nonterminal construction |
| `repeat_range_nonterminal` | `grammar::ast` | `lower::repeat` | repeat helper nonterminal construction |
| `emit_repeat_range` | `grammar::ast` | `lower::repeat` | repeat production emission |
| `lower_sepseq_*` | `grammar::ast` | `lower::separated_sequence` | separator placement invariant |
| `emit_expr_nfa` | `grammar::ast` | `lower::expr_nfa_lower` | automaton-to-production lowering |
| `exact_nonterminal_subtraction_expr` | `grammar::ast` | `lower::exact_subtraction` | local exact alt subtraction |

## Transforms

| symbol/file | old location | new location | reason |
| --- | --- | --- | --- |
| `factor_named_grammar` | `grammar::factoring` | `grammar_ir::transforms::factor` | named grammar transform |
| `simplify_named_grammar` | `grammar::named_simplify` | `grammar_ir::transforms::simplify` | named grammar transform |
| `promote_choice_terminals_exact` | `grammar::terminal_choice_promotion` | `grammar_ir::transforms::terminal_choice` | named grammar transform |
| `lower_exact_subtractions` | `grammar::exact_subtraction_lowering` | `grammar_ir::transforms::exact_subtraction` | named grammar transform |

## GLRM

| symbol | old location | new location | reason |
| --- | --- | --- | --- |
| `from_glrm` | `grammar::glrm` | `grammar_ir::glrm` | source parser into named grammar IR |
| `Lexer`/`GlrmParser` | `grammar::glrm` | `grammar_ir::glrm` | parser implementation details |
| `to_glrm` | `grammar::glrm` | `render::glrm`, re-exported | renderer separated from parser |
