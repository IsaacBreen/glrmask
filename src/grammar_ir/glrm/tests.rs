//! Tests for GLRM parser/renderer round trips.

    use super::*;
    use crate::grammar_ir::lower::lower;
    use crate::grammar_ir::flat::Symbol;

    fn single_path_terminal_names(
        lowered: &crate::grammar_ir::flat::GrammarDef,
        symbol: &Symbol,
    ) -> Vec<String> {
        match symbol {
            Symbol::Terminal(id) => vec![lowered.terminal_display_name(*id)],
            Symbol::Nonterminal(id) => {
                let rules = lowered
                    .rules
                    .iter()
                    .filter(|rule| rule.lhs == *id)
                    .collect::<Vec<_>>();
                assert_eq!(rules.len(), 1, "expected a single-path helper nonterminal");
                rules[0]
                    .rhs
                    .iter()
                    .flat_map(|child| single_path_terminal_names(lowered, child))
                    .collect()
            }
        }
    }

    #[test]
    fn parses_named_expr_nfa_definition() {
        let grammar = from_glrm(
            r#"
start obj;

fa obj ::= {
start 0;
accept 4;

0 -- "\"name\": " --> 1;
1 -- "," "\"email\": " --> 2;
1 -- "," "\"description\": " --> 3;
2 -- "," "\"thumbnail\": " --> 3;
2 --> 4;
3 --> 4;
};
"#,
        )
        .unwrap();

        assert_eq!(grammar.rules.len(), 1);
        assert!(matches!(grammar.rules[0].expr, GrammarExpr::ExprNFA(_)));
        lower(&grammar).unwrap();
    }

    #[test]
    fn dumps_expr_nfa_as_own_definition() {
        let grammar = from_glrm(
            r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- "a" --> 1;
};
"#,
        )
        .unwrap();
        let dumped = to_glrm(&grammar);
        assert!(dumped.contains("fa obj ::= {"), "{dumped}");
        assert!(dumped.contains("  start 0;"), "{dumped}");
        assert!(dumped.contains("  accept 1;"), "{dumped}");
        assert!(dumped.contains("  0 -- \"a\" --> 1;"), "{dumped}");
        assert!(!dumped.contains("ExprNFA("), "{dumped}");
    }

    #[test]
    fn expr_nfa_transition_symbols_accept_full_expressions() {
        let grammar = from_glrm(
            r#"
start obj;
fa obj ::= {
start 0;
accept 1;
0 -- [a-z] - "x" --> 1;
};
"#,
        )
        .unwrap();
        let GrammarExpr::ExprNFA(expr_nfa) = &grammar.rules[0].expr else {
            panic!("expected ExprNFA rule");
        };
        assert!(matches!(
            expr_nfa.symbols.first(),
            Some(GrammarExpr::Exclude { .. })
        ));
    }

    #[test]
    fn exclude_rhs_sequence_requires_parentheses() {
        let err = from_glrm(
            r#"
start z;
nt A ::= a b | c d | e f;
nt z ::= x (A - c d);
"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("RHS sequence subtraction must be parenthesized"), "{err}");
    }

    #[test]
    fn grouped_exclude_rhs_preserves_parenthesized_ref() {
        let grammar = from_glrm(
            r#"
start z;
nt A ::= a b | c d | e f;
nt B ::= c d | e f;
nt z ::= x (A - B) | x (A - (B));
"#,
        )
        .unwrap();
        let GrammarExpr::Choice(options) = &grammar.rules[2].expr else {
            panic!("expected choice");
        };
        assert!(matches!(
            options[0],
            GrammarExpr::Sequence(_)
        ));
        let GrammarExpr::Sequence(second_parts) = &options[1] else {
            panic!("expected sequence");
        };
        let GrammarExpr::Exclude { exclude, .. } = &second_parts[1] else {
            panic!("expected exclude expr");
        };
        assert!(matches!(exclude.as_ref(), GrammarExpr::Grouped(_)));
    }

    #[test]
    fn lowering_subtracts_exact_nonterminal_alternatives() {
        let grammar = from_glrm(
            r#"
start z;
nt A ::= "a" "b" | "c" "d" | "e" "f";
nt B ::= "c" "d" | "e" "f";
nt z ::= "x" (A - B);
"#,
        )
        .unwrap();

        let lowered = lower(&grammar).unwrap();
        let z_rule = lowered
            .rules
            .iter()
            .find(|rule| rule.lhs == lowered.start)
            .expect("start rule should exist");
        assert_eq!(z_rule.rhs.len(), 2);

        let Symbol::Nonterminal(filtered_nt) = z_rule.rhs[1] else {
            panic!("expected filtered nonterminal");
        };
        let filtered_rules = lowered
            .rules
            .iter()
            .filter(|rule| rule.lhs == filtered_nt)
            .collect::<Vec<_>>();
        assert_eq!(filtered_rules.len(), 1);
        assert_eq!(filtered_rules[0].rhs.len(), 2);

        let filtered_terminals = filtered_rules[0]
            .rhs
            .iter()
            .flat_map(|symbol| single_path_terminal_names(&lowered, symbol))
            .collect::<Vec<_>>();
        assert_eq!(filtered_terminals, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn lowering_rejects_parenthesized_ref_without_exact_alternative() {
        let grammar = from_glrm(
            r#"
start z;
nt A ::= "a" "b" | "c" "d" | "e" "f";
nt B ::= "c" "d" | "e" "f";
nt z ::= "x" (A - (B));
"#,
        )
        .unwrap();

        let err = lower(&grammar).unwrap_err().to_string();
        assert!(err.contains("no exact alternative"), "{err}");
    }

    #[test]
    fn rejects_nested_expr_nfa_at_lowering() {
        let nfa_rule = from_glrm(
            r#"
start inner;
fa inner ::= {
start 0;
accept 1;
0 -- "a" --> 1;
};
"#,
        )
        .unwrap()
        .rules
        .into_iter()
        .next()
        .unwrap();

        let grammar = NamedGrammar {
            rules: vec![NamedRule {
                name: "start".to_string(),
                expr: GrammarExpr::Sequence(vec![nfa_rule.expr, GrammarExpr::Literal(b"b".to_vec())]),
                is_terminal: false,
                is_internal: false,
            }],
            start: "start".to_string(),
            ignore: None,
        };

        let err = lower(&grammar).unwrap_err().to_string();
        assert!(err.contains("complete expression of a nonterminal rule"), "{err}");
    }
