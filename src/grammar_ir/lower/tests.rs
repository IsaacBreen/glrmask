//! Tests for named-grammar lowering.

use super::{lower, GrammarExpr, NamedGrammar};
use crate::grammar_ir::ast::NamedRule;

fn nonterminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: false,
            is_internal: false,
        }
    }

    fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: false,
        }
    }

    fn literal(text: &str) -> GrammarExpr {
        GrammarExpr::Literal(text.as_bytes().to_vec())
    }

    fn subtract(lhs: &str, exclude: GrammarExpr) -> GrammarExpr {
        GrammarExpr::Exclude {
            expr: Box::new(GrammarExpr::Ref(lhs.to_string())),
            exclude: Box::new(exclude),
        }
    }

    #[test]
    fn exact_subtraction_matches_nonterminal_alias_body() {
        let grammar = NamedGrammar {
            rules: vec![
                terminal("JSON_STRING_BODY", literal("body\"")),
                nonterminal(
                    "json_string",
                    GrammarExpr::Sequence(vec![
                        literal("\""),
                        GrammarExpr::Ref("JSON_STRING_BODY".to_string()),
                    ]),
                ),
                nonterminal(
                    "json_value",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("json_string".to_string()),
                        literal("0"),
                    ]),
                ),
                nonterminal(
                    "start",
                    subtract("json_value", GrammarExpr::Ref("json_string".to_string())),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_canonicalization_is_cycle_safe() {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "loop",
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("y"),
                    ]),
                ),
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        GrammarExpr::Ref("loop".to_string()),
                        literal("x"),
                    ]),
                ),
                nonterminal("start", subtract("A", literal("z"))),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let err = lower(&grammar).unwrap_err();
        assert!(format!("{err}").contains("no exact alternative"), "{err}");
    }

    #[test]
    fn lower_deduplicates_identical_rules() {
        let grammar = NamedGrammar {
            rules: vec![nonterminal(
                "start",
                GrammarExpr::Choice(vec![literal("a"), literal("a")]),
            )],
            start: "start".to_string(),
            ignore: None,
        };

        let gdef = lower(&grammar).unwrap();
        let start_rules = gdef
            .rules
            .iter()
            .filter(|rule| rule.lhs == gdef.start)
            .count();
        assert_eq!(start_rules, 1, "duplicate alternatives should not create duplicate rules");
    }

    #[test]
    fn nonnullable_sequence_with_nonnullable_part_reduces_rules() {
        let grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "body",
                    GrammarExpr::Choice(vec![
                        literal("abc"),
                        GrammarExpr::Epsilon,
                    ]),
                ),
                nonterminal(
                    "item",
                    GrammarExpr::RepeatOne(Box::new(GrammarExpr::Sequence(vec![
                        literal("{"),
                        GrammarExpr::Ref("body".to_string()),
                        literal("}"),
                    ]))),
                ),
                nonterminal("start", GrammarExpr::Ref("item".to_string())),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let gdef = lower(&grammar).unwrap();
        let brace_rules_count = gdef
            .rules
            .iter()
            .filter(|rule| {
                matches!(
                    rule.rhs.first(),
                    Some(crate::grammar_ir::flat::Symbol::Terminal(tid))
                        if gdef.terminal_display_name(*tid) == "{"
                )
            })
            .count();
        assert_eq!(
            brace_rules_count,
            1,
            "nonnullable sequence should not synthesize duplicate brace-start alternatives"
        );
    }
