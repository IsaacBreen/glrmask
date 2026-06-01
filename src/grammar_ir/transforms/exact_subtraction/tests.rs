//! Tests for exact-subtraction lowering.

    use super::lower_exact_subtractions;
    use crate::grammar_ir::ast::{GrammarExpr, NamedGrammar, NamedRule};
use crate::grammar_ir::lower::lower;
    use crate::dump_json_schema_grammar_glrm;
    use std::{env, ffi::OsString, sync::Mutex};
    use serde_json::json;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::set_var(key, value);
            }
            Self { key, original }
        }

        fn unset(key: &'static str) -> Self {
            let original = env::var_os(key);
            unsafe {
                env::remove_var(key);
            }
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => unsafe {
                    env::set_var(self.key, value);
                },
                None => unsafe {
                    env::remove_var(self.key);
                },
            }
        }
    }

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

    fn find_rule<'a>(grammar: &'a NamedGrammar, name: &str) -> &'a NamedRule {
        grammar
            .rules
            .iter()
            .find(|rule| rule.name == name)
            .unwrap()
    }

    fn contains_exclude(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Exclude { .. } => true,
            GrammarExpr::Grouped(inner)
            | GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner) => contains_exclude(inner),
            GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
                items.iter().any(contains_exclude)
            }
            GrammarExpr::Intersect { expr, intersect } => {
                contains_exclude(expr) || contains_exclude(intersect)
            }
            GrammarExpr::RepeatRange { expr, .. } => contains_exclude(expr),
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items.iter().any(|(item, _)| contains_exclude(item)) || contains_exclude(separator)
            }
            GrammarExpr::ExprNFA(expr_nfa) => expr_nfa.symbols.iter().any(contains_exclude),
            GrammarExpr::Ref(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::Literal(_)
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    #[test]
    fn exact_subtraction_rewrites_sites_into_shared_helpers() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        literal("a"),
                        literal("b"),
                        literal("c"),
                        literal("d"),
                    ]),
                ),
                nonterminal(
                    "start",
                    GrammarExpr::Choice(vec![
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("a"),
                                literal("d"),
                            ]))),
                        ),
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("c"),
                                literal("d"),
                            ]))),
                        ),
                        subtract("A", GrammarExpr::Grouped(Box::new(literal("d")))),
                    ]),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let stats = lower_exact_subtractions(&mut grammar).unwrap();

        assert_eq!(stats.rewritten_sites, 3);
        let start_rule = find_rule(&grammar, "start");
        assert!(!contains_exclude(&start_rule.expr));
        let GrammarExpr::Choice(options) = &start_rule.expr else {
            panic!("expected rewritten start choice: {:?}", start_rule.expr);
        };
        assert!(options.iter().all(|expr| matches!(expr, GrammarExpr::Ref(name) if name.starts_with("__exact_sub_A_result"))));
        assert!(grammar.rules.iter().any(|rule| rule.name.starts_with("__exact_sub_A_part")));
        assert!(grammar.rules.iter().any(|rule| rule.name.starts_with("__exact_sub_A_tree")));
        assert!(grammar.rules.iter().any(|rule| rule.name.starts_with("__exact_sub_A_result")));
        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_partitions_alternatives_by_shared_signature() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal(
                    "A",
                    GrammarExpr::Choice(vec![
                        literal("a"),
                        literal("b"),
                        literal("c"),
                        literal("d"),
                    ]),
                ),
                nonterminal(
                    "start",
                    GrammarExpr::Choice(vec![
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("b"),
                                literal("c"),
                            ]))),
                        ),
                        subtract(
                            "A",
                            GrammarExpr::Grouped(Box::new(GrammarExpr::Choice(vec![
                                literal("b"),
                                literal("c"),
                                literal("d"),
                            ]))),
                        ),
                    ]),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        lower_exact_subtractions(&mut grammar).unwrap();

        assert!(grammar.rules.iter().any(|rule| {
            rule.name.starts_with("__exact_sub_A_part")
                && rule.expr
                    == GrammarExpr::Choice(vec![literal("b"), literal("c")])
        }));
    }

    #[test]
    fn exact_subtraction_errors_on_missing_exact_alternative() {
        let mut grammar = NamedGrammar {
            rules: vec![
                nonterminal("A", GrammarExpr::Choice(vec![literal("a"), literal("b")])),
                nonterminal(
                    "start",
                    subtract("A", GrammarExpr::Grouped(Box::new(literal("c")))),
                ),
            ],
            start: "start".to_string(),
            ignore: None,
        };

        let err = lower_exact_subtractions(&mut grammar).unwrap_err();
        assert!(format!("{err}").contains("no exact alternative"), "{err}");
    }

    #[test]
    fn exact_subtraction_matches_nonterminal_alias_body() {
        let mut grammar = NamedGrammar {
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

        let stats = lower_exact_subtractions(&mut grammar).unwrap();

        assert_eq!(stats.rewritten_sites, 1);
        assert!(!contains_exclude(&find_rule(&grammar, "start").expr));
        lower(&grammar).unwrap();
    }

    #[test]
    fn exact_subtraction_canonicalization_is_cycle_safe() {
        let mut grammar = NamedGrammar {
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

        let err = lower_exact_subtractions(&mut grammar).unwrap_err();
        assert!(format!("{err}").contains("no exact alternative"), "{err}");
    }

    #[test]
    fn exact_subtraction_json_schema_dump_uses_helpers_when_enabled() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let _lower = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS");

        let schema = json!({
            "type": "object",
            "properties": {
                "first": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"}
                    },
                    "additionalProperties": {"type": "string"}
                },
                "second": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "string"}
                    },
                    "patternProperties": {
                        "^x_": {"type": "number"}
                    },
                    "additionalProperties": {"type": "string"}
                }
            },
            "additionalProperties": false
        });

        let glrm = dump_json_schema_grammar_glrm(&schema.to_string()).unwrap();
        assert!(glrm.contains("JSON_STRING JSON_KEY_SEPARATOR - \"\\\"a\\\": \" - \"\\\"b\\\": \""), "{glrm}");
        assert!(!glrm.contains("__exact_sub_AP_SHARED_LITERAL_KEY_SET_result"), "{glrm}");
    }

    #[test]
    fn exact_subtraction_json_schema_dump_keeps_direct_subtraction_when_disabled() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let _lower = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS", "0");
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "type": "object",
            "properties": {
                "first": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "string"},
                        "b": {"type": "string"}
                    },
                    "additionalProperties": {"type": "string"}
                },
                "second": {
                    "type": "object",
                    "properties": {
                        "b": {"type": "string"}
                    },
                    "patternProperties": {
                        "^x_": {"type": "number"}
                    },
                    "additionalProperties": {"type": "string"}
                }
            },
            "additionalProperties": false
        });

        let glrm = dump_json_schema_grammar_glrm(&schema.to_string()).unwrap();
        assert!(glrm.contains("JSON_STRING JSON_KEY_SEPARATOR - \"\\\"a\\\": \" - \"\\\"b\\\": \""), "{glrm}");
        assert!(!glrm.contains("__exact_sub_AP_SHARED_LITERAL_KEY_SET_result"), "{glrm}");
    }
