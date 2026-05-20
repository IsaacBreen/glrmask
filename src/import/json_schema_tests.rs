    use super::common_outer_anchor_pattern;
    use super::decoded_regex_fullmatch_expr;
    use super::decoded_regex_matches_search;
    use super::decoded_regex_search_expr;
    use super::extract_fixed_ascii_class_concat_pattern;
    use super::extract_fixed_ascii_class_search_pattern;
    use super::integer_multiple_expr;
    use super::json_key_merge_config;
    use super::json_literal_key_merge_config;
    use super::json_literal_string_merge_config;
    use super::json_schema_uri_mode;
    use super::json_string_char_lexer_expr;
    use super::json_string_merge_config;
    use super::json_uri_merge_config;
    use super::key_colon_body_regex;
    use super::lexer_repeat;
    use super::literal_alternation_search_literals;
    use super::pattern_all_branches_anchored;
    use super::per_object_ap_keys_enabled_for_plan;
    use super::promote_literal_choices_enabled;
    use super::schema_to_named_grammar;
    use super::string_value_body_regex;
    use super::strip_branch_outer_anchors;
    use super::try_fixed_ascii_class_bounded_search_dfa;
    use super::uri_quote_merge_warning_needed;
    use super::wrap_string_value_expr_parts;
    use super::JsonSchemaUriMode;
    use super::SchemaCtx;
    use super::SharedAdditionalKeyPlan;
    use crate::automata::lexer::ast::dfa as lexer_dfa_expr;
    use crate::automata::lexer::ast::Expr as LexerExpr;
    use crate::automata::lexer::compile::build_regex;
    use crate::automata::lexer::regex::parse_regex;
    use crate::dump_json_schema_grammar_glrm;
    use crate::grammar::ast::lower;
    use crate::grammar::ast::GrammarExpr;
    use crate::grammar::glrm::to_glrm;
    use crate::GlrMaskError;
    use serde_json::{json, Value};
    use std::{env, ffi::OsString, sync::Mutex};

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

    fn dump_glrm(schema: serde_json::Value) -> String {
        let named = schema_to_named_grammar(&schema).unwrap();
        to_glrm(&named)
    }

    fn expr_contains_expr_nfa(expr: &GrammarExpr) -> bool {
        match expr {
            GrammarExpr::Grouped(inner) => expr_contains_expr_nfa(inner),
            GrammarExpr::ExprNFA(_) => true,
            GrammarExpr::Sequence(parts) | GrammarExpr::Choice(parts) => {
                parts.iter().any(expr_contains_expr_nfa)
            }
            GrammarExpr::Optional(inner)
            | GrammarExpr::Repeat(inner)
            | GrammarExpr::RepeatOne(inner) => expr_contains_expr_nfa(inner),
            GrammarExpr::RepeatRange { expr, .. } => expr_contains_expr_nfa(expr),
            GrammarExpr::Exclude { expr, exclude } | GrammarExpr::Intersect { expr, intersect: exclude } => {
                expr_contains_expr_nfa(expr) || expr_contains_expr_nfa(exclude)
            }
            GrammarExpr::SeparatedSequence { items, separator, .. } => {
                items.iter().any(|(item, _)| expr_contains_expr_nfa(item))
                    || expr_contains_expr_nfa(separator)
            }
            GrammarExpr::Ref(_)
            | GrammarExpr::Literal(_)
            | GrammarExpr::Epsilon
            | GrammarExpr::CharClass { .. }
            | GrammarExpr::RawRegex(_)
            | GrammarExpr::LexerDfa(_)
            | GrammarExpr::AnyByte => false,
        }
    }

    fn grammar_contains_expr_nfa(schema: serde_json::Value) -> bool {
        let named = schema_to_named_grammar(&schema).unwrap();
        named.rules
            .iter()
            .any(|rule| expr_contains_expr_nfa(&rule.expr))
    }

    #[test]
    fn schema_build_budget_guard_returns_schema_too_large() {
        let schema = json!({});
        let mut ctx = SchemaCtx::new(&schema);
        ctx.schema_build_limit = 1;

        assert!(ctx.increment_schema_build_count().is_ok());
        let err = ctx.increment_schema_build_count().unwrap_err();
        assert!(matches!(err, crate::GlrMaskError::GrammarParse(message) if message == "schema too large"));
    }

    #[test]
    fn literal_alternation_fast_path_matches_o48423_pattern() {
        let pattern = "Red|Blue|Yellow|Gold|Silver|Crystal|Ruby|Sapphire|Emerald|FireRed|LeafGreen|Diamond|Pearl|Platinum|HeartGold|SoulSilver|Black|White|Black 2|White 2|X|Y|Omega Ruby|Alpha Sapphire|Sun|Moon|Ultra Sun|Ultra Moon|Let's Go Pikachu|Let's Go Eevee";

        assert!(decoded_regex_matches_search(pattern, "Pokemon Red Version"));
        assert!(decoded_regex_matches_search(pattern, "Pokemon Black 2 Version"));
        assert!(decoded_regex_matches_search(pattern, "Pokemon Let's Go Eevee Save Data"));
        assert!(!decoded_regex_matches_search(pattern, "Pokemon Scarlet Version"));
    }

    #[test]
    fn literal_alternation_fast_path_rejects_non_literal_patterns() {
        for pattern in ["^foo$", "foo.*bar", "[abc]", r"foo\d+", "(foo|bar)"] {
            assert!(literal_alternation_search_literals(pattern).is_none(), "pattern should fall back: {pattern}");
        }

        assert!(decoded_regex_matches_search("^foo$", "foo"));
        assert!(decoded_regex_matches_search("foo.*bar", "xxfoobazbaryy"));
        assert!(decoded_regex_matches_search("[abc]", "zzayy"));
        assert!(decoded_regex_matches_search(r"foo\d+", "prefix foo123 suffix"));
        assert!(decoded_regex_matches_search("(foo|bar)", "prefix bar suffix"));
    }

    fn fixed_ascii_bounded_search_accepts(
        pattern: &str,
        min_len: usize,
        max_len: usize,
        body: &[u8],
    ) -> bool {
        let dfa = try_fixed_ascii_class_bounded_search_dfa(pattern, min_len, max_len)
            .expect("strict fixed-ascii extractor should accept pattern");
        let regex = build_regex(&[lexer_dfa_expr(dfa)]);
        let mut state = 0u32;
        for &byte in body {
            let Some(next) = regex.step(state, byte) else {
                return false;
            };
            state = next;
        }
        regex.dfa.finalizers(state).contains(0)
    }

    #[test]
    fn fixed_ascii_class_concat_extractor_accepts_platform_id_pattern() {
        let pattern = "[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]";
        let classes = extract_fixed_ascii_class_concat_pattern(pattern).expect("pattern should extract");
        assert_eq!(classes.len(), 11);
        assert!(classes[0].contains(b'R'));
        assert!(classes[0].contains(b'D'));
        assert!(classes[1].contains(b'0'));
        assert!(classes[1].contains(b'9'));
        assert!(classes[10].contains(b'K'));
        assert!(classes[10].contains(b'G'));
    }

    #[test]
    fn fixed_ascii_class_search_extractor_accepts_repeated_class_pattern() {
        let classes = extract_fixed_ascii_class_search_pattern("[A-z]+")
            .expect("repeated class search should extract");
        assert_eq!(classes.len(), 1);
        assert!(classes[0].contains(b'A'));
        assert!(classes[0].contains(b'z'));

        let classes = extract_fixed_ascii_class_search_pattern("[A-z]{3,64}")
            .expect("bounded repeated class search should extract");
        assert_eq!(classes.len(), 3);
    }

    #[test]
    fn fixed_ascii_class_concat_extractor_rejects_unsupported_shapes() {
        for pattern in [
            "^[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]$",
            "[RD][0-9]{9}[KMRASPDHEG]",
            "[RD][0-9][0-9]|foo",
            "[\\d][0-9]",
            "é[0-9]",
        ] {
            assert!(extract_fixed_ascii_class_concat_pattern(pattern).is_none(), "pattern should reject: {pattern}");
        }
    }

    #[test]
    fn fixed_ascii_bounded_search_accepts_platform_id_examples() {
        let pattern = "[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]";
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"R123456789K"));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"D000000000M"));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"xxR123456789Kyy"));
    }

    #[test]
    fn fixed_ascii_bounded_search_accepts_unanchored_repeated_class_examples() {
        let pattern = "[A-z]+";
        assert!(fixed_ascii_bounded_search_accepts(pattern, 3, 64, b"abc"));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 3, 64, b"12A"));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 3, 64, b"12\\\\"));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 3, 64, b"123"));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 3, 64, b"12\\\""));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 3, 64, b"ab"));
    }

    #[test]
    fn fixed_ascii_bounded_search_rejects_invalid_platform_id_examples() {
        let pattern = "[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]";
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"X123456789K"));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"R12345678K"));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"R123456789Z"));
    }

    #[test]
    fn fixed_ascii_bounded_search_honors_prefix_suffix_and_length_limit() {
        let pattern = "[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]";
        let body = format!("{}R123456789K", "A".repeat(25));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, body.as_bytes()));

        let body = format!("R123456789K{}", "A".repeat(25));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, body.as_bytes()));

        let body = format!("{}R123456789K{}", "A".repeat(10), "B".repeat(15));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, body.as_bytes()));

        let body = format!("{}R123456789K", "A".repeat(26));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, body.as_bytes()));
    }

    #[test]
    fn fixed_ascii_bounded_search_treats_named_escapes_as_nonmatching_chars() {
        let pattern = "[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]";
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"\\nR123456789K"));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"\\\\R123456789K"));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"\\\"R123456789K"));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"\\u0052123456789K"));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, b"R12345\\u00363789K"));
    }

    #[test]
    fn fixed_ascii_bounded_search_counts_direct_utf8_tail_chars() {
        let pattern = "[RD][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][KMRASPDHEG]";
        let body = format!("{}éR123456789K", "A".repeat(24));
        assert!(fixed_ascii_bounded_search_accepts(pattern, 0, 36, body.as_bytes()));

        let body = format!("{}éR123456789K", "A".repeat(25));
        assert!(!fixed_ascii_bounded_search_accepts(pattern, 0, 36, body.as_bytes()));
    }

    #[test]
    fn fixed_ascii_bounded_search_dfa_matches_generic_bounded_construction() {
        let pattern = "[AB][0-1]";
        let min_len = 0;
        let max_len = 4;
        let direct = build_regex(&[lexer_dfa_expr(
            try_fixed_ascii_class_bounded_search_dfa(pattern, min_len, max_len)
                .expect("strict fixed-ascii DFA should build"),
        )]);
        let generic_expr = LexerExpr::Intersect {
            expr: Box::new(decoded_regex_search_expr(pattern, Some(max_len))),
            intersect: Box::new(lexer_repeat(
                json_string_char_lexer_expr(),
                min_len,
                Some(max_len),
            )),
        };
        let generic = build_regex(&[generic_expr]);

        let tokens: Vec<Vec<u8>> = vec![
            b"A".to_vec(),
            b"B".to_vec(),
            b"0".to_vec(),
            b"1".to_vec(),
            b"X".to_vec(),
            b"\\n".to_vec(),
            b"\\\"".to_vec(),
            b"\\\\".to_vec(),
            "é".as_bytes().to_vec(),
        ];

        let accepts = |regex: &crate::automata::lexer::compile::Regex, body: &[u8]| {
            let mut state = 0u32;
            for &byte in body {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        let mut bodies = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = bodies.clone();
            for body in &bodies {
                for token in &tokens {
                    let mut extended = body.clone();
                    extended.extend_from_slice(token);
                    next.push(extended);
                }
            }
            bodies = next;
        }

        for body in bodies {
            assert_eq!(
                accepts(&direct, &body),
                accepts(&generic, &body),
                "body mismatch for {:?}",
                String::from_utf8_lossy(&body),
            );
        }
    }

    #[test]
    fn nested_if_then_reports_unimplemented_keys() {
        let schema = json!({
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "if": {
                        "properties": {
                            "kind": { "const": "x" }
                        }
                    },
                    "then": {
                        "required": ["value"]
                    }
                }
            }
        });

        let err = schema_to_named_grammar(&schema).unwrap_err();
        assert!(matches!(err, crate::GlrMaskError::GrammarParse(message) if message == "Unimplemented keys: [\"if\", \"then\"]"));
    }

    #[test]
    fn local_ref_if_then_reports_unimplemented_keys() {
        let schema = json!({
            "$ref": "#/$defs/branch",
            "$defs": {
                "branch": {
                    "type": "object",
                    "if": {
                        "properties": {
                            "kind": { "const": "x" }
                        }
                    },
                    "then": {
                        "required": ["value"]
                    }
                }
            }
        });

        let err = schema_to_named_grammar(&schema).unwrap_err();
        assert!(matches!(err, crate::GlrMaskError::GrammarParse(message) if message == "Unimplemented keys: [\"if\", \"then\"]"));
    }

    fn find_rule_line_with_prefix<'a>(glrm: &'a str, rule_prefix: &str) -> &'a str {
        glrm.lines()
            .find(|line| line.starts_with(&format!("t {rule_prefix}")))
            .unwrap_or_else(|| panic!("missing rule prefix {rule_prefix} in grammar:\n{glrm}"))
    }

    #[test]
    fn anyof_object_variants_lower_through_expr_nfa() {
        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "left"},
                        "a": {"type": "string"}
                    },
                    "required": ["kind"],
                    "additionalProperties": {"type": "number"}
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "right"},
                        "b": {"type": "boolean"}
                    },
                    "required": ["kind"],
                    "additionalProperties": {"type": "number"}
                }
            ]
        });

        assert!(grammar_contains_expr_nfa(schema));
    }

    #[test]
    fn anyof_object_expr_nfa_open_variants_lower_without_nested_excludes() {
        let schema = json!({
            "definitions": {
                "organization": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "links": {"type": "array", "items": {"type": "string"}},
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "sort": {"type": "string"}
                    },
                    "required": ["name"]
                },
                "person": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "email": {"type": "string"},
                        "links": {"type": "array", "items": {"type": "string"}},
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "sort": {"type": "string"},
                        "thumbnail": {"type": "string"}
                    },
                    "required": ["name"]
                }
            },
            "anyOf": [
                {"$ref": "#/definitions/person"},
                {"$ref": "#/definitions/organization"}
            ]
        });

        let named = schema_to_named_grammar(&schema).unwrap();
        assert!(
            !named.rules.iter().any(|rule| expr_contains_expr_nfa(&rule.expr)),
            "{named:#?}"
        );
        lower(&named).unwrap();
    }

    #[test]
    fn anyof_open_object_dominance_does_not_reduce_constrained_additional_properties() {
        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "thumbnail": {"type": "string"}
                    },
                    "required": ["name"],
                    "additionalProperties": {"type": "number"}
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"],
                    "additionalProperties": {"type": "number"}
                }
            ]
        });

        let named = schema_to_named_grammar(&schema).unwrap();
        assert!(named.rules.iter().any(|rule| expr_contains_expr_nfa(&rule.expr)));
        lower(&named).unwrap();
    }

    #[test]
    fn anyof_object_expr_nfa_fixed_keys_preserve_key_quote_defaults() {
        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "thumbnail": {"type": "string"}
                    },
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "additionalProperties": false
                }
            ]
        });

        let glrm = dump_glrm(schema);
        assert!(glrm.contains("fa obj_anyof_fa_0_body"), "{glrm}");
        assert!(
            glrm.contains("-- \"\\\"\" \"thumbnail\\\"\" \": \" -->"),
            "{glrm}"
        );
        assert!(
            !glrm.contains("-- \"\\\"thumbnail\\\"\" \": \" -->"),
            "{glrm}"
        );
    }

    #[test]
    fn anyof_object_expr_nfa_uses_one_ap_terminal_with_sibling_literals() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "thumbnail": {"type": "string"}
                    },
                    "additionalProperties": {"type": "number"}
                },
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "additionalProperties": {"type": "number"}
                }
            ]
        });

        let glrm = dump_glrm(schema);
        assert!(glrm.contains("t AP_SHARED_KEY ::="), "{glrm}");
        assert!(
            glrm.contains("-- AP_SHARED_KEY -->")
                && glrm.contains("-- AP_SHARED_LITERAL_KEY_SET - (\"name\\\"\") -->"),
            "{glrm}"
        );
        assert!(!glrm.contains("obj_anyof_fa_0_ap_key_v1"), "{glrm}");
        assert_eq!(glrm.matches("t AP_SHARED_KEY ::=").count(), 1, "{glrm}");
    }

    #[test]
    fn anyof_object_expr_nfa_person_org_shape_uses_shared_ap_key_paths() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "name": {"type": "string"},
                        "sameAs": {
                            "items": {"type": "string"},
                            "type": "array"
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": {"type": "number"}
                },
                {
                    "type": "object",
                    "properties": {
                        "additionalName": {"type": "string"},
                        "email": {"type": "string"},
                        "familyName": {"type": "string"},
                        "givenName": {"type": "string"},
                        "name": {"type": "string"},
                        "sameAs": {
                            "items": {"type": "string"},
                            "type": "array"
                        }
                    },
                    "required": ["name"],
                    "additionalProperties": {"type": "number"}
                }
            ]
        });

        let glrm = dump_glrm(schema.clone());
        assert!(glrm.contains("fa obj_anyof_fa_0_body"), "{glrm}");
    assert!(glrm.contains("-- AP_SHARED_KEY -->"), "{glrm}");
        assert!(!glrm.contains("obj_anyof_fa_0_ap_key_v0"), "{glrm}");
        assert!(!glrm.contains("obj_anyof_fa_0_ap_key_v1"), "{glrm}");

        let named = schema_to_named_grammar(&schema).unwrap();
        lower(&named).unwrap();
    }

    #[test]
    fn repeated_anyof_object_expr_nfa_sites_share_generated_fa() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "definitions": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "thumbnail": {"type": "string"}
                    },
                    "additionalProperties": {"type": "number"}
                },
                "organization": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "additionalProperties": {"type": "number"}
                }
            },
            "type": "object",
            "properties": {
                "attributions": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"$ref": "#/definitions/person"},
                            {"$ref": "#/definitions/organization"}
                        ]
                    }
                },
                "copyright_holder": {
                    "anyOf": [
                        {"$ref": "#/definitions/person"},
                        {"$ref": "#/definitions/organization"}
                    ]
                }
            }
        });

        let glrm = dump_glrm(schema);
        assert_eq!(glrm.matches("fa obj_anyof_fa_").count(), 1, "{glrm}");
        assert_eq!(glrm.matches("t AP_SHARED_KEY ::=").count(), 1, "{glrm}");
        assert!(glrm.contains("nt obj_anyof_fa_"), "{glrm}");
    }

    #[test]
    fn coerced_oneof_object_variants_use_expr_nfa_like_anyof() {
        let schema = json!({
            "x-guidance": {"coerce_one_of": true},
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "left"},
                        "a": {"type": "string"}
                    },
                    "required": ["kind"],
                    "additionalProperties": {"type": "number"}
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "right"},
                        "b": {"type": "boolean"}
                    },
                    "required": ["kind"],
                    "additionalProperties": {"type": "number"}
                }
            ]
        });

        let named = schema_to_named_grammar(&schema).unwrap();
        assert!(named.rules.iter().any(|rule| expr_contains_expr_nfa(&rule.expr)));
        lower(&named).unwrap();
    }

    #[test]
    fn anyof_object_expr_nfa_supports_pattern_properties_tail() {
        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "left"},
                        "a": {"type": "string"}
                    },
                    "patternProperties": {
                        "^x_": {"type": "string"}
                    },
                    "required": ["kind"],
                    "additionalProperties": {"type": "number"}
                },
                {
                    "type": "object",
                    "properties": {
                        "kind": {"const": "right"},
                        "b": {"type": "boolean"}
                    },
                    "patternProperties": {
                        "^y_": {"type": "boolean"}
                    },
                    "required": ["kind"],
                    "additionalProperties": {"type": "number"}
                }
            ]
        });

        let named = schema_to_named_grammar(&schema).unwrap();
        assert!(named.rules.iter().any(|rule| expr_contains_expr_nfa(&rule.expr)));
        lower(&named).unwrap();
    }

    #[test]
    fn match_all_pattern_properties_use_left_recursive_pair_list() {
        let schema = json!({
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            }
        });

        let glrm = dump_glrm(schema.clone());
        assert!(glrm.contains("nt obj_pairs_"), "{glrm}");
        assert!(glrm.contains("_list ::= obj_pairs_"), "{glrm}");
        let start_line = glrm
            .lines()
            .find(|line| line.starts_with("nt start ::="))
            .unwrap();
        assert!(!start_line.contains("\", \" ~ ("), "{glrm}");
        lower(&schema_to_named_grammar(&schema).unwrap()).unwrap();
    }

    #[test]
    fn anyof_object_expr_nfa_keeps_non_object_options() {
        let schema = json!({
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": false
                },
                {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "additionalProperties": false
                },
                {"type": "string"}
            ]
        });

        let glrm = dump_glrm(schema);
        assert!(glrm.contains("fa obj_anyof_fa_0_body"), "{glrm}");
        assert!(glrm.contains("json_string"), "{glrm}");
    }

    #[test]
    fn implicit_object_anyof_reduction_requires_object_only_keywords() {
        let object_only = json!({
            "properties": {"a": {"type": "string"}},
            "additionalProperties": false,
            "description": "annotation-only"
        });
        assert!(SchemaCtx::schema_is_object_like_for_anyof_reduction(
            object_only.as_object().unwrap()
        ));

        let constrained_non_objects = json!({
            "properties": {"a": {"type": "string"}},
            "additionalProperties": false,
            "enum": [{"a": "x"}]
        });
        assert!(!SchemaCtx::schema_is_object_like_for_anyof_reduction(
            constrained_non_objects.as_object().unwrap()
        ));
    }

    #[test]
    fn duplicate_self_ref_anyof_closed_object_compiles() {
        let schema = json!({
            "$schema": "http://json-schema.org/draft-04/schema#",
            "definitions": {
                "Monster": {
                    "type": "object",
                    "properties": {
                        "any_ambiguous": {
                            "anyOf": [
                                { "$ref": "#/definitions/Monster" },
                                { "$ref": "#/definitions/Monster" }
                            ]
                        }
                    },
                    "additionalProperties": false
                }
            },
            "$ref": "#/definitions/Monster"
        });

        assert!(schema_to_named_grammar(&schema).is_ok());
    }

    #[test]
    fn enum_string_values_with_same_membership_share_terminal() {
        let schema = json!({
            "type": "object",
            "properties": {
                "left": { "enum": ["a", "b"] },
                "right": { "enum": ["a", "b"] }
            },
            "required": ["left", "right"],
            "additionalProperties": false
        });

        let glrm = dump_glrm(schema);

        assert!(glrm.contains("t JSON_ENUM_STRING_0 ::= \"a\\\"\" | \"b\\\"\";"), "{glrm}");
        assert_eq!(glrm.matches("t JSON_ENUM_STRING_").count(), 1, "{glrm}");
        assert!(!glrm.contains("\"a\\\"\" | \"b\\\"\" |"), "{glrm}");
    }

    #[test]
    fn shared_additional_properties_key_exclusions_are_on_by_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "type": "object",
            "properties": {
                "left": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": {"type": "string"}
                },
                "right": {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "additionalProperties": {"type": "string"}
                }
            },
            "additionalProperties": false
        });

        let glrm = dump_glrm(schema);
        assert!(glrm.contains("t AP_SHARED_KEY ::= "), "{glrm}");
        assert!(glrm.contains("AP_SHARED_LITERAL_KEY_0 ::= \"a\\\"\""), "{glrm}");
        assert!(glrm.contains("AP_SHARED_LITERAL_KEY_1 ::= \"b\\\"\""), "{glrm}");
        assert!(glrm.contains("AP_SHARED_LITERAL_KEY_2 ::= \"left\\\"\""), "{glrm}");
        assert!(glrm.contains("AP_SHARED_LITERAL_KEY_3 ::= \"right\\\"\""), "{glrm}");
    }

    #[test]
    fn shared_additional_properties_key_exclusions_are_ubiquitous() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _shared = EnvVarGuard::set("GLRMASK_AP_SHARED_EXCLUSIONS", "0");
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let schema = json!({
            "type": "object",
            "properties": {
                "left": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "additionalProperties": {"type": "string"}
                },
                "right": {
                    "type": "object",
                    "properties": {"b": {"type": "string"}},
                    "additionalProperties": {"type": "string"}
                }
            },
            "additionalProperties": false
        });

        let glrm = dump_glrm(schema);
        assert!(glrm.contains("t AP_SHARED_KEY ::= "), "{glrm}");
    }

    #[test]
    fn shared_additional_properties_key_excludes_global_patterns_and_adds_back_siblings() {
        let _lock = ENV_LOCK.lock().unwrap();
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

        let glrm = dump_glrm(schema);
        assert!(glrm.contains("t AP_SHARED_KEY ::="), "{glrm}");
        assert!(glrm.contains("AP_SHARED_PATTERN_KEY_0"), "{glrm}");
        assert!(glrm.contains(" - AP_SHARED_PATTERN_KEY_0"), "{glrm}");
        assert!(
            glrm.contains("AP_SHARED_LITERAL_KEY_SET - (\"a\\\"\" | \"b\\\"\")"),
            "{glrm}"
        );
        assert!(
            glrm.contains(
                "ap_shared_pattern_key_filtered_6 ::= AP_SHARED_PATTERN_KEY_0_5 - AP_SHARED_LITERAL_KEY_0 - AP_SHARED_LITERAL_KEY_1"
            ),
            "{glrm}"
        );
        assert!(!glrm.contains("AP_SHARED_KEY | \"a\\\"\""), "{glrm}");
    }

    #[test]
    fn shared_additional_properties_key_exclusions_are_not_thresholded() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _promote = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES", "0");

        let mut properties = serde_json::Map::new();
        properties.insert(
            "target".into(),
            json!({
                "type": "object",
                "properties": {"target_key": {"type": "string"}},
                "additionalProperties": {"type": "string"}
            }),
        );

        for index in 0..40 {
            properties.insert(
                format!("branch_{index}"),
                json!({
                    "type": "object",
                    "properties": {
                        format!("shared_key_{index}"): {"type": "string"}
                    },
                    "additionalProperties": {"type": "string"}
                }),
            );
        }

        let schema = serde_json::Value::Object(serde_json::Map::from_iter([
            ("type".into(), json!("object")),
            ("properties".into(), serde_json::Value::Object(properties)),
            ("additionalProperties".into(), json!(false)),
        ]));

        let _shared = EnvVarGuard::unset("GLRMASK_AP_SHARED_EXCLUSIONS");
        let default_glrm = dump_glrm(schema.clone());
        assert!(default_glrm.contains("t AP_SHARED_KEY ::= "), "{default_glrm}");

        let _shared = EnvVarGuard::set("GLRMASK_AP_SHARED_EXCLUSIONS", "1");
        let forced_glrm = dump_glrm(schema);
        assert!(forced_glrm.contains("t AP_SHARED_KEY ::= "), "{forced_glrm}");
    }

    #[test]
    fn per_object_ap_keys_override_forces_true_for_small_plan() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _enabled = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEYS", "1");
        let _threshold = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEY_THRESHOLD", "1000");

        let plan = SharedAdditionalKeyPlan::default();

        assert!(per_object_ap_keys_enabled_for_plan(&plan));
    }

    #[test]
    fn per_object_ap_keys_override_forces_false_for_large_plan() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _enabled = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEYS", "0");
        let _threshold = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEY_THRESHOLD", "1");

        let plan = SharedAdditionalKeyPlan {
            literal_keys: (0..32).map(|idx| format!("k{idx}")).collect(),
            pattern_keys: [String::from("^p")].into_iter().collect(),
            ..Default::default()
        };

        assert!(!per_object_ap_keys_enabled_for_plan(&plan));
    }

    #[test]
    fn per_object_ap_keys_auto_enables_at_threshold() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _enabled = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEYS");
        let _threshold = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEY_THRESHOLD", "17");

        let plan = SharedAdditionalKeyPlan {
            literal_keys: (0..9).map(|idx| format!("k{idx}")).collect(),
            pattern_keys: [String::from("^p")].into_iter().collect(),
            ..Default::default()
        };

        assert!(per_object_ap_keys_enabled_for_plan(&plan));
    }

    #[test]
    fn per_object_ap_keys_auto_stays_off_below_threshold() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _enabled = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEYS");
        let _threshold = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PER_OBJECT_AP_KEY_THRESHOLD", "18");

        let plan = SharedAdditionalKeyPlan {
            literal_keys: (0..9).map(|idx| format!("k{idx}")).collect(),
            pattern_keys: [String::from("^p")].into_iter().collect(),
            ..Default::default()
        };

        assert!(!per_object_ap_keys_enabled_for_plan(&plan));
    }

    #[test]
    fn shared_additional_properties_promote_literal_choices_by_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _promote = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_PROMOTE_LITERAL_CHOICES");

        assert!(promote_literal_choices_enabled());
    }

    #[test]
    fn shared_additional_properties_can_disable_literal_choice_promotion() {
        let _lock = ENV_LOCK.lock().unwrap();
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
        assert!(!glrm.contains("__GLRMASK_LITERAL_CHOICE_"), "{glrm}");
        assert!(glrm.contains("nt AP_SHARED_LITERAL_KEY_SET ::= "), "{glrm}");
    }

    #[test]
    fn enum_number_values_are_grouped_separately_from_strings() {
        let schema = json!({
            "anyOf": [
                { "enum": [1, 2, "1", "2"] },
                { "enum": [1, 2, "1", "2"] }
            ]
        });

        let glrm = dump_glrm(schema);

        assert!(glrm.contains("t JSON_ENUM_NUMBER_0 ::= \"1\" | \"2\";"), "{glrm}");
        assert!(glrm.contains("t JSON_ENUM_STRING_0 ::= \"1\\\"\" | \"2\\\"\";"), "{glrm}");
    }

    #[test]
    fn unconstrained_string_values_share_literal_and_pattern_exclusions() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _compat = EnvVarGuard::unset("GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT");
        let _shared = EnvVarGuard::set("GLRMASK_GLOBAL_SHARED_STRING_VALUE_EXCLUSIONS", "1");
        let _limit = EnvVarGuard::unset("GLRMASK_SHARED_STRING_VALUE_EXCLUSION_LIMIT");

        let schema = json!({
            "type": "object",
            "properties": {
                "plain": {"type": "string"},
                "literal": {"const": "ready"},
                "choice": {"enum": ["open", "closed"]},
                "patterned": {"type": "string", "pattern": "^x_"}
            },
            "additionalProperties": false
        });

        let glrm = dump_glrm(schema);

        assert!(glrm.contains("t STRING_SHARED_VALUE ::="), "{glrm}");
        assert!(glrm.contains("STRING_SHARED_LITERAL_VALUE"), "{glrm}");
        assert!(glrm.contains("STRING_SHARED_PATTERN_VALUE"), "{glrm}");
        assert!(glrm.contains(" - \"ready\""), "{glrm}");
        assert!(glrm.contains(" - STRING_SHARED_PATTERN_VALUE"), "{glrm}");
        assert!(
            glrm.contains("STRING_SHARED_VALUE | \"ready\\\"\"")
                && glrm.contains("STRING_SHARED_PATTERN_VALUE"),
            "{glrm}"
        );
    }

    #[test]
    fn shared_string_value_exclusion_limit_caps_global_plan() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _compat = EnvVarGuard::unset("GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT");
        let _shared = EnvVarGuard::set("GLRMASK_GLOBAL_SHARED_STRING_VALUE_EXCLUSIONS", "1");
        let _limit = EnvVarGuard::set("GLRMASK_SHARED_STRING_VALUE_EXCLUSION_LIMIT", "2");

        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"const": "a"},
                {"const": "b"},
                {"const": "c"}
            ]
        });

        let glrm = dump_glrm(schema);

        assert!(!glrm.contains("t STRING_SHARED_VALUE ::="), "{glrm}");
    }

    #[test]
    fn abdcffb6b_string_value_compat_restores_capped_global_mode() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _compat = EnvVarGuard::set("GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT", "abdcffb6b");
        let _shared = EnvVarGuard::unset("GLRMASK_GLOBAL_SHARED_STRING_VALUE_EXCLUSIONS");
        let _limit = EnvVarGuard::unset("GLRMASK_SHARED_STRING_VALUE_EXCLUSION_LIMIT");

        let small_schema = json!({
            "anyOf": [
                {"type": "string"},
                {"const": "ready"}
            ]
        });
        let small_glrm = dump_glrm(small_schema);
        assert!(small_glrm.contains("t STRING_SHARED_VALUE ::="), "{small_glrm}");
        assert!(
            !small_glrm.contains("t STRING_ANYOF_VALUE_"),
            "{small_glrm}"
        );

        let large_values = (0..33)
            .map(|idx| json!({"const": format!("value-{idx}")}))
            .collect::<Vec<_>>();
        let mut options = Vec::with_capacity(1 + large_values.len());
        options.push(json!({"type": "string"}));
        options.extend(large_values);
        let large_schema = json!({"anyOf": options});
        let large_glrm = dump_glrm(large_schema);

        assert!(!large_glrm.contains("t STRING_SHARED_VALUE ::="), "{large_glrm}");
        assert!(!large_glrm.contains("t STRING_ANYOF_VALUE_"), "{large_glrm}");
    }

    #[test]
    fn shared_string_value_exclusions_are_enabled_by_default_without_a_threshold() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _compat = EnvVarGuard::unset("GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT");
        let _shared = EnvVarGuard::unset("GLRMASK_SHARED_STRING_VALUE_EXCLUSIONS");

        let values = (0..40)
            .map(|idx| Value::String(format!("value-{idx}")))
            .collect::<Vec<_>>();
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"enum": values}
            ]
        });

        let glrm = dump_glrm(schema);

        assert!(glrm.contains("t STRING_ANYOF_VALUE_"), "{glrm}");
        assert!(glrm.contains(" - \"value-0\""), "{glrm}");
    }

    #[test]
    fn shared_string_value_exclusions_can_be_disabled() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _compat = EnvVarGuard::unset("GLRMASK_STRING_VALUE_EXCLUSIONS_COMPAT");
        let _shared = EnvVarGuard::set("GLRMASK_SHARED_STRING_VALUE_EXCLUSIONS", "0");

        let values = (0..40)
            .map(|idx| Value::String(format!("value-{idx}")))
            .collect::<Vec<_>>();
        let schema = json!({
            "anyOf": [
                {"type": "string"},
                {"enum": values}
            ]
        });

        let glrm = dump_glrm(schema);

        assert!(!glrm.contains("t STRING_ANYOF_VALUE_"), "{glrm}");
    }

    #[test]
    fn json_string_merge_defaults_preserve_split_open_and_merged_close() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _string_open = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_OPEN");
        let _string_close = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_CLOSE");
        let _literal_open = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_STRING_MERGE_OPEN");
        let _literal_close = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_STRING_MERGE_CLOSE");
        let _pattern_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_OPEN");
        let _pattern_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_CLOSE");

        let default_cfg = json_string_merge_config(false);
        let literal_cfg = json_literal_string_merge_config();
        let pattern_cfg = json_string_merge_config(true);
        assert_eq!(default_cfg.merge_open, false);
        assert_eq!(default_cfg.merge_close, true);
        assert_eq!(literal_cfg, default_cfg);
        assert_eq!(pattern_cfg.merge_open, true);
        assert_eq!(pattern_cfg.merge_close, false);
        assert_eq!(string_value_body_regex("a+", false), "(?:a+)\"");
        assert_eq!(string_value_body_regex("a+", true), "\"(?:a+)");
    }

    #[test]
    fn json_key_merge_defaults_preserve_old_split_open_and_merged_close() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _key_open = EnvVarGuard::unset("GLRMASK_JSON_KEY_MERGE_OPEN");
        let _key_close = EnvVarGuard::unset("GLRMASK_JSON_KEY_MERGE_CLOSE");
        let _literal_key_open = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_KEY_MERGE_OPEN");
        let _literal_key_close = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_KEY_MERGE_CLOSE");
        let _pattern_key_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_KEY_MERGE_OPEN");
        let _pattern_key_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_KEY_MERGE_CLOSE");

        let default_cfg = json_key_merge_config(false);
        let literal_cfg = json_literal_key_merge_config();
        let pattern_cfg = json_key_merge_config(true);
        assert_eq!(default_cfg.merge_open, false);
        assert_eq!(default_cfg.merge_close, true);
        assert_eq!(literal_cfg, default_cfg);
        assert_eq!(pattern_cfg, default_cfg);
        assert_eq!(key_colon_body_regex("a+"), "(?:a+)\"");
    }

    #[test]
    fn bounded_non_pattern_split_keeps_open_quote_separate_by_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _chunk = EnvVarGuard::set("GLRMASK_STRING_REPEAT_CHUNK", "4");

        let glrm = dump_glrm(json!({
            "type": "string",
            "minLength": 5,
            "maxLength": 5,
        }));

        assert!(glrm.contains("json_string_bounded_split"), "{glrm}");
        assert!(glrm.contains("::= \"\\\"\" ("), "{glrm}");
        assert!(glrm.contains("JSON_STRING_CHAR_EXACT_CLOSE"), "{glrm}");
        assert!(!glrm.contains("JSON_STRING_CHAR_EXACT_OPEN"), "{glrm}");
    }

    #[test]
    fn bounded_simple_pattern_merges_open_quote_not_close_by_default() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _string_open = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_OPEN");
        let _string_close = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_CLOSE");
        let _pattern_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_OPEN");
        let _pattern_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_CLOSE");

        let glrm = dump_glrm(json!({
            "type": "string",
            "minLength": 1,
            "maxLength": 1024,
            "pattern": "^[A-Za-z0-9_-]+$",
        }));

        let pattern_line = glrm
            .lines()
            .find(|line| line.starts_with("t JSON_STRING_BOUNDED_PATTERN_"))
            .unwrap_or_else(|| panic!("{glrm}"));
        let start_line = glrm
            .lines()
            .find(|line| line.starts_with("nt start ::="))
            .unwrap_or_else(|| panic!("{glrm}"));

        assert!(
            pattern_line.contains("::= \"\\\"\" (JSON_STRING_PATTERN_CHAR_"),
            "{glrm}"
        );
        assert!(!pattern_line.ends_with("\"\\\"\";"), "{glrm}");
        assert!(
            start_line.contains("JSON_STRING_BOUNDED_PATTERN_") && start_line.ends_with("\"\\\"\";"),
            "{glrm}"
        );
    }

    #[test]
    fn outer_parenthesized_anchors_count_as_branch_anchors() {
        assert_eq!(
            strip_branch_outer_anchors(r"(^(?:\S+\s+){0,19}\S+$)"),
            (true, true, r"(?:\S+\s+){0,19}\S+")
        );
        assert!(pattern_all_branches_anchored(r"^$|(^(?:\S+\s+){0,19}\S+$)"));
        assert_eq!(
            common_outer_anchor_pattern(r"^$|(^(?:\S+\s+){0,19}\S+$)"),
            Some((true, true, r"(?:|(?:\S+\s+){0,19}\S+)".to_string()))
        );
    }

    #[test]
    fn integer_power_of_ten_multiple_uses_compact_expression() {
        let regex = build_regex(&[integer_multiple_expr(10, true)]);
        let accepts = |text: &str| {
            let mut state = 0u32;
            for &byte in text.as_bytes() {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        assert!(accepts("0"));
        assert!(accepts("10"));
        assert!(accepts("-120.000"));
        assert!(!accepts("12"));
        assert!(!accepts("10.5"));
        assert!(regex.num_states() < 20);
    }

    #[test]
    fn decoded_regex_fullmatch_expr_preserves_multi_digit_separator_language() {
        let regex = build_regex(&[decoded_regex_fullmatch_expr(r"[0-9]+x[0-9]+")]);
        let accepts = |text: &str| {
            let mut state = 0u32;
            for &byte in text.as_bytes() {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        assert!(accepts("1x2"));
        assert!(accepts("1920x1080"));
        assert!(!accepts("1x23x"));
        assert!(!accepts("1920x108x0"));
    }

    #[test]
    fn decoded_regex_fullmatch_expr_accepts_utf8_under_negated_char_class() {
        let regex = build_regex(&[decoded_regex_fullmatch_expr(r"^[^&]*$")]);
        let accepts_bytes = |bytes: &[u8]| {
            let mut state = 0u32;
            for &byte in bytes {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };
        let accepts = |text: &str| accepts_bytes(text.as_bytes());

        assert!(accepts("и"));
        assert!(accepts("ивання"));
        assert!(accepts(" веществ"));
        assert!(accepts(" balık"));
        assert!(accepts("홈"));
        assert!(accepts("工程"));
        assert!(!accepts_bytes(b"\x80"));
        assert!(!accepts_bytes(b"\xbb"));
        assert!(!accepts_bytes(b"&\xbb"));
        assert!(!accepts("a&b"));
    }

    #[test]
    fn parse_regex_utf8_negated_ascii_class_accepts_utf8() {
        let regex = build_regex(&[parse_regex(r"[^&]*", true)]);
        let accepts = |text: &str| {
            let mut state = 0u32;
            for &byte in text.as_bytes() {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        assert!(accepts("и"));
        assert!(accepts("ивання"));
        assert!(accepts("홈"));
        assert!(!accepts("a&b"));
    }

    #[test]
    fn decoded_regex_fullmatch_expr_accepts_positive_high_byte_literal() {
        let regex = build_regex(&[decoded_regex_fullmatch_expr("ивання")]);
        let accepts = |text: &str| {
            let mut state = 0u32;
            for &byte in text.as_bytes() {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        assert!(accepts("ивання"));
        assert!(!accepts("Ð¸Ð²Ð°Ð½Ð½Ñ"));
        assert!(!accepts("ivannya"));
    }

    #[test]
    fn decoded_regex_fullmatch_expr_accepts_positive_high_byte_class() {
        let regex = build_regex(&[decoded_regex_fullmatch_expr(r"^[\xD0-\xD1][\x80-\xBF]$")]);
        let accepts = |text: &str| {
            let mut state = 0u32;
            for &byte in text.as_bytes() {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        assert!(accepts("и"));
        assert!(!accepts("ивання"));
        assert!(!accepts("hello"));
        assert!(!accepts("&"));
    }

    #[test]
    fn decoded_regex_fullmatch_expr_dot_rejects_lone_continuation_byte() {
        let regex = build_regex(&[decoded_regex_fullmatch_expr(r"^.*\.md$")]);
        let accepts_bytes = |bytes: &[u8]| {
            let mut state = 0u32;
            for &byte in bytes {
                let Some(next) = regex.step(state, byte) else {
                    return false;
                };
                state = next;
            }
            regex.dfa.finalizers(state).contains(0)
        };

        assert!(accepts_bytes("и.md".as_bytes()));
        assert!(accepts_bytes("Task description.md".as_bytes()));
        assert!(!accepts_bytes(b"\xbb.md"));
        assert!(!accepts_bytes(b"Task description\xbb.md"));
    }

    #[test]
    fn string_merge_env_layers_resolve_pattern_and_literal_independently() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _string_open = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_OPEN");
        let _string_close = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_CLOSE");
        let _literal_open = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_STRING_MERGE_OPEN");
        let _literal_close = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_STRING_MERGE_CLOSE");
        let _pattern_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_OPEN");
        let _pattern_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_CLOSE");

        let _string_open = EnvVarGuard::set("GLRMASK_JSON_STRING_MERGE_OPEN", "1");
        let _string_close = EnvVarGuard::set("GLRMASK_JSON_STRING_MERGE_CLOSE", "0");
        let _literal_open = EnvVarGuard::set("GLRMASK_JSON_LITERAL_STRING_MERGE_OPEN", "0");
        let _literal_close = EnvVarGuard::set("GLRMASK_JSON_LITERAL_STRING_MERGE_CLOSE", "1");

        let generic_cfg = json_string_merge_config(false);
        let literal_cfg = json_literal_string_merge_config();
        let pattern_cfg = json_string_merge_config(true);

        assert_eq!(generic_cfg.merge_open, true);
        assert_eq!(generic_cfg.merge_close, false);
        assert_eq!(literal_cfg.merge_open, false);
        assert_eq!(literal_cfg.merge_close, true);
        assert_eq!(pattern_cfg.merge_open, true);
        assert_eq!(pattern_cfg.merge_close, false);

        let _pattern_open = EnvVarGuard::set("GLRMASK_JSON_PATTERN_STRING_MERGE_OPEN", "0");
        let _pattern_close = EnvVarGuard::set("GLRMASK_JSON_PATTERN_STRING_MERGE_CLOSE", "1");
        let pattern_override_cfg = json_string_merge_config(true);
        assert_eq!(pattern_override_cfg.merge_open, false);
        assert_eq!(pattern_override_cfg.merge_close, true);
    }

    #[test]
    fn key_merge_env_layers_resolve_pattern_and_literal_independently() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _key_open = EnvVarGuard::unset("GLRMASK_JSON_KEY_MERGE_OPEN");
        let _key_close = EnvVarGuard::unset("GLRMASK_JSON_KEY_MERGE_CLOSE");
        let _literal_key_open = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_KEY_MERGE_OPEN");
        let _literal_key_close = EnvVarGuard::unset("GLRMASK_JSON_LITERAL_KEY_MERGE_CLOSE");
        let _pattern_key_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_KEY_MERGE_OPEN");
        let _pattern_key_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_KEY_MERGE_CLOSE");

        let _key_open = EnvVarGuard::set("GLRMASK_JSON_KEY_MERGE_OPEN", "1");
        let _key_close = EnvVarGuard::set("GLRMASK_JSON_KEY_MERGE_CLOSE", "0");
        let _literal_key_open = EnvVarGuard::set("GLRMASK_JSON_LITERAL_KEY_MERGE_OPEN", "0");
        let _literal_key_close = EnvVarGuard::set("GLRMASK_JSON_LITERAL_KEY_MERGE_CLOSE", "1");

        let generic_cfg = json_key_merge_config(false);
        let literal_cfg = json_literal_key_merge_config();
        let pattern_cfg = json_key_merge_config(true);

        assert_eq!(generic_cfg.merge_open, true);
        assert_eq!(generic_cfg.merge_close, false);
        assert_eq!(literal_cfg.merge_open, false);
        assert_eq!(literal_cfg.merge_close, true);
        assert_eq!(pattern_cfg.merge_open, true);
        assert_eq!(pattern_cfg.merge_close, false);

        let _pattern_key_open = EnvVarGuard::set("GLRMASK_JSON_PATTERN_KEY_MERGE_OPEN", "0");
        let _pattern_key_close = EnvVarGuard::set("GLRMASK_JSON_PATTERN_KEY_MERGE_CLOSE", "1");
        let pattern_override_cfg = json_key_merge_config(true);
        assert_eq!(pattern_override_cfg.merge_open, false);
        assert_eq!(pattern_override_cfg.merge_close, true);
    }

    #[test]
    fn json_uri_merge_defaults_inherit_pattern_string_defaults() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _string_open = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_OPEN");
        let _string_close = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_CLOSE");
        let _pattern_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_OPEN");
        let _pattern_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_CLOSE");
        let _uri_open = EnvVarGuard::unset("GLRMASK_JSON_URI_MERGE_OPEN");
        let _uri_close = EnvVarGuard::unset("GLRMASK_JSON_URI_MERGE_CLOSE");

        let cfg = json_uri_merge_config();
        assert_eq!(cfg.merge_open, true);
        assert_eq!(cfg.merge_close, false);
    }

    #[test]
    fn json_uri_merge_env_overrides_still_apply_on_top_of_pattern_defaults() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _string_open = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_OPEN");
        let _string_close = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_CLOSE");
        let _pattern_open = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_OPEN");
        let _pattern_close = EnvVarGuard::unset("GLRMASK_JSON_PATTERN_STRING_MERGE_CLOSE");
        let _uri_open = EnvVarGuard::set("GLRMASK_JSON_URI_MERGE_OPEN", "0");
        let _uri_close = EnvVarGuard::set("GLRMASK_JSON_URI_MERGE_CLOSE", "1");

        let cfg = json_uri_merge_config();
        assert_eq!(cfg.merge_open, false);
        assert_eq!(cfg.merge_close, true);
    }

    #[test]
    fn json_schema_uri_mode_defaults_and_explicit_values() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _uri_mode = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_URI_MODE");

        assert_eq!(
            json_schema_uri_mode(),
            JsonSchemaUriMode::StructuredSingleTerminal
        );

        let _uri_mode = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "structured_single_terminal");
        assert_eq!(json_schema_uri_mode(), JsonSchemaUriMode::StructuredSingleTerminal);

        let _uri_mode = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "structured");
        assert_eq!(json_schema_uri_mode(), JsonSchemaUriMode::Structured);

        let _uri_mode = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "llguidance_pattern");
        assert_eq!(json_schema_uri_mode(), JsonSchemaUriMode::LlguidancePattern);

        let _uri_mode = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "approx");
        assert_eq!(json_schema_uri_mode(), JsonSchemaUriMode::Approx);
    }

    #[test]
    fn uri_quote_merge_warning_is_only_for_strict_structured_mode() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _uri_open = EnvVarGuard::unset("GLRMASK_JSON_URI_MERGE_OPEN");
        let _uri_close = EnvVarGuard::unset("GLRMASK_JSON_URI_MERGE_CLOSE");

        assert!(!uri_quote_merge_warning_needed(JsonSchemaUriMode::Structured));

        let _uri_open = EnvVarGuard::set("GLRMASK_JSON_URI_MERGE_OPEN", "1");
        assert!(uri_quote_merge_warning_needed(JsonSchemaUriMode::Structured));
        assert!(!uri_quote_merge_warning_needed(JsonSchemaUriMode::StructuredSingleTerminal));
        assert!(!uri_quote_merge_warning_needed(JsonSchemaUriMode::LlguidancePattern));
        assert!(!uri_quote_merge_warning_needed(JsonSchemaUriMode::Approx));
    }

    #[test]
    fn structured_single_terminal_uri_mode_respects_uri_quote_merge_envs() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _uri_mode = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_URI_MODE", "structured_single_terminal");
        let _uri_open = EnvVarGuard::set("GLRMASK_JSON_URI_MERGE_OPEN", "1");
        let _uri_close = EnvVarGuard::set("GLRMASK_JSON_URI_MERGE_CLOSE", "0");
        let _string_open = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_OPEN");
        let _string_close = EnvVarGuard::unset("GLRMASK_JSON_STRING_MERGE_CLOSE");

        let schema = json!({
            "type": "string",
            "format": "uri"
        });

        let glrm = dump_glrm(schema);
        let uri_rule = find_rule_line_with_prefix(&glrm, "JSON_FORMAT_URI_STRUCTURED");
        assert!(uri_rule.contains("::= \"\\\"\" "));
        assert!(!uri_rule.ends_with("\";"));
    }

    #[test]
    fn pattern_string_wrap_keeps_intersect_top_level() {
        let body = GrammarExpr::Intersect {
            expr: Box::new(GrammarExpr::Ref("lhs".into())),
            intersect: Box::new(GrammarExpr::Ref("rhs".into())),
        };

        let (wrapped, _) = wrap_string_value_expr_parts(body, true);

        match wrapped {
            GrammarExpr::Intersect { expr, intersect } => {
                assert_eq!(
                    *expr,
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Literal(b"\"".to_vec()),
                        GrammarExpr::Ref("lhs".into()),
                    ])
                );
                assert_eq!(
                    *intersect,
                    GrammarExpr::Sequence(vec![
                        GrammarExpr::Literal(b"\"".to_vec()),
                        GrammarExpr::Ref("rhs".into()),
                    ])
                );
            }
            other => panic!("expected top-level intersect, got {other:?}"),
        }
    }

    #[test]
    fn large_required_only_shared_ap_object_errors_as_schema_too_large() {
        let required = (0..33)
            .map(|idx| format!("field_{idx}"))
            .collect::<Vec<_>>();
        let schema = json!({
            "type": "object",
            "required": required,
            "additionalProperties": {"type": "string"}
        });

        let err = schema_to_named_grammar(&schema).unwrap_err();
        assert!(matches!(
            err,
            GlrMaskError::GrammarParse(message) if message == "schema too large"
        ));
    }

    #[test]
    fn small_required_only_shared_ap_object_still_builds() {
        let required = (0..4)
            .map(|idx| format!("field_{idx}"))
            .collect::<Vec<_>>();
        let schema = json!({
            "type": "object",
            "required": required,
            "additionalProperties": {"type": "string"}
        });

        let named = schema_to_named_grammar(&schema).unwrap();
        let glrm = to_glrm(&named);
        assert!(glrm.contains("obj_req_any_"), "{glrm}");
        lower(&named).unwrap();
    }

    fn imported_optional_word_list_pattern_expr(max_pairs: usize) -> LexerExpr {
        let pattern = format!(r"^$|(^(?:\S+\s+){{0,{max_pairs}}}\S+$)");
        LexerExpr::Seq(vec![
            LexerExpr::U8Seq(vec![b'"']),
            decoded_regex_search_expr(&pattern, None),
        ])
    }

    #[test]
    #[ignore = "reproduces slow bounded-repeat suffix determinization"]
    fn imported_json_pattern_optional_choice_build_regex_repro() {
        let expr = imported_optional_word_list_pattern_expr(199);
        let started = std::time::Instant::now();
        let regex = build_regex(std::slice::from_ref(&expr));
        let elapsed = started.elapsed();

        eprintln!(
            "imported_json_pattern_optional_choice_build_regex_repro elapsed={elapsed:?} states={}",
            regex.num_states(),
        );
        assert!(regex.num_states() > 0);
    }
