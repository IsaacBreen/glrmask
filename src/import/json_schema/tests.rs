use serde_json::json;
use std::{env, ffi::OsString, sync::Mutex};

use super::schema_to_named_grammar;
use super::string::property_name_matches_pattern;
use super::lower_exact_subtractions_enabled;
use crate::grammar::ast::{lower, GrammarExpr, NamedGrammar};
use crate::grammar::glrm::to_glrm;

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

fn start_expr(grammar: &NamedGrammar) -> &GrammarExpr {
    &grammar
        .rules
        .iter()
        .find(|rule| rule.name == grammar.start)
        .expect("start rule exists")
        .expr
}

#[test]
fn exact_subtraction_lowering_env_var_defaults_true_and_accepts_falsey_values() {
    let _lock = ENV_LOCK.lock().unwrap();

    let _unset = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS");
    assert!(lower_exact_subtractions_enabled());

    for value in ["", "0", "false", "FALSE", "no", "off"] {
        let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS", value);
        assert!(!lower_exact_subtractions_enabled(), "value {value:?} should disable exact-sub lowering");
    }

    let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS", "1");
    assert!(lower_exact_subtractions_enabled());
}

fn contains_separated_sequence(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::SeparatedSequence { .. } => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_separated_sequence(inner),
        GrammarExpr::RepeatRange { expr, .. } => contains_separated_sequence(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(contains_separated_sequence)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_separated_sequence(expr) || contains_separated_sequence(exclude)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            contains_separated_sequence(expr) || contains_separated_sequence(intersect)
        }
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte
        | GrammarExpr::ExprNFA(_) => false,
    }
}

fn contains_expr_nfa(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::ExprNFA(_) => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_expr_nfa(inner),
        GrammarExpr::RepeatRange { expr, .. } => contains_expr_nfa(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_expr_nfa),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_expr_nfa(expr) || contains_expr_nfa(exclude)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            contains_expr_nfa(expr) || contains_expr_nfa(intersect)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_expr_nfa(item)) || contains_expr_nfa(separator)
        }
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => false,
    }
}

fn count_rules_with_prefix(grammar: &NamedGrammar, prefix: &str) -> usize {
    grammar.rules.iter().filter(|rule| rule.name.starts_with(prefix)).count()
}

fn contains_exclude(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Exclude { .. } => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_exclude(inner),
        GrammarExpr::RepeatRange { expr, .. } => contains_exclude(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_exclude),
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_exclude(item)) || contains_exclude(separator)
        }
        GrammarExpr::Intersect { expr, intersect } => contains_exclude(expr) || contains_exclude(intersect),
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte
        | GrammarExpr::ExprNFA(_) => false,
    }
}

fn contains_intersect(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Intersect { .. } => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_intersect(inner),
        GrammarExpr::RepeatRange { expr, .. } => contains_intersect(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_intersect),
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_intersect(item)) || contains_intersect(separator)
        }
        GrammarExpr::Exclude { expr, exclude } => contains_intersect(expr) || contains_intersect(exclude),
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte
        | GrammarExpr::ExprNFA(_) => false,
    }
}

fn contains_ref_named(expr: &GrammarExpr, name: &str) -> bool {
    match expr {
        GrammarExpr::Ref(rule_name) => rule_name == name,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_ref_named(inner, name),
        GrammarExpr::RepeatRange { expr, .. } => contains_ref_named(expr, name),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(|item| contains_ref_named(item, name))
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_ref_named(item, name))
                || contains_ref_named(separator, name)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_ref_named(expr, name) || contains_ref_named(exclude, name)
        }
        GrammarExpr::Intersect { expr, intersect } => {
            contains_ref_named(expr, name) || contains_ref_named(intersect, name)
        }
        GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte
        | GrammarExpr::ExprNFA(_) => false,
    }
}

#[test]
fn closed_object_lowers_to_expr_nfa_body() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!contains_separated_sequence(start_expr(&grammar)));
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn open_no_pattern_object_lowers_to_expr_nfa_body() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "additionalProperties": {"type": "string"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!contains_separated_sequence(start_expr(&grammar)));
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    assert!(grammar.rules.iter().any(|rule| rule.name == "JSON_ADDITIONAL_KEY_COLON_SHARED"));
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_open_object_uses_fused_prefix_chain_rules() {
    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(format!("k{index}"), json!({"type": "string"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("additionalProperties".to_string(), json!({"type": "string"})),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(count_rules_with_prefix(&grammar, "json_open_object_prefix") > 0);
    assert_eq!(count_rules_with_prefix(&grammar, "json_closed_object_body"), 0);
    assert!(glrm.contains("\", \\\"k1\\\": \" JSON_STRING"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_open_object_allow_any_scalars_still_uses_fused_prefix_chain_rules() {
    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(format!("k{index}"), json!({"type": "string"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("additionalProperties".to_string(), json!(true)),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(count_rules_with_prefix(&grammar, "json_open_object_prefix") > 0);
    assert_eq!(count_rules_with_prefix(&grammar, "json_closed_object_body"), 0);
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_open_object_allow_any_object_valued_at_16_still_uses_fused_prefix_chain_rules() {
    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(
            format!("k{index}"),
            json!({
                "type": "object",
                "properties": {
                    "nested": {"type": "string"}
                }
            }),
        );
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("additionalProperties".to_string(), json!(true)),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(count_rules_with_prefix(&grammar, "json_open_object_prefix") > 0);
    assert!(!contains_expr_nfa(start_expr(&grammar)));
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_open_object_allow_any_object_valued_at_32_uses_expr_nfa_body() {
    let mut properties = serde_json::Map::new();
    for index in 0..32 {
        properties.insert(
            format!("k{index}"),
            json!({
                "type": "object",
                "properties": {
                    "nested": {"type": "string"}
                }
            }),
        );
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("additionalProperties".to_string(), json!(true)),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert_eq!(count_rules_with_prefix(&grammar, "json_open_object_prefix"), 0);
    assert!(count_rules_with_prefix(&grammar, "json_closed_object_body") > 0);
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn large_required_open_object_does_not_use_fused_prefix_chain_rules() {
    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(format!("k{index}"), json!({"type": "string"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("required".to_string(), json!(["k0"])),
        ("additionalProperties".to_string(), json!({"type": "string"})),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert_eq!(count_rules_with_prefix(&grammar, "json_open_object_prefix"), 0);
    assert!(count_rules_with_prefix(&grammar, "json_closed_object_body") > 0);
    lower(&grammar).unwrap();
}

#[test]
fn pattern_property_object_still_uses_separated_sequence() {
    let schema = json!({
        "type": "object",
        "properties": {"kind": {"const": "event"}},
        "patternProperties": {"^x": {"type": "string"}},
        "required": ["kind"],
        "additionalProperties": {"type": "string"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(contains_separated_sequence(start_expr(&grammar)));
    lower(&grammar).unwrap();
}

#[test]
fn allof_drops_vacuous_untyped_object_branch_for_typed_property() {
    let schema = json!({
        "type": "object",
        "properties": {
            "version": {"type": "number"}
        },
        "required": ["version"],
        "additionalProperties": false,
        "patternProperties": {
            "^.+$": {
                "properties": {
                    "parameters": {"type": "object"}
                }
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!contains_intersect(start_expr(&grammar)));
    lower(&grammar).unwrap();
}

#[test]
fn large_snowplow_like_pattern_property_object_uses_expr_nfa_body() {
    let mut properties = serde_json::Map::new();
    for index in 0..64 {
        properties.insert(format!("k{index}"), json!({"type": "string"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        (
            "patternProperties".to_string(),
            json!({
                "^contexts_.*": {"type": "array"},
                "^unstruct_event_.*": {"type": "string"}
            }),
        ),
        ("additionalProperties".to_string(), json!(false)),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!contains_separated_sequence(start_expr(&grammar)));
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn shared_additional_key_colon_terminal_is_emitted_once() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {"known": {"type": "string"}},
                "additionalProperties": false
            },
            "b": {
                "type": "object",
                "additionalProperties": {"type": "integer"}
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let count = grammar
        .rules
        .iter()
        .filter(|rule| rule.name == "JSON_ADDITIONAL_KEY_COLON_SHARED")
        .count();
    assert_eq!(count, 1);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_COLON_SHARED"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn additional_properties_factoring_uses_shared_key_colon_terminal() {
    let schema = json!({
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "comments": {"type": "string"},
                    "contexts": {"type": "string"}
                },
                "additionalProperties": {"type": "string"}
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_COLON_SHARED"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn arrays_use_item_schema_and_min_max_items() {
    let schema = json!({
        "type": "array",
        "items": {"enum": ["a", "b"]},
        "minItems": 1,
        "maxItems": 3
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("{1,3}"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn prefix_items_lower_with_no_tail() {
    let schema = json!({
        "type": "array",
        "prefixItems": [
            {"const": "a"},
            {"const": "b"}
        ],
        "items": false,
        "minItems": 1,
        "maxItems": 2
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\\\"a\\\""), "{glrm}");
    assert!(glrm.contains("\\\"b\\\""), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn legacy_tuple_items_use_additional_items_tail() {
    let schema = json!({
        "type": "array",
        "items": [
            {"const": "head"}
        ],
        "additionalItems": {"type": "integer"},
        "minItems": 1,
        "maxItems": 3
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\\\"head\\\""), "{glrm}");
    assert!(glrm.contains("JSON_INTEGER") || glrm.contains("JSON_NUMBER"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn string_pattern_lowers_as_terminal_pattern() {
    let schema = json!({
        "type": "string",
        "minLength": 2,
        "maxLength": 8,
        "pattern": "^[A-Za-z]+$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("& /\"(?:(?:[A-Za-z])+)\"/"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn medium_bounded_string_uses_split_chunk_rules_by_default() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _terminalize_guard = EnvVarGuard::unset(
        "GLRMASK_JSON_SCHEMA_TERMINALIZE_BOUNDED_STRING_MAX",
    );

    let schema = json!({
        "type": "string",
        "maxLength": 1024
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(
        !grammar
            .rules
            .iter()
            .any(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained")),
        "{:?}",
        grammar.rules
    );

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_string_char_exact_50"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn medium_bounded_string_terminalizes_with_env_override() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _terminalize_guard = EnvVarGuard::set(
        "GLRMASK_JSON_SCHEMA_TERMINALIZE_BOUNDED_STRING_MAX",
        "1024",
    );

    let schema = json!({
        "type": "string",
        "maxLength": 1024
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(
        grammar
            .rules
            .iter()
            .any(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained")),
        "{:?}",
        grammar.rules
    );

    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("json_string_char_exact_50"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn very_large_bounded_string_still_uses_split_chunk_rules() {
    let schema = json!({
        "type": "string",
        "maxLength": 32767
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(
        !grammar
            .rules
            .iter()
            .any(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained")),
        "{:?}",
        grammar.rules
    );

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_string_char_exact_50"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn decoded_string_patterns_are_matched_against_json_string_bodies() {
    assert!(property_name_matches_pattern(r#"^/[^/]+$"#, "/abc").unwrap());
    assert!(!property_name_matches_pattern(r#"^/[^/]+$"#, "/abc/def").unwrap());
    assert!(property_name_matches_pattern("^\"$",
        "\""
    ).unwrap());
    assert!(!property_name_matches_pattern("^\"$", "x").unwrap());
}

#[test]
fn string_format_is_ignored_as_annotation() {
    let schema = json!({
        "type": "string",
        "format": "uuid"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("[0-9A-Fa-f]{8}"), "{glrm}");
    assert!(glrm.contains("JSON_STRING"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn string_pattern_takes_precedence_over_format() {
    let schema = json!({
        "type": "string",
        "format": "uuid",
        "pattern": "^abc$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("& /\"(?:abc)\"/"), "{glrm}");
    assert!(!glrm.contains("[0-9A-Fa-f]{8}"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_nonterminals_reference_terminalized_key_and_string_patterns() {
    let schema = json!({
        "type": "object",
        "properties": {
            "last_modification": {"type": "string", "maxLength": 32, "format": "date-time"},
            "strings": {
                "type": "object",
                "patternProperties": {"^/": {"type": "string"}},
                "additionalProperties": {"type": "string"}
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    for rule in grammar.rules.iter().filter(|rule| !rule.is_terminal) {
        assert!(
            !contains_intersect(&rule.expr),
            "nonterminal {} contains intersect: {:?}",
            rule.name,
            rule.expr
        );
        assert!(
            !contains_ref_named(&rule.expr, "JSON_STRING_CHAR"),
            "nonterminal {} contains JSON_STRING_CHAR: {:?}",
            rule.name,
            rule.expr
        );
    }
    assert!(
        grammar
            .rules
            .iter()
            .any(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
    );
    assert!(
        grammar
            .rules
            .iter()
            .any(|rule| rule.is_terminal && rule.name.starts_with("json_pattern_key_colon"))
    );

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\\\"last_modification\\\": "), "{glrm}");
    assert!(!glrm.contains("\\\"last_modification\\\" JSON_KEY_SEPARATOR"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn overlapping_literal_and_pattern_keys_still_lower_with_shared_factoring() {
    let schema = json!({
        "type": "object",
        "properties": {
            "x-name": {"type": "string"}
        },
        "patternProperties": {
            "^x-": {"type": "string"}
        },
        "additionalProperties": {"type": "string"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_COLON_SHARED"), "{glrm}");
    assert!(glrm.contains("\\\"x-name\\\": "), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn json_separators_are_canonical_space_separated() {
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("(?:, )") || glrm.contains("\", \""), "{glrm}");
    assert!(glrm.contains("(?:: )") || glrm.contains("\": \""), "{glrm}");
    assert!(!glrm.contains("[ \\t\\n\\r]*"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn legacy_id_metadata_is_accepted() {
    let schema = json!({
        "definitions": {
            "commandObject": {
                "id": "command-object",
                "type": "object",
                "properties": {
                    "directory": {"type": "string"}
                }
            }
        },
        "$ref": "#/definitions/commandObject"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn unknown_format_is_ignored_as_annotation() {
    let schema = json!({
        "type": "string",
        "format": "made-up"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn unknown_metadata_keys_are_ignored() {
    let schema = json!({
        "type": "string",
        "version": "x",
        "example": "abc"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_errors_as_unimplemented_key() {
    let schema = json!({
        "oneOf": [
            {"const": "left"},
            {"const": "right"}
        ]
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("oneOf"), "{error}");
}

#[test]
fn not_errors_as_unimplemented_key() {
    let schema = json!({
        "type": "string",
        "not": {"const": "forbidden"}
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("not"), "{error}");
}

#[test]
fn enum_and_const_lower_to_exact_json_literals() {
    let schema = json!({"enum": [null, true, "ready", 7]});
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\"null\""), "{glrm}");
    assert!(glrm.contains("\"true\""), "{glrm}");
    assert!(glrm.contains("\"\\\"ready\\\"\""), "{glrm}");
    assert!(glrm.contains("\"7\""), "{glrm}");
}

#[test]
fn large_string_enum_at_root_uses_raw_regex() {
    let values = (0..80)
        .map(|index| json!(format!("value-{index:02}")))
        .collect::<Vec<_>>();
    let schema = json!({"type": "string", "enum": values});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::RawRegex(_)));
    lower(&grammar).unwrap();
}

#[test]
fn small_string_enum_at_root_remains_choice() {
    let schema = json!({"type": "string", "enum": ["red", "green", "blue"]});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    lower(&grammar).unwrap();
}

#[test]
fn patterned_string_enum_does_not_use_raw_regex_fast_path() {
    let values = (0..80)
        .map(|index| json!(format!("value{index}")))
        .collect::<Vec<_>>();
    let schema = json!({
        "type": "string",
        "pattern": "^value[0-9]+$",
        "enum": values
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::RawRegex(_)));
    lower(&grammar).unwrap();
}

#[test]
fn mixed_type_enum_does_not_use_raw_regex_fast_path() {
    let schema = json!({"enum": ["red", 7, "blue"]});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::RawRegex(_)));
    lower(&grammar).unwrap();
}

#[test]
fn integer_power_of_ten_multiple_lowers_to_regex() {
    let schema = json!({"type": "integer", "multipleOf": 10});
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("/[1-9][0-9]*0") || glrm.contains("/-?(0|[1-9][0-9]*0)/"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn finite_integer_range_multiple_lowers_to_literals() {
    let schema = json!({
        "type": "integer",
        "minimum": 1,
        "maximum": 6,
        "multipleOf": 2
    });
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\"2\" | \"4\" | \"6\""), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn bounded_number_lowers_to_range_regex_not_plain_json_number() {
    let schema = json!({
        "type": "number",
        "minimum": 0,
        "maximum": 65535
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::RawRegex(_)));
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_NUMBER"));
    lower(&grammar).unwrap();
}

#[test]
fn large_bounded_integer_lowers_to_range_regex_not_plain_json_integer() {
    let schema = json!({
        "type": "integer",
        "minimum": 0,
        "maximum": 65535
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::RawRegex(_)));
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_INTEGER"));
    lower(&grammar).unwrap();
}

#[test]
fn number_integer_union_uses_json_number_once() {
    let schema = json!({"type": ["number", "integer"]});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Ref(name) if name == "JSON_NUMBER"));
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_INTEGER"));
    lower(&grammar).unwrap();
}

#[test]
fn anyof_lowers_to_choice() {
    let schema = json!({
        "anyOf": [
            {"type": "null"},
            {"const": "ok"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    lower(&grammar).unwrap();
}

#[test]
fn anyof_allows_sibling_assertions() {
    let schema = json!({
        "anyOf": [
            {"type": "string", "pattern": "^a+$"},
            {"type": "string", "pattern": "^b+$"}
        ],
        "minLength": 2
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn anyof_required_property_object_factors_into_single_expr_nfa_body() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"},
            "c": {"type": "boolean"}
        },
        "additionalProperties": false,
        "anyOf": [
            {"required": ["a"]},
            {"required": ["b"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_closed_object_body"), 1);
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn anyof_closed_object_variants_factor_into_single_expr_nfa_body() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "boolean"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "boolean"},
                    "x": {"type": "boolean"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "boolean"},
                    "y": {"type": "boolean"}
                },
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn anyof_required_property_factoring_falls_back_for_nontrivial_branch() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"},
            "c": {"type": "boolean"}
        },
        "additionalProperties": false,
        "anyOf": [
            {"required": ["a", "b"]},
            {"required": ["c"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    lower(&grammar).unwrap();
}

#[test]
fn anyof_closed_object_variant_factoring_falls_back_for_two_variant_properties() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "boolean"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "boolean"},
                    "x": {"type": "boolean"},
                    "y": {"type": "boolean"}
                },
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    lower(&grammar).unwrap();
}

#[test]
fn anyof_closed_object_variant_factoring_falls_back_for_mismatched_common_schema() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "boolean"},
                    "x": {"type": "boolean"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "y": {"type": "boolean"}
                },
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    lower(&grammar).unwrap();
}

#[test]
fn anyof_closed_object_variants_with_shared_required_prefix_use_exact_variant_nfa() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "boolean"}
                },
                "required": ["a"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "c": {"type": "integer"}
                },
                "required": ["a", "c"],
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_untyped_closed_object_variants_keep_non_object_alternatives() {
    let schema = json!({
        "anyOf": [
            {
                "properties": {
                    "a": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "properties": {
                    "b": {"type": "boolean"}
                },
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start = start_expr(&grammar);
    let GrammarExpr::Choice(alternatives) = start else {
        panic!("expected start choice, got {start:?}");
    };
    assert_eq!(alternatives.len(), 6, "{start:?}");
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    assert!(glrm.contains("json_array"), "{glrm}");
    assert!(glrm.contains("JSON_STRING"), "{glrm}");
    assert!(glrm.contains("JSON_NUMBER"), "{glrm}");
    assert!(glrm.contains("JSON_BOOL"), "{glrm}");
    assert!(glrm.contains("JSON_NULL"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_untyped_closed_object_variants_with_sibling_required_use_exact_variant_nfa() {
    let schema = json!({
        "required": ["image"],
        "anyOf": [
            {
                "properties": {
                    "image": {"type": "string"},
                    "context": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "properties": {
                    "image": {"type": "string"},
                    "docker": {"type": "string"}
                },
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start = start_expr(&grammar);
    let GrammarExpr::Choice(alternatives) = start else {
        panic!("expected start choice, got {start:?}");
    };
    assert_eq!(alternatives.len(), 6, "{start:?}");
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    assert!(glrm.contains("json_array"), "{glrm}");
    assert!(glrm.contains("JSON_STRING"), "{glrm}");
    assert!(glrm.contains("JSON_NUMBER"), "{glrm}");
    assert!(glrm.contains("JSON_BOOL"), "{glrm}");
    assert!(glrm.contains("JSON_NULL"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_explicit_object_variants_do_not_add_non_object_alternatives() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "b": {"type": "boolean"}
                },
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    lower(&grammar).unwrap();
}

#[test]
fn untyped_plain_object_assertions_keep_non_object_alternatives() {
    let schema = json!({
        "properties": {
            "name": {"type": "string"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start = start_expr(&grammar);
    let GrammarExpr::Choice(alternatives) = start else {
        panic!("expected start choice, got {start:?}");
    };
    assert_eq!(alternatives.len(), 6, "{start:?}");
    assert!(glrm.contains("json_closed_object_body"), "{glrm}");
    assert!(glrm.contains("json_array"), "{glrm}");
    assert!(glrm.contains("JSON_STRING"), "{glrm}");
    assert!(glrm.contains("JSON_NUMBER"), "{glrm}");
    assert!(glrm.contains("JSON_BOOL"), "{glrm}");
    assert!(glrm.contains("JSON_NULL"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn explicit_plain_object_assertions_remain_object_only() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    lower(&grammar).unwrap();
}

#[test]
fn untyped_object_and_array_assertions_do_not_take_plain_object_fallback() {
    let schema = json!({
        "properties": {
            "name": {"type": "string"}
        },
        "items": {
            "type": "string"
        }
    });

    assert!(schema_to_named_grammar(&schema).is_err());
}

#[test]
fn anyof_required_property_factoring_falls_back_for_unknown_required_name() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"}
        },
        "additionalProperties": true,
        "anyOf": [
            {"required": ["missing"]},
            {"required": ["a"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    lower(&grammar).unwrap();
}

#[test]
fn allof_merges_plain_object_branches() {
    let schema = json!({
        "allOf": [
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"]
            },
            {
                "type": "object",
                "properties": {"b": {"type": "boolean"}},
                "required": ["b"],
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\\\"a\\\""), "{glrm}");
    assert!(glrm.contains("\\\"b\\\""), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn open_object_anyof_uses_single_object_body_nfa() {
    let schema = json!({
        "type": "object",
        "properties": {
            "ctx": {
                "type": "object",
                "patternProperties": {
                    "^[0-9a-zA-Z_-]{1,255}$": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "string", "maxLength": 32767},
                                    "b": {"type": "number"},
                                    "c": {
                                        "type": "object",
                                        "properties": {
                                            "key": {
                                                "type": "string",
                                                "pattern": "^[0-9a-zA-Z_-]{1,255}$"
                                            },
                                            "value": {
                                                "type": "string",
                                                "minLength": 1,
                                                "maxLength": 255
                                            }
                                        },
                                        "additionalProperties": false
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "string",
                                        "pattern": "^[A-Fa-f\\d]{24}$"
                                    },
                                    "name": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 255
                                    },
                                    "description": {
                                        "type": "string",
                                        "maxLength": 32767
                                    },
                                    "tags": {
                                        "type": "object",
                                        "patternProperties": {
                                            "^[0-9a-zA-Z_-]{1,255}$": {
                                                "type": "array",
                                                "minItems": 1,
                                                "items": {
                                                    "type": "string",
                                                    "minLength": 1,
                                                    "maxLength": 255
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "additionalProperties": false
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    assert!(
        !glrm.contains("\"{\" json_closed_object_body")
            || !glrm.contains("| \"{\" json_closed_object_body"),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn ref_with_sibling_assertions_is_intersected() {
    let schema = json!({
        "$defs": {
            "base": {"type": "string"}
        },
        "$ref": "#/$defs/base",
        "minLength": 2
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("&"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn singleton_allof_ref_without_siblings_reuses_ref_rule() {
    let schema = json!({
        "$defs": {
            "base": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "name": {"type": "string"}
                },
                "additionalProperties": false
            }
        },
        "type": "object",
        "properties": {
            "first": {"allOf": [{"$ref": "#/$defs/base"}]},
            "second": {"allOf": [{"$ref": "#/$defs/base"}]}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert_eq!(count_rules_with_prefix(&grammar, "schema_ref"), 1);
    lower(&grammar).unwrap();
}

#[test]
fn singleton_allof_ref_with_noop_object_siblings_reuses_ref_rule() {
    let schema = json!({
        "$defs": {
            "base": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "name": {"type": "string"}
                },
                "additionalProperties": false
            }
        },
        "type": "object",
        "properties": {
            "wrapped": {
                "allOf": [{"$ref": "#/$defs/base"}],
                "type": "object"
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert_eq!(count_rules_with_prefix(&grammar, "schema_ref"), 1);
    lower(&grammar).unwrap();
}

#[test]
fn singleton_allof_ref_with_restrictive_additional_properties_skips_fast_path() {
    let schema = json!({
        "$defs": {
            "base": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "name": {"type": "string"}
                },
                "additionalProperties": false
            }
        },
        "type": "object",
        "properties": {
            "wrapped": {
                "allOf": [{"$ref": "#/$defs/base"}],
                "type": "object",
                "additionalProperties": false
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert_eq!(count_rules_with_prefix(&grammar, "schema_ref"), 0);
    lower(&grammar).unwrap();
}
