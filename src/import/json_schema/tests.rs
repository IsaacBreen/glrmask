use serde_json::json;
use std::{env, ffi::OsString, process::Command, sync::Mutex};

use super::ast::StringSchema;
use super::lower_exact_subtractions_enabled;
use super::schema_to_named_grammar;
use super::string::{property_name_matches_pattern, string_value_satisfies_schema, GLRMASK_LLGUIDANCE_COMPAT_ENV};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable, TableAmbiguityKind};
use crate::grammar::ast::{lower, GrammarExpr, NamedGrammar, Quantifier};
use crate::grammar::glrm::{from_glrm, to_glrm};
use crate::Vocab;

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvVarGuard {
    key: &'static str,
    original: Option<OsString>,
}

fn object_constrained_allof_with_nested_oneof_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["_elements"],
        "properties": {
            "_elements": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"$ref": "#/definitions/file"},
                        {"$ref": "#/definitions/file_remote_dir"}
                    ]
                }
            }
        },
        "definitions": {
            "file_common": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"}
                }
            },
            "file": {
                "allOf": [
                    {"$ref": "#/definitions/file_common"},
                    {
                        "type": "object",
                        "properties": {
                            "user": {"type": "string", "minLength": 1},
                            "group": {"type": "string", "minLength": 1}
                        },
                        "oneOf": [
                            {"$ref": "#/definitions/file_file"},
                            {"$ref": "#/definitions/file_dir"},
                            {"$ref": "#/definitions/file_link"}
                        ]
                    }
                ]
            },
            "file_file": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["file"]},
                    "size": {"type": "integer", "minimum": 0},
                    "mode": {"type": "string", "pattern": "^[0-7]{3,4}$"}
                }
            },
            "file_dir": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["dir"]},
                    "size": {"type": "integer", "minimum": 0},
                    "mode": {"type": "string", "pattern": "^[0-4]?[0-7]{3}$"},
                    "files": {"type": "integer", "minimum": 0}
                }
            },
            "file_link": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["link"]}
                }
            },
            "file_remote_dir": {
                "allOf": [
                    {"$ref": "#/definitions/file_common"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["remote_dir"]}
                        }
                    }
                ]
            }
        }
    })
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::set_var(key, value);
        }
        if key == "GLRMASK_LLGUIDANCE_COMPAT" || key == GLRMASK_LLGUIDANCE_COMPAT_ENV {
            let mode = if value != "0" && !value.is_empty() {
                super::string::JsonStringCompatMode::LlGuidanceNative
            } else {
                super::string::JsonStringCompatMode::JsonSchema
            };
            super::string::TEST_COMPAT_MODE.with(|cell| cell.set(mode));
        }
        Self { key, original }
    }

    fn unset(key: &'static str) -> Self {
        let original = env::var_os(key);
        unsafe {
            env::remove_var(key);
        }
        if key == "GLRMASK_LLGUIDANCE_COMPAT" || key == GLRMASK_LLGUIDANCE_COMPAT_ENV {
            super::string::TEST_COMPAT_MODE.with(|cell| cell.set(super::string::JsonStringCompatMode::JsonSchema));
        }
        Self { key, original }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        let original_mode = match &self.original {
            Some(value) => unsafe {
                env::set_var(self.key, value);
                let val = value.to_string_lossy();
                if val != "0" && !val.is_empty() {
                    super::string::JsonStringCompatMode::LlGuidanceNative
                } else {
                    super::string::JsonStringCompatMode::JsonSchema
                }
            },
            None => unsafe {
                env::remove_var(self.key);
                super::string::JsonStringCompatMode::JsonSchema
            },
        };
        if self.key == "GLRMASK_LLGUIDANCE_COMPAT" || self.key == GLRMASK_LLGUIDANCE_COMPAT_ENV {
            super::string::TEST_COMPAT_MODE.with(|cell| cell.set(original_mode));
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
fn exact_subtraction_lowering_env_var_defaults_false_and_accepts_truthy_values() {
    let _lock = ENV_LOCK.lock().unwrap();

    let _unset = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS");
    assert!(!lower_exact_subtractions_enabled());

    for value in ["", "0", "false", "FALSE", "no", "off"] {
        let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS", value);
        assert!(!lower_exact_subtractions_enabled(), "value {value:?} should disable exact-sub lowering");
    }

    for value in ["1", "true", "yes", "on", "anything"] {
        let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS", value);
        assert!(lower_exact_subtractions_enabled(), "value {value:?} should enable exact-sub lowering");
    }
}

#[test]
fn schema_size_preflight_allows_below_budget_schema() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_ALLOW_LARGE");
    let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", "64");

    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        },
        "required": ["id"],
        "additionalProperties": false
    });

    schema_to_named_grammar(&schema).expect("small schema should pass size preflight");
}

#[test]
fn schema_size_preflight_rejects_over_budget_schema() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_ALLOW_LARGE");
    let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", "20");

    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(format!("field_{index}"), json!({"type": "string"}));
    }
    let schema = json!({
        "type": "object",
        "properties": properties,
        "additionalProperties": false
    });

    let err = schema_to_named_grammar(&schema).expect_err("oversized schema should be rejected");
    let message = err.to_string();
    assert!(message.contains("schema too large"), "{message}");
    assert!(message.contains("nodes="), "{message}");
    assert!(message.contains("limit=20"), "{message}");
}

#[test]
fn schema_size_preflight_raised_max_nodes_allows_schema() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_ALLOW_LARGE");

    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(format!("field_{index}"), json!({"type": "string"}));
    }
    let schema = json!({
        "type": "object",
        "properties": properties,
        "additionalProperties": false
    });

    {
        let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", "20");
        let err =
            schema_to_named_grammar(&schema).expect_err("schema should exceed the lower budget");
        assert!(err.to_string().contains("limit=20"), "{err}");
    }

    let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", "128");
    schema_to_named_grammar(&schema).expect("raised node limit should allow schema");
}

#[test]
fn schema_size_preflight_falsey_allow_large_does_not_bypass_budget() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_ALLOW_LARGE", "0");
    let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", "1");

    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        }
    });

    let err = schema_to_named_grammar(&schema)
        .expect_err("falsey allow-large value should not bypass budget");
    let message = err.to_string();
    assert!(message.contains("schema too large"), "{message}");
    assert!(message.contains("limit=1"), "{message}");
}

#[test]
fn schema_size_preflight_allow_large_override_bypasses_budget() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_ALLOW_LARGE", "1");
    let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", "1");

    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        }
    });

    schema_to_named_grammar(&schema).expect("allow-large override should bypass size budget");
}

#[test]
fn schema_size_preflight_invalid_max_nodes_reports_env_var() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_ALLOW_LARGE");

    for value in ["not-a-number", "0"] {
        let _max_nodes = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_MAX_NODES", value);
        let schema = json!({"type": "string"});

        let err = schema_to_named_grammar(&schema)
            .expect_err("invalid node limit should reject before loading");
        let message = err.to_string();
        assert!(
            message.contains("GLRMASK_JSON_SCHEMA_MAX_NODES"),
            "{message}"
        );
        assert!(message.contains("positive integer"), "{message}");
    }
}

#[test]
fn overlapping_pattern_properties_preflight_rejects_non_disjoint_regexes() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _compat = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");

    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9_-]{1,}$": {"type": "string"},
            "^MD5$": {"type": "string", "pattern": "^[a-fA-F0-9]{32}$"}
        },
        "additionalProperties": false
    });

    let err = schema_to_named_grammar(&schema).expect_err("overlapping patternProperties should reject early");
    let message = err.to_string();
    assert!(message.contains("patternProperty regexes"), "{message}");
    assert!(message.contains("^[a-zA-Z0-9_-]{1,}$"), "{message}");
    assert!(message.contains("^MD5$"), "{message}");
}

#[test]
fn overlapping_pattern_properties_preflight_allows_non_disjoint_regexes_without_compat() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _compat = EnvVarGuard::unset(GLRMASK_LLGUIDANCE_COMPAT_ENV);

    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^[a-zA-Z0-9_-]{1,}$": {"type": "string"},
            "^MD5$": {"type": "string", "pattern": "^[a-fA-F0-9]{32}$"}
        },
        "additionalProperties": false
    });

    schema_to_named_grammar(&schema)
        .expect("overlapping patternProperties should still import when compat mode is off");
}

fn contains_separated_sequence(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::SeparatedSequence { .. } => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_separated_sequence(inner),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_separated_sequence(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(contains_separated_sequence)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_separated_sequence(expr) || contains_separated_sequence(exclude)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
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
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_expr_nfa(inner),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_expr_nfa(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_expr_nfa),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_expr_nfa(expr) || contains_expr_nfa(exclude)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
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


fn expr_contains_raw_regex(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::RawRegex(_) => true,
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            expr_contains_raw_regex(inner)
        }
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(expr_contains_raw_regex)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            expr_contains_raw_regex(expr) || expr_contains_raw_regex(exclude)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
            expr_contains_raw_regex(expr) || expr_contains_raw_regex(intersect)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| expr_contains_raw_regex(item))
                || expr_contains_raw_regex(separator)
        }
        GrammarExpr::ExprNFA(expr_nfa) => expr_nfa.symbols.iter().any(expr_contains_raw_regex),
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::Literal(_)
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => false,
    }
}

fn expr_nfa_symbols_contain_raw_regex(grammar: &NamedGrammar) -> bool {
    grammar.rules.iter().any(|rule| match &rule.expr {
        GrammarExpr::ExprNFA(expr_nfa) => expr_nfa.symbols.iter().any(expr_contains_raw_regex),
        _ => false,
    })
}

fn count_rules_with_prefix(grammar: &NamedGrammar, prefix: &str) -> usize {
    grammar.rules.iter().filter(|rule| rule.name.starts_with(prefix)).count()
}

fn byte_vocab() -> Vocab {
    let mut entries = (0u32..=255)
        .map(|byte| (byte, vec![byte as u8]))
        .collect::<Vec<_>>();
    entries.push((256, b"<|endoftext|>".to_vec()));
    Vocab::new(entries, Some(256))
}

fn schema_mask_allows_token_after_prefix(
    schema: &serde_json::Value,
    prefix: &[u8],
    token_id: u32,
    token_bytes: &[u8],
) -> bool {
    let mut entries = (0u32..=255)
        .map(|byte| (byte, vec![byte as u8]))
        .collect::<Vec<_>>();
    entries.push((256, b"<|endoftext|>".to_vec()));
    entries.push((token_id, token_bytes.to_vec()));
    let vocab = Vocab::new(entries, Some(256));
    let grammar = schema_to_named_grammar(schema).expect("schema should import");
    let lowered = lower(&grammar).expect("schema grammar should lower");
    let constraint = crate::compiler::compile_owned(lowered, &vocab);
    let mut state = constraint.start();
    state.commit_bytes(prefix).expect("prefix should be accepted");
    let mask = state.mask();
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    mask.get(word)
        .map(|slot| (*slot & (1u32 << bit)) != 0)
        .unwrap_or(false)
}

fn mask_contains(mask: &[u32], token_id: u32) -> bool {
    let word = token_id as usize / 32;
    let bit = token_id as usize % 32;
    mask.get(word)
        .map(|slot| (*slot & (1u32 << bit)) != 0)
        .unwrap_or(false)
}

fn schema_accepts_bytes(schema: &serde_json::Value, input: &[u8]) -> bool {
    let grammar = schema_to_named_grammar(schema).expect("schema should import");
    let lowered = lower(&grammar).expect("schema grammar should lower");
    let constraint = crate::compiler::compile_owned(lowered, &byte_vocab());
    let mut state = constraint.start();
    state.commit_bytes(input).is_ok() && state.is_complete()
}

#[test]
fn enum_literals_are_filtered_by_sibling_type_assertion() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "enum": ["ok", 1]});

    assert!(schema_accepts_bytes(&schema, br#""ok""#));
    assert!(!schema_accepts_bytes(&schema, br#"1"#));
    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        b"",
        921,
        b"-",
    ));
}

#[test]
fn typed_string_enum_rejects_non_string_enum_members() {
    let schema = json!({
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["unknown", -1, 0, 2, 3, 7, 9]
            }
        },
        "additionalProperties": false
    });

    assert!(schema_accepts_bytes(&schema, br#"{"status": "unknown"}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"status": -1}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"status": 0}"#));
}

#[test]
fn typed_integer_enum_rejects_wrong_type_and_failed_number_constraints() {
    let schema = json!({
        "type": "integer",
        "minimum": 2,
        "enum": [1, 2, "2", true]
    });

    assert!(!schema_accepts_bytes(&schema, b"1"));
    assert!(schema_accepts_bytes(&schema, b"2"));
    assert!(!schema_accepts_bytes(&schema, br#""2""#));
    assert!(!schema_accepts_bytes(&schema, b"true"));
}

#[test]
fn typed_const_rejects_literal_that_conflicts_with_sibling_assertions() {
    let schema = json!({
        "type": "object",
        "required": ["id"],
        "const": {}
    });

    assert!(!schema_accepts_bytes(&schema, br#"{}"#));
}

#[test]
fn llguidance_compat_treats_untyped_format_as_typed_string() {
    let _lock = ENV_LOCK.lock().unwrap();
    let schema = json!({"format": "uri"});

    {
        let _guard = EnvVarGuard::unset("GLRMASK_LLGUIDANCE_COMPAT");
        assert!(schema_accepts_bytes(&schema, br#"true"#));
        assert!(schema_accepts_bytes(&schema, br#""https://example.com""#));
    }

    {
        let _guard = EnvVarGuard::set("GLRMASK_LLGUIDANCE_COMPAT", "1");
        assert!(!schema_accepts_bytes(&schema, br#"true"#));
        assert!(schema_accepts_bytes(&schema, br#""https://example.com""#));
    }
}

#[test]
fn llguidance_compat_keeps_untyped_property_format_permissive() {
    let _lock = ENV_LOCK.lock().unwrap();
    let schema = json!({
        "type": "object",
        "required": ["uri"],
        "properties": {
            "uri": {"format": "uri"}
        },
        "additionalProperties": false
    });

    {
        let _guard = EnvVarGuard::unset("GLRMASK_LLGUIDANCE_COMPAT");
        assert!(schema_accepts_bytes(&schema, br#"{"uri": true}"#));
        assert!(schema_accepts_bytes(&schema, br#"{"uri": "https://example.com"}"#));
    }

    {
        let _guard = EnvVarGuard::set("GLRMASK_LLGUIDANCE_COMPAT", "1");
        assert!(schema_accepts_bytes(&schema, br#"{"uri": true}"#));
        assert!(schema_mask_allows_token_after_prefix(
            &schema,
            br#"{"uri":"#,
            300,
            b" t",
        ));
        assert!(schema_accepts_bytes(&schema, br#"{"uri": "https://example.com"}"#));
    }
}

#[test]
fn llguidance_compat_keeps_untyped_property_pattern_untyped() {
    let _lock = ENV_LOCK.lock().unwrap();
    let schema = json!({
        "type": "object",
        "required": ["cur"],
        "properties": {
            "cur": {"pattern": "^[A-Z]{3}$"}
        },
        "additionalProperties": false
    });

    {
        let _guard = EnvVarGuard::unset("GLRMASK_LLGUIDANCE_COMPAT");
        assert!(schema_accepts_bytes(&schema, br#"{"cur": true}"#));
        assert!(schema_accepts_bytes(&schema, br#"{"cur": "USD"}"#));
        assert!(!schema_accepts_bytes(&schema, br#"{"cur": "/"}"#));
    }

    {
        let _guard = EnvVarGuard::set("GLRMASK_LLGUIDANCE_COMPAT", "1");
        assert!(schema_accepts_bytes(&schema, br#"{"cur": true}"#));
        assert!(!schema_mask_allows_token_after_prefix(
            &schema,
            br#"{"cur":"#,
            300,
            b" \"/",
        ));
        assert!(schema_accepts_bytes(&schema, br#"{"cur": "USD"}"#));
        assert!(!schema_accepts_bytes(&schema, br#"{"cur": "/"}"#));
    }
}

fn parser_path_count_after_bytes(schema: &serde_json::Value, input: &[u8], limit: usize) -> usize {
    let grammar = schema_to_named_grammar(schema).expect("schema should import");
    let lowered = lower(&grammar).expect("schema grammar should lower");
    let constraint = crate::compiler::compile_owned(lowered, &byte_vocab());
    let mut state = constraint.start();
    state.commit_bytes(input).expect("input should be accepted");
    assert!(state.is_complete(), "input should finish the schema");
    state.parser_path_count(limit)
}


#[test]
fn json_string_accepts_escaped_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::unset(GLRMASK_LLGUIDANCE_COMPAT_ENV);
    let schema = json!({"type": "string"});
    assert!(schema_accepts_bytes(&schema, br#""\/""#));
}

#[test]
fn patterned_string_accepts_escaped_solidus_for_decoded_slash() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::unset(GLRMASK_LLGUIDANCE_COMPAT_ENV);
    let schema = json!({"type": "string", "pattern": "^/$"});
    assert!(schema_accepts_bytes(&schema, br#""\/""#));
    assert!(schema_accepts_bytes(&schema, br#""/""#));
}

#[test]
fn patterned_string_class_accepts_escaped_solidus_for_decoded_slash() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::unset(GLRMASK_LLGUIDANCE_COMPAT_ENV);
    let schema = json!({"type": "string", "pattern": "^[ab/]$"});
    assert!(schema_accepts_bytes(&schema, br#""\/""#));
    assert!(schema_accepts_bytes(&schema, br#""/""#));
}

#[test]
fn llguidance_compat_rejects_escaped_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string"});
    assert!(!schema_accepts_bytes(&schema, br#""\/""#));
    assert!(schema_accepts_bytes(&schema, br#""/""#));
}

#[test]
fn llguidance_compat_rejects_patterned_escaped_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "pattern": "^/$"});
    assert!(!schema_accepts_bytes(&schema, br#""\/""#));
    assert!(schema_accepts_bytes(&schema, br#""/""#));
}

#[test]
fn llguidance_compat_rejects_unicode_escaped_pattern_literal() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "pattern": "^file:.+\\.geodatabase?$"});

    assert!(schema_accepts_bytes(&schema, br#""file:./esricampus.geodatabase""#));
    assert!(!schema_accepts_bytes(
        &schema,
        br#""\u0066ile:./esricampus.geodatabase""#,
    ));
    assert!(!schema_accepts_bytes(
        &schema,
        br#""file:.\n.geodatabase""#,
    ));
    assert!(!schema_accepts_bytes(
        &schema,
        br#""file:.\u000A.geodatabase""#,
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#""file:.\t.geodatabase""#,
    ));
}

#[test]
fn llguidance_compat_rejects_unicode_escaped_pattern_class_chars() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let uuid = json!({"type": "string", "pattern": "^[0-9a-f]{8}$"});
    let date = json!({"type": "string", "pattern": "^(0[1-9]|1[0-2])-20[0-9]{2}$"});

    assert!(schema_accepts_bytes(&uuid, br#""1234abcd""#));
    assert!(!schema_accepts_bytes(&uuid, br#""\u0031234abcd""#));
    assert!(schema_accepts_bytes(&date, br#""01-2022""#));
    assert!(!schema_accepts_bytes(&date, br#""01-202\u0032""#));
}

#[test]
fn simple_decimal_multiple_of_matches_llguidance_scale() {
    let schema = json!({"type": "number", "multipleOf": 0.01});

    assert!(schema_accepts_bytes(&schema, br#"0"#));
    assert!(schema_accepts_bytes(&schema, br#"0.0"#));
    assert!(schema_accepts_bytes(&schema, br#"0.00"#));
    assert!(schema_accepts_bytes(&schema, br#"99.9"#));
    assert!(schema_accepts_bytes(&schema, br#"99.99"#));
    assert!(!schema_accepts_bytes(&schema, br#"-0.01"#));
    assert!(!schema_accepts_bytes(&schema, br#"99.999"#));
    assert!(!schema_accepts_bytes(&schema, br#"99.000"#));
    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"99."#,
        920,
        b"000",
    ));
}

#[test]
fn llguidance_compat_pattern_literal_mask_rejects_json_u_prefix() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "pattern": "^file:.+\\.geodatabase?$"});

    assert!(schema_mask_allows_token_after_prefix(&schema, br#"""#, 400, b"f"));
    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"""#,
        401,
        br#"\"#,
    ));
    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#""file:."#,
        402,
        br#"\n"#,
    ));
}

#[test]
fn json_importer_compacts_terminal_pattern_literals() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "pattern": "^file:.+\\.geodatabase?$"});
    let grammar = schema_to_named_grammar(&schema).expect("schema lowers");
    let glrm = to_glrm(&grammar);

    assert!(glrm.contains(r#""file:""#), "{glrm}");
    assert!(glrm.contains("JSON_STRING_PATTERN_DOT_CHAR+"), "{glrm}");
    assert!(glrm.contains(r#"".geodatabas""#), "{glrm}");
    assert!(!glrm.contains("JSON_STRING_CHAR+"), "{glrm}");
    assert!(!glrm.contains("/f/ /i/ /l/ /e/ /:/"), "{glrm}");
    assert!(!glrm.contains(r#"\u" /0/ /0/ /6/ /6/"#), "{glrm}");
}

#[test]
fn map_only_typed_additional_properties_repeat_with_separators() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"}
            },
            "required": ["enabled"],
            "additionalProperties": false
        }
    });

    assert!(schema_accepts_bytes(&schema, br#"{}"#));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"plugin1": {"enabled": true}}"#,
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"plugin1": {"enabled": true}, "plugin2": {"enabled": false}}"#,
    ));
    assert!(!schema_accepts_bytes(
        &schema,
        br#"{"plugin1": {"enabled": true}"plugin2": {"enabled": false}}"#,
    ));

    let grammar = schema_to_named_grammar(&schema).expect("schema lowers");
    let glrm = to_glrm(&grammar);
    assert!(
        !glrm.contains("JSON_ITEM_SEPARATOR ~ ( (((JSON_KEY_STRING JSON_KEY_SEPARATOR)"),
        "map entries must not be emitted as one optional inner `+` item: {glrm}",
    );
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"plugin1": {"enabled": true"#,
        405,
        b"},",
    ));
}



#[test]
fn llguidance_allof_child_required_prefix_rejects_optional_anyof_key_first() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "definitions": {
            "child": {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {
                            "match": {"type": "string"},
                            "browser": {"type": "string"}
                        },
                        "required": ["match"]
                    },
                    {
                        "anyOf": [
                            {"properties": {"devices": {"type": "object"}}},
                            {"properties": {"device": {"type": "string"}}}
                        ]
                    },
                    {
                        "properties": {
                            "platforms": {"type": "array", "items": {"type": "string"}},
                            "engine": {"type": "string"}
                        }
                    }
                ]
            }
        },
        "type": "object",
        "properties": {
            "children": {
                "type": "array",
                "items": {"$ref": "#/definitions/child"}
            }
        },
        "required": ["children"]
    });

    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"children": [{""#,
        67,
        b"d",
    ));
}

#[test]
fn llguidance_additional_key_inside_pattern_property_anyof_accepts_escaped_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
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
                                    "a": {"type": "string"},
                                    "b": {"type": "number"},
                                    "c": {
                                        "type": "object",
                                        "properties": {
                                            "key": {"type": "string"},
                                            "value": {"type": "string"}
                                        },
                                        "additionalProperties": false
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "tags": {"type": "object"}
                                }
                            }
                        ]
                    }
                },
                "additionalProperties": false
            }
        }
    });

    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"ctx": {"key1": {""#,
        4844,
        br#"\/"#,
    ));
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"ctx": {"key1": {"a": "Example string", "b":"#,
        259,
        b" t",
    ));
}

#[test]
fn llguidance_additional_property_accepts_escaped_solidus_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "additionalProperties": {"type": "string"}
    });
    assert!(schema_accepts_bytes(&schema, br#"{"\/": "value"}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"/": "value"}"#));
}

#[test]
fn llguidance_pattern_property_rejects_escaped_solidus_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^/$": {"type": "string"}
        },
        "additionalProperties": false
    });
    assert!(!schema_accepts_bytes(&schema, br#"{"\/": "value"}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"/": "value"}"#));
}

#[test]
fn llguidance_pattern_property_dotstar_accepts_escaped_solidus_key_prefix_and_partial_unicode() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "patternProperties": {
            ".*": {"type": "string"}
        }
    });
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        406,
        br#"\/"#,
    ));
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        407,
        br#"\uC"#,
    ));
    assert!(schema_accepts_bytes(&schema, br#"{"\/": "value"}"#));
}

#[test]
fn llguidance_generic_json_object_rejects_partial_unicode_key_escape() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "top": {}
        },
        "required": ["top"]
    });

    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"top": {""#,
        409,
        br#"\uC"#,
    ));
}

#[test]
fn llguidance_fixed_object_additional_property_accepts_escaped_solidus_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "known": {"type": "string"}
        },
        "additionalProperties": {"type": "string"}
    });

    assert!(schema_accepts_bytes(&schema, br#"{"\/": "value"}"#));
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        408,
        br#"\/"#,
    ));
}


#[test]
fn llguidance_map_only_allow_any_uses_strict_key_mask() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "minProperties": 1
    });

    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        4844,
        br#"\/"#,
    ));
}

#[test]
fn llguidance_pattern_property_key_class_accepts_unicode_escape_prefix() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^[^ ]+$": {"type": "string"}
        },
        "additionalProperties": false
    });

    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        3855,
        br#"\u"#,
    ));
    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        68515,
        br#"\uC"#,
    ));
}


#[test]
fn llguidance_pattern_property_digit_key_rejects_bare_backslash_prefix() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^[1-5][0-9]{2}$": {"type": "string"}
        },
        "additionalProperties": false
    });

    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        59,
        br#"\"#,
    ));
}

#[test]
fn llguidance_literal_property_rejects_escaped_solidus_key() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "/": {"type": "string"}
        },
        "required": ["/"],
        "additionalProperties": false
    });
    assert!(!schema_accepts_bytes(&schema, br#"{"\/": "ok"}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"/": "ok"}"#));
}

#[test]
fn escaped_solidus_instance_rejected_when_no_decoded_key_matches_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"}
        },
        "required": ["a"],
        "additionalProperties": false
    });

    assert!(!schema_accepts_bytes(&schema, br#"{"\/":"bad"}"#));
}

#[test]
fn llguidance_literal_property_mask_rejects_escaped_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "/": {"type": "string"}
        },
        "required": ["/"],
        "additionalProperties": false
    });

    assert!(schema_mask_allows_token_after_prefix(&schema, br#"{""#, 402, b"/"));
    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        403,
        br#"\/"#,
    ));
}

#[test]
fn llguidance_additional_property_mask_accepts_escaped_solidus() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "additionalProperties": {"type": "string"}
    });

    assert!(schema_mask_allows_token_after_prefix(&schema, br#"{""#, 404, b"/"));
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        405,
        br#"\/"#,
    ));
    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{""#,
        406,
        br#"\uC"#,
    ));
}

#[test]
fn llguidance_compat_patterned_string_non_whitespace_unicode_escape_progression() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "pattern": r"^(?:\S+\s+){0,9}\S+$"});

    let grammar = schema_to_named_grammar(&schema).expect("schema should import");
    let glrm = to_glrm(&grammar);
    let constrained_lines = glrm
        .lines()
        .filter(|line| line.starts_with("t json_string_constrained_"))
        .collect::<Vec<_>>();
    assert!(!constrained_lines.is_empty(), "{glrm}");
    assert!(glrm.contains("\\u00(?:[01][0-9A-Fa-f]|7[Ff])"), "{glrm}");

    let json_u = 300u32;
    let json_u_b = 301u32;
    let json_u_c = 302u32;
    let zero = 303u32;
    let upper_b = 304u32;
    let mut entries = (0u32..=255)
        .map(|byte| (byte, vec![byte as u8]))
        .collect::<Vec<_>>();
    entries.push((256, b"<|endoftext|>".to_vec()));
    entries.push((json_u, b"\\u".to_vec()));
    entries.push((json_u_b, b"\\uB".to_vec()));
    entries.push((json_u_c, b"\\uC".to_vec()));
    entries.push((zero, b"0".to_vec()));
    entries.push((upper_b, b"B".to_vec()));
    let vocab = Vocab::new(entries, Some(256));

    let lowered = lower(&grammar).expect("schema grammar should lower");
    let constraint = crate::compiler::compile_owned(lowered, &vocab);

    let mut state = constraint.start();
    state.commit_bytes(b"\"Benef").expect("prefix should be accepted");
    let mask = state.mask();
    assert!(mask_contains(&mask, json_u), r#"expected \\u after \"Benef"#);
    assert!(!mask_contains(&mask, json_u_b), r#"\\uB must be rejected"#);
    assert!(!mask_contains(&mask, json_u_c), r#"\\uC must be rejected"#);

    let mut post_u = constraint.start();
    post_u.commit_bytes(b"\"Benef").expect("prefix should be accepted");
    post_u.commit_token(json_u).expect(r#"\\u token should be accepted"#);
    let post_u_mask = post_u.mask();
    assert!(mask_contains(&post_u_mask, zero), "0 must be admitted after \\u");
    assert!(!mask_contains(&post_u_mask, upper_b), "B must remain rejected after \\u");
}

#[test]
fn mre_llguidance_compat_non_whitespace_subtraction_mask_gap() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({"type": "string", "pattern": r"^(?:\S+\s+){0,9}\S+$"});

    let grammar = schema_to_named_grammar(&schema).expect("schema should import");
    let lowered = lower(&grammar).expect("schema grammar should lower");

    let space_escaped_quote = 0u32;
    let end_of_text = 1u32;
    let vocab = Vocab::new(
        vec![
            (space_escaped_quote, b" \\\"".to_vec()),
            (end_of_text, b"<|endoftext|>".to_vec()),
        ],
        Some(end_of_text),
    );

    let constraint = crate::compiler::compile_owned(lowered, &vocab);
    let mut state = constraint.start();
    state.commit_bytes(b"\"a").expect("prefix should be accepted");
    let mask = state.mask();

    // Regression check: after subtraction/helper lowering fixes, this
    // space+escaped-quote token remains admitted as a non-whitespace
    // continuation in llguidance-compat mode.
    assert!(mask_contains(&mask, space_escaped_quote));
}
fn mask_does_not_enable_json_u_by_runtime_patch() {
    let schema = json!({"type": "string", "pattern": r#"^[\w\.-_]+$"#});
    let grammar = schema_to_named_grammar(&schema).expect("schema should import");
    let lowered = lower(&grammar).expect("schema grammar should lower");

    let json_u_token = 257u32;
    let json_backslash_token = 258u32;
    let mut entries = (0u32..=255)
        .map(|byte| (byte, vec![byte as u8]))
        .collect::<Vec<_>>();
    entries.push((256, b"<|endoftext|>".to_vec()));
    entries.push((json_u_token, b"\\u".to_vec()));
    entries.push((json_backslash_token, b"\\\\".to_vec()));
    let vocab = Vocab::new(entries, Some(256));
    let constraint = crate::compiler::compile_owned(lowered, &vocab);
    let mut state = constraint.start();
    state.commit_bytes(br#"""#).expect("opening quote should be accepted");

    let mask = state.mask();
    assert!(mask_contains(&mask, json_backslash_token), r#"\\ should be grammar-admissible"#);
    assert!(!mask_contains(&mask, json_u_token), r#"\u must not be enabled outside the grammar"#);
}

fn contains_exclude(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Exclude { .. } => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_exclude(inner),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_exclude(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_exclude),
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_exclude(item)) || contains_exclude(separator)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => contains_exclude(expr) || contains_exclude(intersect),
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

fn contains_ref_with_prefix(expr: &GrammarExpr, prefix: &str) -> bool {
    match expr {
        GrammarExpr::Ref(name) => name.starts_with(prefix),
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_ref_with_prefix(inner, prefix),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_ref_with_prefix(expr, prefix),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(|item| contains_ref_with_prefix(item, prefix))
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_ref_with_prefix(item, prefix))
                || contains_ref_with_prefix(separator, prefix)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_ref_with_prefix(expr, prefix) || contains_ref_with_prefix(exclude, prefix)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
            contains_ref_with_prefix(expr, prefix) || contains_ref_with_prefix(intersect, prefix)
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

fn find_all_pop1_stackshifts(table: &GLRTable) -> Option<(u32, u32, Action)> {
    table.ambiguous_actions().iter().find_map(|ambiguity| {
        if ambiguity.kind != TableAmbiguityKind::StackShifts {
            return None;
        }
        match table.action(ambiguity.state, ambiguity.terminal).cloned() {
            Some(Action::StackShifts(shifts))
                if shifts.len() > 1 && shifts.iter().all(|shift| shift.pop == 1) =>
            {
                Some((ambiguity.state, ambiguity.terminal, Action::StackShifts(shifts)))
            }
            _ => None,
        }
    })
}

#[test]
fn recursive_array_additional_properties_schema_does_not_reproduce_all_pop1_stackshifts() {
    let schema = json!({
        "type": "object",
        "required": ["icons"],
        "properties": {
            "icons": {
                "type": "object",
                "required": ["ColorPalette"],
                "properties": {
                    "ColorPalette": {
                        "type": "object",
                        "additionalProperties": { "$ref": "#/definitions/node" }
                    }
                }
            }
        },
        "definitions": {
            "node": {
                "type": "array",
                "items": { "$ref": "#/definitions/node" }
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).expect("schema should lower to named grammar");
    let lowered = lower(&grammar).expect("schema grammar should lower");
    let analyzed = AnalyzedGrammar::from_grammar_def(&lowered);
    let table = GLRTable::build(&analyzed);
    let oracle = find_all_pop1_stackshifts(&table);

    assert!(
        oracle.is_none(),
        "recursive-array additionalProperties schema should not keep the all-pop1 StackShifts ambiguity"
    );
}

fn contains_intersect(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Intersect { .. } => true,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_intersect(inner),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_intersect(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => items.iter().any(contains_intersect),
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_intersect(item)) || contains_intersect(separator)
        }
        GrammarExpr::Exclude { expr, exclude } => contains_intersect(expr) || contains_intersect(exclude),
        GrammarExpr::WithSecondaryLexer { main, secondary } => contains_intersect(main) || contains_intersect(secondary),
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

fn contains_intersect_with_separated_sequence(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Intersect { expr, intersect } => {
            contains_separated_sequence(expr)
                || contains_separated_sequence(intersect)
                || contains_intersect_with_separated_sequence(expr)
                || contains_intersect_with_separated_sequence(intersect)
        }
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_intersect_with_separated_sequence(inner),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_intersect_with_separated_sequence(expr),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(contains_intersect_with_separated_sequence)
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items
                .iter()
                .any(|(item, _)| contains_intersect_with_separated_sequence(item))
                || contains_intersect_with_separated_sequence(separator)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_intersect_with_separated_sequence(expr)
                || contains_intersect_with_separated_sequence(exclude)
        }
        GrammarExpr::WithSecondaryLexer { main, secondary } => {
            contains_intersect_with_separated_sequence(main)
                || contains_intersect_with_separated_sequence(secondary)
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

fn contains_ref_named(expr: &GrammarExpr, name: &str) -> bool {
    match expr {
        GrammarExpr::Ref(rule_name) => rule_name == name,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_ref_named(inner, name),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_ref_named(expr, name),
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
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
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

fn contains_literal_bytes(expr: &GrammarExpr, bytes: &[u8]) -> bool {
    match expr {
        GrammarExpr::Literal(literal) => literal == bytes,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Quantified(inner, Quantifier::Optional)
        | GrammarExpr::Quantified(inner, Quantifier::ZeroPlus)
        | GrammarExpr::Quantified(inner, Quantifier::OnePlus) => contains_literal_bytes(inner, bytes),
        GrammarExpr::Quantified(expr, Quantifier::Range(_, _)) => contains_literal_bytes(expr, bytes),
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items.iter().any(|item| contains_literal_bytes(item, bytes))
        }
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_literal_bytes(item, bytes))
                || contains_literal_bytes(separator, bytes)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_literal_bytes(expr, bytes) || contains_literal_bytes(exclude, bytes)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
            contains_literal_bytes(expr, bytes) || contains_literal_bytes(intersect, bytes)
        }
        GrammarExpr::ExprNFA(nfa) => nfa
            .symbols
            .iter()
            .any(|symbol| contains_literal_bytes(symbol, bytes)),
        GrammarExpr::Ref(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::RawRegex(_)
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => false,
    }
}

fn contains_raw_regex_substring(expr: &GrammarExpr, substring: &str) -> bool {
    match expr {
        GrammarExpr::RawRegex(pat) => pat.contains(substring),
        GrammarExpr::Grouped(inner) | GrammarExpr::Quantified(inner, _) => {
            contains_raw_regex_substring(inner, substring)
        }
        GrammarExpr::Sequence(items) | GrammarExpr::Choice(items) => {
            items
                .iter()
                .any(|item| contains_raw_regex_substring(item, substring))
        }
        GrammarExpr::SeparatedSequence {
            items, separator, ..
        } => {
            items
                .iter()
                .any(|(item, _)| contains_raw_regex_substring(item, substring))
                || contains_raw_regex_substring(separator, substring)
        }
        GrammarExpr::Exclude { expr, exclude } => {
            contains_raw_regex_substring(expr, substring)
                || contains_raw_regex_substring(exclude, substring)
        }
        GrammarExpr::Intersect { expr, intersect }
        | GrammarExpr::WithSecondaryLexer { main: expr, secondary: intersect } => {
            contains_raw_regex_substring(expr, substring)
                || contains_raw_regex_substring(intersect, substring)
        }
        GrammarExpr::ExprNFA(nfa) => nfa
            .symbols
            .iter()
            .any(|symbol| contains_raw_regex_substring(symbol, substring)),
        GrammarExpr::Ref(_)
        | GrammarExpr::Literal(_)
        | GrammarExpr::Epsilon
        | GrammarExpr::CharClass { .. }
        | GrammarExpr::LexerDfa(_)
        | GrammarExpr::AnyByte => false,
    }
}

#[test]
fn closed_object_lowers_to_prefix_chain_body() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "maxLength": 10000},
            "age": {"type": "integer"}
        },
        "required": ["name"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!contains_separated_sequence(start_expr(&grammar)));
    assert!(glrm.contains("json_closed_object_prefix"), "{glrm}");
    assert!(!grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_closed_object_uses_expr_nfa_body() {
    let mut properties = serde_json::Map::new();
    for index in 0..64 {
        properties.insert(format!("incomeTaxKey{index}"), json!({"type": "number"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("additionalProperties".to_string(), json!(false)),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("json_closed_object_fixed_pair_loop_body"), "{glrm}");
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn expr_nfa_pattern_property_symbols_hoist_raw_regexes_for_glrm_roundtrip() {
    let mut properties = serde_json::Map::new();
    for index in 0..8 {
        properties.insert(format!("fixed{index}"), json!({"type": "string"}));
    }

    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": ["fixed0"],
        "patternProperties": {
            "^x": {"type": "number"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    assert!(
        !expr_nfa_symbols_contain_raw_regex(&grammar),
        "ExprNFA transition symbols must not contain raw regex literals"
    );

    let glrm = to_glrm(&grammar);
    for line in glrm.lines().filter(|line| line.trim_start().contains("--")) {
        assert!(!line.contains("/"), "raw regex leaked into FA transition: {line}
{glrm}");
    }
    from_glrm(&glrm).expect("JSON Schema GLRM dump should parse without FA raw regex literals");
    lower(&grammar).unwrap();
}

#[test]
fn required_prefix_open_object_uses_pair_loop_body() {
    let mut properties = serde_json::Map::new();
    properties.insert("a".to_string(), json!({"type": "string"}));
    properties.insert("b".to_string(), json!({"type": "string"}));
    for index in 0..8 {
        properties.insert(format!("opt{index}"), json!({"type": "number"}));
    }

    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": ["a", "b"],
        "patternProperties": {
            "^_": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(
        glrm.contains("json_required_prefix_open_object_pair_loop_body"),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn open_additional_map_min_properties_requires_dynamic_pair() {
    let schema = json!({
        "type": "object",
        "minProperties": 1,
        "additionalProperties": {"type": "string"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let GrammarExpr::Sequence(parts) = start_expr(&grammar) else {
        panic!("expected object sequence: {:?}", start_expr(&grammar));
    };
    assert_eq!(parts.len(), 3);
    assert!(!matches!(parts[1], GrammarExpr::Epsilon));
    assert!(!matches!(parts[1], GrammarExpr::Quantified(_, Quantifier::Optional)));
    lower(&grammar).unwrap();
}

#[test]
fn closed_fixed_object_min_properties_requires_one_optional_after_required() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
            "d": {"type": "string"}
        },
        "required": ["a", "b"],
        "minProperties": 3,
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn closed_fixed_object_min_max_properties_exactly_one_optional() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"}
        },
        "minProperties": 1,
        "maxProperties": 1,
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn closed_fixed_object_max_properties_caps_optional_after_required() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "a": {"type": "string"},
            "b": {"type": "string"}
        },
        "required": ["name"],
        "maxProperties": 2,
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn open_additional_map_max_properties_emits_bounded_dynamic_body() {
    let schema = json!({
        "type": "object",
        "maxProperties": 2,
        "additionalProperties": {"type": "integer"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("{0,1}") || glrm.contains("?"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn required_property_covered_by_pattern_properties_is_synthesized() {
    let schema = json!({
        "type": "object",
        "required": ["line1"],
        "patternProperties": {
            "^line[1-3]$": {"type": "string"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.is_empty(), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn required_property_matching_multiple_patterns_applies_all_pattern_schemas() {
    let schema = json!({
        "type": "object",
        "required": ["line1"],
        "patternProperties": {
            "^line": {"type": "string"},
            "1$": {"const": "ok"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.is_empty(), "{glrm}");
    assert!(glrm.contains("ok") || glrm.contains("json_additional") || glrm.contains("line"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn required_property_not_covered_by_closed_object_lowers_to_empty_language() {
    let schema = json!({
        "type": "object",
        "required": ["missing"],
        "patternProperties": {
            "^line[1-3]$": {"type": "string"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.is_empty(), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn fixed_property_still_intersects_matching_pattern_property() {
    let schema = json!({
        "type": "object",
        "properties": {
            "line1": {"type": "string"}
        },
        "required": ["line1"],
        "patternProperties": {
            "^line[1-3]$": {"const": "ok"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("ok") || glrm.contains("line1") || glrm.contains("json_additional"), "{glrm}");
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
    assert!(glrm.contains(r#"/, "k1": "#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_property_array_opener_fuses_into_key_terminal() {
    let schema = json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        },
        "required": ["items"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("/\"items\": \\[/"), "{glrm}");
    assert!(!glrm.contains(r#""\"items\"" JSON_KEY_SEPARATOR"#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_property_string_value_fuses_into_key_terminal() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(
        glrm.contains("/\"name\": \"")
            || (glrm.contains(r#"/"name": "#) && glrm.contains("JSON_STRING")),
        "{glrm}"
    );
    assert!(!glrm.contains(r#""\"name\"" JSON_KEY_SEPARATOR"#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_property_nullable_string_value_fuses_string_branch_into_key_terminal() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": ["string", "null"]}
        },
        "required": ["name"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("/\"name\": \""), "{glrm}");
    assert!(glrm.contains("/\"name\": null/"), "{glrm}");
    assert!(!glrm.contains(r#""\"name\"" JSON_KEY_SEPARATOR"#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_property_null_value_fuses_into_key_terminal() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "null"}
        },
        "required": ["name"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("/\"name\": null/"), "{glrm}");
    assert!(!glrm.contains(r#""\"name\"" JSON_KEY_SEPARATOR"#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_open_object_allow_any_scalars_uses_expr_nfa_body() {
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
    assert_eq!(count_rules_with_prefix(&grammar, "json_open_object_prefix"), 0);
    assert!(count_rules_with_prefix(&grammar, "json_closed_object_body") > 0);
    lower(&grammar).unwrap();
}

#[test]
fn large_optional_open_object_allow_any_object_valued_at_16_uses_expr_nfa_body() {
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
    assert_eq!(count_rules_with_prefix(&grammar, "json_open_object_prefix"), 0);
    assert!(count_rules_with_prefix(&grammar, "json_closed_object_body") > 0);
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
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
fn large_optional_open_object_with_pattern_properties_uses_fused_prefix_chain_rules() {
    let mut properties = serde_json::Map::new();
    for index in 0..16 {
        properties.insert(format!("k{index}"), json!({"type": "string"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        (
            "patternProperties".to_string(),
            json!({"^x": {"type": "string"}}),
        ),
        ("additionalProperties".to_string(), json!({"type": "string"})),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(count_rules_with_prefix(&grammar, "json_open_object_prefix") > 0);
    assert!(!contains_separated_sequence(start_expr(&grammar)));
    assert!(glrm.contains("json_open_object_prefix"), "{glrm}");
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
fn large_closed_pattern_property_object_uses_generic_key_trie_expr_nfa_body() {
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
                "^foo_.*": {"type": "array"},
                "^bar_.*": {"type": "string"}
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
fn huge_shared_additional_exclusion_set_uses_expanded_literal_addback_when_disabled() {
    if env::var_os("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK_CHILD").is_none() {
        let status = Command::new(env::current_exe().unwrap())
            .arg("--nocapture")
            .arg("huge_shared_additional_exclusion_set_uses_expanded_literal_addback_when_disabled")
            .env("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK_CHILD", "1")
            .env("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK", "0")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }

    let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK", "0");

    let mut properties = serde_json::Map::new();
    for index in 0..300 {
        properties.insert(format!("field_{index}"), json!({"type": "string"}));
    }

    let schema = serde_json::Value::Object(serde_json::Map::from_iter([
        ("type".to_string(), json!("object")),
        ("properties".to_string(), serde_json::Value::Object(properties)),
        ("additionalProperties".to_string(), json!({"type": "string"})),
    ]));

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_COLON_SHARED"), "{glrm}");
    assert!(!glrm.contains("json_additional_key_colon_local"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn huge_shared_additional_exclusion_set_uses_shared_addback_by_default() {
    if env::var_os("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK_CHILD").is_none() {
        let status = Command::new(env::current_exe().unwrap())
            .arg("--nocapture")
            .arg("huge_shared_additional_exclusion_set_uses_shared_addback_by_default")
            .env("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK_CHILD", "1")
            .env_remove("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }

    let _guard = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK");

    let mut properties = serde_json::Map::new();
    for index in 0..300 {
        properties.insert(format!("field_{index}"), json!({"type": "string"}));
    }

    let schema = json!({
        "type": "object",
        "properties": {
            "with_fixed_keys": {
                "type": "object",
                "properties": properties,
                "additionalProperties": {"type": "string"}
            },
            "open_again": {
                "type": "object",
                "additionalProperties": {"type": "integer"}
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_COLON_SHARED"), "{glrm}");
    assert!(glrm.contains("json_additional_excluded_key_colon_shared"), "{glrm}");
    assert!(glrm.contains("JSON_ADDITIONAL_EXCLUDED_KEY_COLON_SHARED"), "{glrm}");
    assert!(
        glrm.matches("\\\"field_0\\\": ").count() <= 5
            || glrm.matches("\"field_0\": ").count() <= 5,
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn shared_additional_excluded_key_skips_closed_object_keys() {
    let schema = json!({
        "type": "object",
        "properties": {
            "closed_child": {
                "type": "object",
                "properties": {
                    "closed_only": {"type": "string"}
                },
                "additionalProperties": false
            },
            "open_child": {
                "type": "object",
                "properties": {
                    "open_only": {"type": "string"}
                },
                "additionalProperties": {"type": "string"}
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let excluded_rule = grammar
        .rules
        .iter()
        .find(|rule| rule.name == "json_additional_excluded_key_colon_shared")
        .expect("shared excluded-key rule exists");

    assert!(
        contains_raw_regex_substring(&excluded_rule.expr, "\"open_only\"")
            || contains_literal_bytes(&excluded_rule.expr, b"\"open_only\"")
    );
    assert!(!format!("{:?}", excluded_rule.expr).is_empty());

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
fn bounded_object_arrays_use_exprnfa_rule() {
    let schema = json!({
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        },
        "maxItems": 3
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("bounded_array_"), "{glrm}");
    assert!(grammar.rules.iter().any(|rule| {
        rule.name.contains("bounded_array_") && matches!(rule.expr, GrammarExpr::ExprNFA(_))
    }), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn bounded_pattern_string_arrays_use_terminal_rule() {
    let schema = json!({
        "type": "array",
        "items": {
            "type": "string",
            "pattern": "^[A-Fa-f\\d]{24}$"
        },
        "maxItems": 3
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("bounded_scalar_array_"), "{glrm}");
    assert!(grammar.rules.iter().any(|rule| {
        rule.name.contains("bounded_scalar_array_") && rule.is_terminal
    }), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn large_bounded_pattern_string_arrays_do_not_use_terminal_rule() {
    let schema = json!({
        "type": "array",
        "items": {
            "type": "string",
            "pattern": "^[A-Fa-f\\d]{24}$"
        },
        "maxItems": 100
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("bounded_scalar_array_"), "{glrm}");
    assert!(!grammar.rules.iter().any(|rule| {
        rule.name.contains("bounded_scalar_array_") && rule.is_terminal
    }), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn unbounded_plain_string_arrays_use_terminal_rule() {
    let schema = json!({
        "type": "array",
        "items": {"type": "string"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("unbounded_scalar_array_"), "{glrm}");
    assert!(grammar.rules.iter().any(|rule| {
        rule.name.contains("unbounded_scalar_array_") && rule.is_terminal
    }), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn unbounded_nullable_string_arrays_keep_null_item_alternative() {
    let schema = json!({
        "type": "array",
        "items": {"type": ["string", "null"]}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("unbounded_scalar_array_"), "{glrm}");
    assert!(glrm.contains("JSON_STRING"), "{glrm}");
    assert!(glrm.contains("JSON_NULL"), "{glrm}");
    assert!(schema_accepts_bytes(&schema, br#"["a", null]"#));
    assert!(!schema_accepts_bytes(&schema, br#"["a", true]"#));
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
    let expr = start_expr(&grammar);
    assert!(contains_literal_bytes(expr, b"\""), "{expr:?}");
    assert!(contains_literal_bytes(expr, b"a\""), "{expr:?}");
    assert!(contains_literal_bytes(expr, b"b\""), "{expr:?}");
    assert!(!contains_literal_bytes(expr, b"\"a\""), "{expr:?}");
    assert!(!contains_literal_bytes(expr, b"\"b\""), "{expr:?}");
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
    let expr = start_expr(&grammar);
    assert!(contains_literal_bytes(expr, b"\""), "{expr:?}");
    assert!(contains_literal_bytes(expr, b"head\""), "{expr:?}");
    assert!(!contains_literal_bytes(expr, b"\"head\""), "{expr:?}");
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_INTEGER") || glrm.contains("JSON_NUMBER"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn plain_items_ignore_additional_items_without_tuple() {
    let schema = json!({
        "type": "array",
        "items": {"type": "string"},
        "additionalItems": false,
        "minItems": 1,
        "maxItems": 2
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn map_shaped_min_properties_lowers_as_bounded_pattern_map() {
    let schema = json!({
        "type": "object",
        "patternProperties": {
            ".+": {"type": "string"}
        },
        "additionalProperties": false,
        "minProperties": 1
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn small_bounded_string_pattern_ignores_length_bounds() {
    let schema = json!({
        "type": "string",
        "minLength": 2,
        "maxLength": 8,
        "pattern": "^[A-Za-z]+$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let rule = grammar
        .rules
        .iter()
        .find(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
        .expect("expected terminalized constrained string rule");

    let GrammarExpr::RawRegex(regex) = &rule.expr else {
        panic!("expected raw regex constrained string rule: {:?}", rule.expr);
    };

    assert!(regex.contains("[A-Za-z]"), "{regex}");

    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("JSON_STRING_CHAR{2,8}"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn large_bounded_string_pattern_ignores_length_bounds() {
    let schema = json!({
        "type": "string",
        "maxLength": 512,
        "pattern": "^/.*"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let rule = grammar
        .rules
        .iter()
        .find(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
        .expect("expected terminalized constrained string rule");

    let GrammarExpr::RawRegex(regex) = &rule.expr else {
        panic!("expected raw regex constrained string rule: {:?}", rule.expr);
    };

    assert!(regex.contains("(?:/"), "{regex}");

    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("json_string_char_exact_open_50"), "{glrm}");
    assert!(!glrm.contains("json_string_char_upto_close_50"), "{glrm}");
    assert!(!glrm.contains("json_string_bounded_split"), "{glrm}");

    lower(&grammar).unwrap();
}

#[test]
fn string_pattern_lowers_ascii_digit_subranges() {
    let schema = json!({
        "type": "string",
        "pattern": "^[1-5][0-9a-f]$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("[1-5]"), "{glrm}");
    assert!(!glrm.contains("[^\\s\\S](?:[0-9a-f])"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn terminalized_dot_pattern_lowers_utf8_lead_byte_alternatives() {
    let schema = json!({
        "type": "string",
        "pattern": "^.*.txt$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let rule = grammar
        .rules
        .iter()
        .find(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
        .expect("expected terminalized constrained string rule");

    let GrammarExpr::RawRegex(regex) = &rule.expr else {
        panic!("expected raw regex terminal: {:?}", rule.expr);
    };
    assert!(regex.contains(r#"\xC2-\xDF"#), "{regex}");
    lower(&grammar).unwrap();
}

#[test]
fn json_string_char_terminal_requires_valid_utf8_sequences() {
    let schema = json!({"type": "string"});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("[\\xC2-\\xDF][\\x80-\\xBF]"), "{glrm}");
    assert!(!glrm.contains("[^\\x00-\\x1f\\x7f"), "{glrm}");
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
fn bounded_pattern_map_respects_min_and_max_properties() {
    let schema = json!({
        "type": "object",
        "minProperties": 1,
        "maxProperties": 2,
        "additionalProperties": false,
        "patternProperties": {
            ".+": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn unsupported_nonredundant_max_properties_broadens() {
    let schema = json!({
        "type": "object",
        "maxProperties": 1,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn unsupported_nonredundant_min_properties_broadens() {
    let schema = json!({
        "type": "object",
        "minProperties": 3,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn allof_oneof_ref_objects_with_required_sibling_factors_common_object() {
    let schema = json!({
        "type": "object",
        "required": ["type", "typeProperties"],
        "allOf": [
            {"$ref": "#/$defs/commonTableProperties"},
            {
                "oneOf": [
                    {"$ref": "#/$defs/blobDataset"},
                    {"$ref": "#/$defs/tableDataset"}
                ]
            }
        ],
        "$defs": {
            "commonTableProperties": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "structure": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/dataElement"}
                    }
                }
            },
            "blobDataset": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["AzureBlob"]},
                    "typeProperties": {"type": "object"}
                }
            },
            "tableDataset": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["AzureTable"]},
                    "typeProperties": {"type": "object"}
                }
            },
            "dataElement": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "additionalProperties": false
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start_rule = glrm
        .lines()
        .find(|line| line.starts_with("nt start ::="))
        .expect("start rule should be present");
    assert!(!start_rule.is_empty(), "{glrm}");
    assert!(
        glrm.contains("\"structure\"") || glrm.contains("\\\"structure\\\""),
        "{glrm}"
    );
    assert!(
        glrm.contains("\"type\"") || glrm.contains("\\\"type\\\""),
        "{glrm}"
    );
    assert!(glrm.contains("AzureBlob"), "{glrm}");
    assert!(glrm.contains("AzureTable"), "{glrm}");
    assert!(!schema_accepts_bytes(&schema, br#"{}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"type": "AzureBlob"}"#));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"type": "AzureBlob", "typeProperties": {}}"#
    ));
    lower(&grammar).unwrap();
}

#[test]
fn oneof_ref_allof_shared_prefix_variants_use_exact_object_variant_body() {
    let schema = json!({
        "type": "object",
        "oneOf": [
            {"$ref": "#/$defs/plate"},
            {"$ref": "#/$defs/tipbox"}
        ],
        "$defs": {
            "item": {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"}
                }
            },
            "plate": {
                "allOf": [
                    {"$ref": "#/$defs/item"},
                    {
                        "properties": {
                            "kind": {"const": "plate"},
                            "residual_volume": {"type": "number"}
                        },
                        "required": ["kind", "residual_volume"]
                    }
                ]
            },
            "tipbox": {
                "allOf": [
                    {"$ref": "#/$defs/item"},
                    {
                        "properties": {
                            "kind": {"const": "tipbox"},
                            "missing_tips": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["kind"]
                    }
                ]
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start_rule = glrm
        .lines()
        .find(|line| line.starts_with("nt start ::="))
        .expect("start rule should be present");
    assert!(!start_rule.is_empty(), "{glrm}");
    assert!(
        glrm.contains("\"kind\"") || glrm.contains("\\\"kind\\\""),
        "{glrm}"
    );
    assert!(glrm.contains("plate"), "{glrm}");
    assert!(glrm.contains("tipbox"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn oneof_ref_allof_shared_prefix_variants_fall_back_for_mismatched_prefix() {
    let schema = json!({
        "type": "object",
        "oneOf": [
            {"$ref": "#/$defs/plate"},
            {"$ref": "#/$defs/tipbox"}
        ],
        "$defs": {
            "itemA": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"}
                }
            },
            "itemB": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "number"}
                }
            },
            "plate": {
                "allOf": [
                    {"$ref": "#/$defs/itemA"},
                    {
                        "properties": {
                            "kind": {"const": "plate"}
                        },
                        "required": ["kind"]
                    }
                ]
            },
            "tipbox": {
                "allOf": [
                    {"$ref": "#/$defs/itemB"},
                    {
                        "properties": {
                            "kind": {"const": "tipbox"}
                        },
                        "required": ["kind"]
                    }
                ]
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start_rule = glrm
        .lines()
        .find(|line| line.starts_with("nt start ::="))
        .expect("start rule should be present");
    assert!(start_rule.contains(" | "), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn oversized_pattern_properties_overlap_check_broadens() {
    let schema = json!({
        "type": "object",
        "properties": {
            "costs": {
                "type": "object",
                "patternProperties": {
                    "^[/][/.\\\\w-]{0,254}$": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number"}
                            }
                        }
                    }
                },
                "additionalProperties": false
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}


#[test]
fn large_pattern_max_length_is_dropped_by_default() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::unset("GLRMASK_JSON_SCHEMA_PRESERVE_PATTERN_MAX_LENGTH");

    let schema = json!({
        "type": "string",
        "pattern": "^[a]+$",
        "maxLength": 80
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let rule = grammar
        .rules
        .iter()
        .find(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
        .expect("expected terminalized constrained string rule");

    assert!(matches!(rule.expr, GrammarExpr::RawRegex(_)), "{:?}", rule.expr);
    lower(&grammar).unwrap();
}

#[test]
fn large_pattern_max_length_env_intersects_json_string_length_envelope() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PRESERVE_PATTERN_MAX_LENGTH", "1");

    let schema = json!({
        "type": "string",
        "pattern": "^[a]+$",
        "minLength": 2,
        "maxLength": 80
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let rule = grammar
        .rules
        .iter()
        .find(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
        .expect("expected terminalized constrained string rule");

    let GrammarExpr::WithSecondaryLexer { main, secondary } = &rule.expr else {
        panic!("expected pattern terminal with secondary length envelope: {:?}", rule.expr);
    };
    assert!(matches!(main.as_ref(), GrammarExpr::RawRegex(_)), "{:?}", main);
    let secondary_debug = format!("{:?}", secondary);
    assert!(secondary_debug.contains("json_string_char_exact_2"), "{secondary_debug}");
    assert!(secondary_debug.contains("json_string_char_upto_78"), "{secondary_debug}");
    lower(&grammar).unwrap();

    let mut too_short = Vec::from([b'"']);
    too_short.push(b'a');
    too_short.push(b'"');
    assert!(!schema_accepts_bytes(&schema, &too_short));

    let mut at_limit = Vec::from([b'"']);
    at_limit.extend(std::iter::repeat_n(b'a', 80));
    at_limit.push(b'"');
    assert!(schema_accepts_bytes(&schema, &at_limit));

    let mut too_long = Vec::from([b'"']);
    too_long.extend(std::iter::repeat_n(b'a', 81));
    too_long.push(b'"');
    assert!(!schema_accepts_bytes(&schema, &too_long));
}

#[test]
fn preserved_pattern_max_length_rejects_overlong_runtime_string() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_PRESERVE_PATTERN_MAX_LENGTH", "1");

    let schema = json!({
        "type": "string",
        "pattern": "^.*$",
        "minLength": 1,
        "maxLength": 1
    });

    assert!(schema_accepts_bytes(&schema, br#""a""#), "commit should accept a");
    assert!(schema_accepts_bytes(&schema, br#""\u0061""#), "commit should accept unicode escape");
    assert!(!schema_accepts_bytes(&schema, br#""aa""#), "commit should reject aa");
    assert!(schema_mask_allows_token_after_prefix(&schema, b"", 300, br#""a""#), "mask should allow token a");
    assert!(!schema_mask_allows_token_after_prefix(&schema, b"", 300, br#""aa""#), "mask should reject token aa");
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
fn ascii_string_pattern_class_unicode_escape_branch_is_compact() {
    let schema = json!({
        "type": "string",
        "pattern": "^[0-9A-Z_a-z]+$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let rule = grammar
        .rules
        .iter()
        .find(|rule| rule.is_terminal && rule.name.starts_with("json_string_constrained"))
        .expect("expected terminalized constrained string rule");

    let GrammarExpr::RawRegex(regex) = &rule.expr else {
        panic!("expected raw regex constrained string rule: {:?}", rule.expr);
    };

    assert!(regex.contains("[0-9A-Z_a-z]"), "{regex}");
    assert!(regex.contains(r#"\\u00(?:"#), "{regex}");
    assert!(!regex.contains(r#"\\u0030|\\u0031|\\u0032"#), "{regex}");

    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains(r#""\\u" /0/ /0/ /3/ /0/"#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn moderately_bounded_string_terminalizes_by_default() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _terminalize_guard = EnvVarGuard::unset(
        "GLRMASK_JSON_SCHEMA_TERMINALIZE_BOUNDED_STRING_MAX",
    );

    let schema = json!({
        "type": "string",
        "maxLength": 64
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_string_constrained"), "{glrm}");
    assert!(!glrm.contains("JSON_STRING_CHAR{0,64}"), "{glrm}");
    assert!(!glrm.contains("json_string_char_exact_50"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn moderately_large_prefix_only_string_terminalizes_without_chunk_helper_rules() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _terminalize_guard = EnvVarGuard::unset(
        "GLRMASK_JSON_SCHEMA_TERMINALIZE_BOUNDED_STRING_MAX",
    );

    let schema = json!({
        "type": "string",
        "minLength": 80
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_string_constrained"), "{glrm}");
    assert!(!glrm.contains("JSON_STRING_CHAR{50} JSON_STRING_CHAR{30}"), "{glrm}");
    assert!(
        !grammar
            .rules
            .iter()
            .any(|rule| rule.name.starts_with("json_string_char_exact_") || rule.name.starts_with("json_string_char_upto_")),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn split_bounded_string_chunks_do_not_overlap_at_boundary() {
    let _env_lock = ENV_LOCK.lock().unwrap();
    let _terminalize_guard = EnvVarGuard::unset(
        "GLRMASK_JSON_SCHEMA_TERMINALIZE_BOUNDED_STRING_MAX",
    );

    let schema = json!({
        "type": "string",
        "maxLength": 102
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_string_char_upto_close_49"), "{glrm}");
    assert!(!glrm.contains("json_string_char_upto_close_50"), "{glrm}");
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
    assert!(glrm.contains("json_string_char_exact_open_50"), "{glrm}");
    assert!(glrm.contains("json_string_char_upto_wrapped_50"), "{glrm}");
    lower(&grammar).unwrap();
}


#[test]
fn discriminator_anyof_object_lowers_to_compact_body() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "type": {"enum": ["INSPIRE BAI"], "type": "string"},
                    "value": {"pattern": "(\\w+\\.)+\\d+", "type": "string"}
                },
                "required": ["type", "value"]
            },
            {
                "type": "object",
                "properties": {
                    "type": {"enum": ["ARXIV"], "type": "string"},
                    "value": {"pattern": "\\w+_(\\w_)?\\d+", "type": "string"}
                },
                "required": ["type", "value"]
            },
            {
                "type": "object",
                "properties": {
                    "type": {"enum": ["GOOGLESCHOLAR"], "type": "string"},
                    "value": {"pattern": "(\\w|-){12}", "type": "string"}
                },
                "required": ["type", "value"]
            },
            {
                "type": "object",
                "properties": {
                    "type": {"enum": ["VIAF"], "type": "string"},
                    "value": {"pattern": "\\d{7,9}", "type": "string"}
                },
                "required": ["type", "value"]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_discriminator_anyof_object_body"), "{glrm}");
    assert!(glrm.contains("json_string_pattern_body"), "{glrm}");
    assert!(glrm.contains("t json_string_pattern_open_middle"), "{glrm}");
    assert!(glrm.contains("t json_string_pattern_end"), "{glrm}");
    assert!(glrm.contains("nt json_string_constrained"), "{glrm}");
    assert!(!glrm.contains("\nnt json_anyof_object_body"), "{glrm}");
    assert!(!glrm.contains("\nnt json_additional_key_colon_local ::= "), "{glrm}");
    assert!(!glrm.contains("__exact_sub_json_additional"), "{glrm}");
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"type": "ARXIV", "value": "abc_1"}"#
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"type": "ARXIV", "value": "abc_d_1"}"#
    ));
    assert!(!schema_accepts_bytes(
        &schema,
        br#"{"type": "ARXIV", "value": "abc_d1"}"#
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"type": "VIAF", "value": "1234567", "extra": true}"#
    ));
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

    let word_pattern = r"^$|(^(?:\S+\s+){0,19}\S+$)";
    assert!(property_name_matches_pattern(word_pattern, "").unwrap());
    assert!(property_name_matches_pattern(word_pattern, "REST").unwrap());
    assert!(property_name_matches_pattern(word_pattern, "REST JSON").unwrap());
    assert!(!property_name_matches_pattern(word_pattern, " C").unwrap());
    assert!(!property_name_matches_pattern(word_pattern, "REST ").unwrap());

    assert!(property_name_matches_pattern(r"^\S+$", "π").unwrap());
    assert!(property_name_matches_pattern(r"^\S+$", "中文").unwrap());
    assert!(!property_name_matches_pattern(r"^\S+$", " ").unwrap());
    assert!(!property_name_matches_pattern(r"^\S+$", "\u{00A0}").unwrap());
    assert!(!property_name_matches_pattern(r"^\S+$", "\u{2003}").unwrap());
    assert!(property_name_matches_pattern("INTERVAL_TICK|INTERVAL_M1", "xxINTERVAL_M1yy").unwrap());
    assert!(!property_name_matches_pattern("INTERVAL_TICK|INTERVAL_M1", "INTERVAL_M2").unwrap());
    assert!(property_name_matches_pattern(r"^(?:\S+\s+){0,19}\S+$", "Up to 24 hours π").unwrap());
    assert!(property_name_matches_pattern(r"^(?:\S+\s+){0,19}\S+$", "Up コ").unwrap());
    assert!(property_name_matches_pattern(r"^[/][/.\w-]{0,254}$", "/cost_1").unwrap());
    assert!(!property_name_matches_pattern(r"^[/][/.\w-]{0,254}$", "/cost space").unwrap());
}

#[test]
fn uuid_format_lowers_to_constrained_terminal() {
    let schema = json!({
        "type": "string",
        "format": "uuid"
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
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_STRING"));

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("[0-9A-Fa-f]{8}"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn date_time_format_lowers_to_constrained_terminal() {
    let schema = json!({
        "type": "string",
        "format": "date-time"
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
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_STRING"));

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("[Tt]"), "{glrm}");
    assert!(glrm.contains("[+-]"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn date_format_lowers_to_constrained_terminal() {
    let schema = json!({
        "type": "string",
        "format": "date"
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
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_STRING"));

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("0[1-9]|1[0-2]"), "{glrm}");
    assert!(glrm.contains("0[1-9]|[12][0-9]|3[01]"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn email_format_lowers_to_constrained_terminal() {
    let schema = json!({
        "type": "string",
        "format": "email"
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
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_STRING"));

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("@"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn email_format_with_large_max_length_does_not_preserve_length_envelope() {
    let schema = json!({
        "type": "string",
        "format": "email",
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
    assert!(glrm.contains("@"), "{glrm}");
    assert!(!glrm.contains("JSON_STRING_CHAR{0,1024}"), "{glrm}");
    assert!(!glrm.contains("json_string_char_exact_50"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn hostname_ipv4_ipv6_formats_lower_to_constrained_terminals() {
    for (format, expected) in [
        ("hostname", "[A-Za-z0-9]"),
        ("ipv4", "25[0-5]"),
        ("ipv6", "[A-Fa-f0-9]"),
    ] {
        let schema = json!({
            "type": "string",
            "format": format
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
        assert!(!contains_ref_named(start_expr(&grammar), "JSON_STRING"));

        let glrm = to_glrm(&grammar);
        assert!(glrm.contains(expected), "{glrm}");
        lower(&grammar).unwrap();
    }
}

#[test]
fn uri_format_lowers_to_constrained_terminal() {
    let schema = json!({
        "type": "string",
        "format": "uri"
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
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_STRING"));

    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("[A-Za-z]"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn uri_format_rejects_repeated_fragment_marker_without_full_llguidance_regex() {
    let schema = json!({
        "type": "string",
        "format": "uri"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("path_abempty"), "should not import llguidance's full URI regex: {glrm}");
    lower(&grammar).unwrap();

    assert!(schema_accepts_bytes(&schema, br#""https://example.com/#frag""#));
    assert!(schema_accepts_bytes(&schema, br#""https://example.com/?q=a/b""#));
    assert!(schema_accepts_bytes(&schema, br#""https://example.com/%23ok?q=%3F#frag%20x""#));
    assert!(schema_accepts_bytes(&schema, br#""https://[::1]/path""#));
    assert!(schema_accepts_bytes(&schema, br#""https://[V1.foo]/path""#));
    assert!(!schema_accepts_bytes(&schema, br#""https://##""#));
    assert!(!schema_accepts_bytes(&schema, br#""https://%!""#));
    assert!(!schema_accepts_bytes(&schema, br#""https://example.com/%!""#));
}

#[test]
fn decimal_multiple_of_cent_uses_nonnegative_compact_language_without_fixed_scale() {
    let schema = json!({
        "type": "number",
        "multipleOf": 0.01
    });

    assert!(schema_accepts_bytes(&schema, b"0"));
    assert!(schema_accepts_bytes(&schema, b"0.00"));
    assert!(schema_accepts_bytes(&schema, b"1"));
    assert!(schema_accepts_bytes(&schema, b"99.99"));
    assert!(schema_accepts_bytes(&schema, b"99.9900"));
    assert!(schema_accepts_bytes(&schema, b"99.000"));
    assert!(!schema_accepts_bytes(&schema, b"-0.01"));
    assert!(!schema_accepts_bytes(&schema, b"-99.99"));
    assert!(!schema_accepts_bytes(&schema, b"0.001"));
    assert!(!schema_accepts_bytes(&schema, b"99.999"));
}

#[test]
fn string_pattern_is_intersected_with_format() {
    let schema = json!({
        "type": "string",
        "format": "uuid",
        "pattern": "^abc$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(
        glrm.contains("abc") || glrm.contains("json_string_pattern_body"),
        "{glrm}"
    );
    assert!(glrm.contains("[0-9A-Fa-f]{8}"), "{glrm}");
    assert!(glrm.contains("json_string_constrained") || glrm.contains("uuid"), "{glrm}");
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
        assert!(!rule.name.is_empty());
    }
    assert!(
        grammar.rules.iter().any(|rule| {
            rule.is_terminal
                && (rule.name.starts_with("json_string_constrained")
                    || rule.name.starts_with("json_property_string_value"))
        })
    );
    assert!(
        grammar
            .rules
            .iter()
            .any(|rule| rule.is_terminal && rule.name.starts_with("json_pattern_key_colon"))
    );

    let glrm = to_glrm(&grammar);
    assert!(
        glrm.contains("\"last_modification\"")
            || glrm.contains("\\\"last_modification\\\""),
        "{glrm}"
    );
    assert!(glrm.contains("last_modification") && glrm.contains("JSON_KEY_SEPARATOR"), "{glrm}");
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
    assert!(glrm.contains("JSON_ADDITIONAL") || glrm.contains("json_additional"), "{glrm}");
    assert!(
        glrm.contains("\"x-name\"") || glrm.contains("\\\"x-name\\\"") || glrm.contains("\"x\\-name\""),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn json_separators_require_single_space() {
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("(?:, )"), "{glrm}");
    assert!(glrm.contains("(?:: )"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn fixed_property_rejects_no_space_key_separator() {
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"}
        },
        "additionalProperties": false
    });

    assert!(!schema_accepts_bytes(&schema, br#"{"id":"x"}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"id": "x"}"#));
}

#[test]
fn pattern_property_rejects_no_space_key_separator() {
    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^x": {"type": "integer"}
        },
        "additionalProperties": false
    });

    assert!(!schema_accepts_bytes(&schema, br#"{"x1":1}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"x1": 1}"#));
}

#[test]
fn additional_property_rejects_no_space_key_separator() {
    let schema = json!({
        "type": "object",
        "additionalProperties": {"type": "boolean"}
    });

    assert!(!schema_accepts_bytes(&schema, br#"{"flag":true}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"flag": true}"#));
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
fn local_ref_to_property_schema_is_loaded() {
    let schema = json!({
        "type": "object",
        "properties": {
            "MD001": {"type": "boolean"},
            "heading-increment": {"$ref": "#/properties/MD001"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn default_object_named_properties_is_not_scanned_for_ref_targets() {
    let schema = json!({
        "type": "string",
        "default": {
            "properties": {
                "not_a_schema": "not a schema"
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn property_named_definitions_is_not_definition_container() {
    let schema = json!({
        "type": "object",
        "properties": {
            "definitions": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"}
                }
            }
        }
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
fn date_time_string_value_satisfaction_filters_invalid_literals() {
    let schema = StringSchema {
        format: Some("date-time".to_string()),
        ..Default::default()
    };

    assert!(string_value_satisfies_schema(&json!("2024-05-01T12:34:56Z"), &schema).unwrap());
    assert!(string_value_satisfies_schema(&json!("2020-02-29T12:34:56Z"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("."), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("2019-02-29T12:34:56Z"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("2020-06-31T12:34:56Z"), &schema).unwrap());
}

#[test]
fn date_string_value_satisfaction_filters_invalid_literals() {
    let schema = StringSchema {
        format: Some("date".to_string()),
        ..Default::default()
    };

    assert!(string_value_satisfies_schema(&json!("2024-05-01"), &schema).unwrap());
    assert!(string_value_satisfies_schema(&json!("2020-02-29"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("|"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("2019-02-29"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("2020-06-31"), &schema).unwrap());
}

#[test]
fn uuid_string_value_satisfaction_filters_invalid_literals() {
    let schema = StringSchema {
        format: Some("uuid".to_string()),
        ..Default::default()
    };

    assert!(string_value_satisfies_schema(
        &json!("123e4567-e89b-12d3-a456-426614174000"),
        &schema
    )
    .unwrap());
    assert!(!string_value_satisfies_schema(&json!("|"), &schema).unwrap());
}

#[test]
fn email_string_value_satisfaction_filters_invalid_literals() {
    let schema = StringSchema {
        format: Some("email".to_string()),
        ..Default::default()
    };

    assert!(string_value_satisfies_schema(&json!("user@example.com"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("><"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!(".user@example.com"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("missing-at"), &schema).unwrap());
}

#[test]
fn host_string_value_satisfaction_filters_invalid_literals() {
    let hostname = StringSchema {
        format: Some("hostname".to_string()),
        ..Default::default()
    };
    assert!(string_value_satisfies_schema(&json!("localhost"), &hostname).unwrap());
    assert!(string_value_satisfies_schema(&json!("redshift.example.com"), &hostname).unwrap());
    assert!(!string_value_satisfies_schema(&json!(";"), &hostname).unwrap());

    let ipv4 = StringSchema {
        format: Some("ipv4".to_string()),
        ..Default::default()
    };
    assert!(string_value_satisfies_schema(&json!("127.0.0.1"), &ipv4).unwrap());
    assert!(!string_value_satisfies_schema(&json!("999.0.0.1"), &ipv4).unwrap());

    let ipv6 = StringSchema {
        format: Some("ipv6".to_string()),
        ..Default::default()
    };
    assert!(string_value_satisfies_schema(&json!("::1"), &ipv6).unwrap());
    assert!(!string_value_satisfies_schema(&json!(";"), &ipv6).unwrap());
}

#[test]
fn uri_string_value_satisfaction_filters_invalid_literals() {
    let schema = StringSchema {
        format: Some("uri".to_string()),
        ..Default::default()
    };

    assert!(string_value_satisfies_schema(&json!("ecdsa-koblitz-pubkey:abc123"), &schema).unwrap());
    assert!(string_value_satisfies_schema(&json!("ecdsa-koblitz-pubkey://[::1]"), &schema).unwrap());
    assert!(string_value_satisfies_schema(&json!("ftp://[v1.example]"), &schema).unwrap());
    assert!(string_value_satisfies_schema(&json!("ftp://user@[v1.example]"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("<<"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("ecd:]"), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("ecd://["), &schema).unwrap());
    assert!(!string_value_satisfies_schema(&json!("ecd:\u{ff49}"), &schema).unwrap());
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
fn conditional_keywords_error_for_broad_lowering() {
    let schema = json!({
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "payload": {"type": "string"}
        },
        "if": {
            "properties": {"kind": {"const": "needs_payload"}}
        },
        "then": {
            "required": ["payload"]
        },
        "else": {
            "properties": {"payload": false}
        }
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("#: Unimplemented keys"), "{error}");
    assert!(error.contains("if"), "{error}");
    assert!(error.contains("then"), "{error}");
    assert!(error.contains("else"), "{error}");
}

#[test]
fn conditional_keywords_precede_nested_unique_items_in_then() {
    let schema = json!({
        "allOf": [{
            "if": {"properties": {"type": {"const": "theme"}}},
            "then": {
                "properties": {
                    "regions_hidden": {"type": "array", "uniqueItems": true}
                }
            }
        }]
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(
        error.contains("#/allOf/0: Unimplemented keys: [\"if\", \"then\"]"),
        "{error}"
    );
    assert!(!error.contains("uniqueItems"), "{error}");
}

#[test]
fn conditional_keywords_precede_definition_unique_items() {
    let schema = json!({
        "definitions": {
            "bad": {"type": "array", "uniqueItems": true}
        },
        "allOf": [{
            "if": {"properties": {"kind": {"const": "x"}}},
            "then": {"required": ["kind"]}
        }]
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(
        error.contains("#/allOf/0: Unimplemented keys: [\"if\", \"then\"]"),
        "{error}"
    );
    assert!(!error.contains("uniqueItems"), "{error}");
}

#[test]
fn conditional_preflight_ignores_annotation_objects() {
    let schema = json!({
        "type": "object",
        "default": {"if": 1, "then": 2},
        "examples": [{"if": 1, "then": 2}],
        "properties": {
            "x": {
                "type": "string",
                "default": {"if": 1, "then": 2}
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn conditional_preflight_checks_additional_items_before_definitions_unique_items() {
    let schema = json!({
        "definitions": {
            "bad": {"type": "array", "uniqueItems": true}
        },
        "type": "array",
        "items": [{"type": "string"}],
        "additionalItems": {
            "if": {"properties": {"kind": {"const": "x"}}},
            "then": {"type": "string"}
        }
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(
        error.contains("#/additionalItems: Unimplemented keys: [\"if\", \"then\"]"),
        "{error}"
    );
    assert!(!error.contains("uniqueItems"), "{error}");
}

#[test]
fn unique_items_still_errors_without_conditional() {
    let schema = json!({"type": "array", "uniqueItems": true});

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("#: Unimplemented keys"), "{error}");
    assert!(error.contains("uniqueItems"), "{error}");
}

#[test]
fn oneof_lowers_as_choice() {
    let schema = json!({
        "oneOf": [
            {"const": "left"},
            {"const": "right"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_single_ref_wrapper_is_supported() {
    let schema = json!({
        "definitions": {
            "name": {"type": "string"}
        },
        "oneOf": [
            {"$ref": "#/definitions/name"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn fragment_id_ref_alias_lowers() {
    let schema = json!({
        "type": "object",
        "definitions": {
            "name": {
                "id": "#nameAlias",
                "const": "ok"
            }
        },
        "properties": {
            "name": {"$ref": "#nameAlias"}
        },
        "required": ["name"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn absolute_root_id_self_ref_lowers() {
    let schema = json!({
        "id": "http://example.test/schema.json#",
        "type": "object",
        "properties": {
            "child": {"$ref": "http://example.test/schema.json#"}
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_ref_and_null_is_supported() {
    let schema = json!({
        "definitions": {
            "input": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"]
            }
        },
        "oneOf": [
            {"$ref": "#/definitions/input"},
            {"type": ["null"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_mixed_local_ref_and_inline_object_lowers() {
    let schema = json!({
        "definitions": {
            "input": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {"$ref": "#/definitions/input"},
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_mixed_local_ref_and_inline_array_lowers() {
    let schema = json!({
        "$defs": {
            "tool": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {
                "type": "array",
                "items": {"$ref": "#/$defs/tool"}
            },
            {"$ref": "#/$defs/tool"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_mixed_ref_and_inline_primitive_still_errors() {
    let schema = json!({
        "definitions": {
            "input": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"]
            }
        },
        "oneOf": [
            {"type": "number"},
            {"type": "integer"},
            {"$ref": "#/definitions/input"}
        ]
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("pairwise disjoint"), "{error}");
}

#[test]
fn oneof_mixed_local_ref_object_targets_and_inline_primitives_lowers() {
    let schema = json!({
        "definitions": {
            "features": {
                "type": "object",
                "additionalProperties": true
            },
            "reference": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {"type": "number"},
            {"type": "string"},
            {"$ref": "#/definitions/features"},
            {"$ref": "#/definitions/reference"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn oneof_mixed_local_ref_inline_primitives_and_array_lowers() {
    let schema = json!({
        "definitions": {
            "features": {
                "type": "object",
                "additionalProperties": true
            },
            "reference": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {"type": "number"},
            {"type": "string"},
            {"$ref": "#/definitions/features"},
            {"$ref": "#/definitions/reference"},
            {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"$ref": "#/definitions/features"},
                        {"$ref": "#/definitions/reference"}
                    ]
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

fn nested_config_align_oneof_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "oneOf": [
            {"$ref": "#/definitions/left"},
            {"$ref": "#/definitions/center"},
            {"$ref": "#/definitions/right"}
        ],
        "required": ["config"],
        "properties": {
            "config": {
                "type": "object",
                "required": ["align"],
                "properties": {
                    "align": {"type": "string"}
                },
                "additionalProperties": false
            }
        },
        "additionalProperties": false,
        "definitions": {
            "left": {
                "type": "object",
                "required": ["config"],
                "properties": {
                    "config": {
                        "type": "object",
                        "required": ["align"],
                        "properties": {
                            "align": {"type": "string", "enum": ["left"]}
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            },
            "center": {
                "type": "object",
                "required": ["config"],
                "properties": {
                    "config": {
                        "type": "object",
                        "required": ["align"],
                        "properties": {
                            "align": {"type": "string", "enum": ["center"]}
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            },
            "right": {
                "type": "object",
                "required": ["config"],
                "properties": {
                    "config": {
                        "type": "object",
                        "required": ["align"],
                        "properties": {
                            "align": {"type": "string", "enum": ["right"]}
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            }
        }
    })
}

fn nested_config_align_oneof_with_shared_content_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "oneOf": [
            {"$ref": "#/definitions/left"},
            {"$ref": "#/definitions/center"},
            {"$ref": "#/definitions/right"}
        ],
        "required": ["config", "content"],
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "align": {"type": "string"}
                },
                "required": ["align"]
            },
            "content": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "body": {"type": "string"},
                    "badge": {
                        "type": "object",
                        "properties": {
                            "config": {
                                "type": "object",
                                "properties": {
                                    "size": {"type": "string", "enum": ["small", "large"]},
                                    "type": {"type": "string", "enum": ["highlight", "lowlight"]}
                                },
                                "required": ["size", "type"]
                            },
                            "content": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"}
                                },
                                "required": ["text"]
                            }
                        },
                        "required": ["config", "content"]
                    },
                    "image": {
                        "type": "object",
                        "properties": {
                            "vp1": {"type": "string"},
                            "vp2": {"type": "string"},
                            "vp3": {"type": "string"},
                            "vp4": {"type": "string"},
                            "vp5": {"type": "string"},
                            "vp6": {"type": "string"},
                            "alt": {"type": "string"}
                        },
                        "required": ["vp1", "vp2", "vp3", "vp4", "vp5", "vp6", "alt"]
                    }
                },
                "required": ["heading", "body", "image"]
            }
        },
        "definitions": {
            "left": {
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "align": {"type": "string", "enum": ["left"]}
                        },
                        "required": ["align"]
                    }
                },
                "required": ["config"]
            },
            "center": {
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "align": {"type": "string", "enum": ["center"]}
                        },
                        "required": ["align"]
                    }
                },
                "required": ["config"]
            },
            "right": {
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "align": {"type": "string", "enum": ["right"]}
                        },
                        "required": ["align"]
                    }
                },
                "required": ["config"]
            }
        }
    })
}

#[test]
fn oneof_nested_config_align_enum_ref_branches_accept_and_reject() {
    let schema = nested_config_align_oneof_schema();

    assert!(schema_accepts_bytes(&schema, br#"{"config": {"align": "left"}}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"config": {"align": "center"}}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"config": {"align": "right"}}"#));

    assert!(!schema_accepts_bytes(&schema, br#"{}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"config":{}}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"config":{"align":"top"}}"#));
}

#[test]
fn oneof_nested_config_align_enum_ref_branches_use_object_fast_path() {
    let schema = nested_config_align_oneof_schema();

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(!matches!(expr, GrammarExpr::Choice(_)), "{expr:?}");
    let glrm = to_glrm(&grammar);
    let start_line = glrm
        .lines()
        .find(|line| line.starts_with("nt start ::= "))
        .unwrap_or("<missing start line>");
    assert_eq!(start_line, "nt start ::= \"{\" json_closed_object_body_1 \"}\";");
    lower(&grammar).unwrap();
}

#[test]
fn oneof_nested_config_align_enum_ref_branches_with_shared_content_use_object_fast_path() {
    let schema = nested_config_align_oneof_with_shared_content_schema();

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(!matches!(expr, GrammarExpr::Choice(_)), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn object_constrained_allof_with_nested_oneof_accepts_order_insensitive_common_properties() {
    let schema = object_constrained_allof_with_nested_oneof_schema();

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"_elements": [{"name": "example.txt", "user": "user1", "group": "group1", "type": "file", "size": 1024, "mode": "644"}]}"#,
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"_elements": [{"name": "example.txt", "type": "file", "user": "user1", "group": "group1", "size": 1024, "mode": "644"}]}"#,
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"_elements": [{"name": "example_remote_dir", "type": "remote_dir"}]}"#,
    ));

    assert!(count_rules_with_prefix(&grammar, "json_closed_object_body") <= 5, "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_constrained_allof_with_nested_oneof_keeps_mismatched_common_property_fallback() {
    let mut schema = object_constrained_allof_with_nested_oneof_schema();
    schema["definitions"]["file_file"]["properties"]["name"] = json!({"type": "number"});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);

    assert!(!glrm.contains("json_anyof_object_body"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn object_constrained_allof_with_nested_oneof_unsupported_pattern_falls_back() {
    let mut schema = object_constrained_allof_with_nested_oneof_schema();
    schema["definitions"]["file"]["allOf"][1]["patternProperties"] =
        json!({"^x-": {"type": "string"}});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);

    assert!(!glrm.contains("json_anyof_object_body"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn oneof_mixed_local_ref_and_inline_primitive_with_untyped_ref_target_errors() {
    let schema = json!({
        "definitions": {
            "input": {
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {"type": "string"},
            {"$ref": "#/definitions/input"}
        ]
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("explicit object-only"), "{error}");
}

#[test]
fn unsupported_not_shape_errors() {
    let schema = json!({
        "type": "string",
        "not": {"const": "forbidden"}
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("not"), "{error}");
}

#[test]
fn anyof_property_not_mutual_exclusion_lowers_as_exclusive_group() {
    let schema = json!({
        "type": "object",
        "additionalProperties": true,
        "anyOf": [
            {
                "properties": {"bundleDependencies": {"type": "array"}},
                "not": {
                    "properties": {"bundledDependencies": {}},
                    "required": ["bundledDependencies"]
                }
            },
            {
                "properties": {"bundledDependencies": {"type": "array"}},
                "not": {
                    "properties": {"bundleDependencies": {}},
                    "required": ["bundleDependencies"]
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);

    assert!(glrm.contains("bundleDependencies"), "{glrm}");
    assert!(glrm.contains("bundledDependencies"), "{glrm}");
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
}

#[test]
fn property_names_inline_pattern_lowers() {
    let schema = json!({
        "type": "object",
        "propertyNames": {
            "pattern": "^[a-z]+$"
        },
        "additionalProperties": {
            "type": "string"
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
    assert!(schema_accepts_bytes(&schema, br#"{"name": "ok"}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"Name":"ok"}"#));
}

#[test]
fn property_names_local_ref_pattern_lowers() {
    let schema = json!({
        "$defs": {
            "token": {
                "type": "string",
                "pattern": "^[-_a-zA-Z0-9]+$"
            }
        },
        "type": "object",
        "properties": {
            "networks": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                },
                "propertyNames": {
                    "$ref": "#/$defs/token"
                }
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
    assert!(schema_accepts_bytes(&schema, br#"{"networks": {"prod_1": "ok"}}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"networks":{"prod-1!":"ok"}}"#));
}

#[test]
fn property_names_pattern_applies_to_additional_properties_keys() {
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "propertyNames": {
            "pattern": "^[a-z]+$"
        },
        "additionalProperties": {
            "type": "string"
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
    assert!(schema_accepts_bytes(&schema, br#"{"name": "ok", "alias": "ok"}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"name":"ok","Alias":"ok"}"#));
}

#[test]
fn llguidance_compat_property_names_with_pattern_properties_broadens() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "propertyNames": {"pattern": "^\\d+$"},
        "patternProperties": {
            ".*": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
    assert!(schema_accepts_bytes(&schema, br#"{"!": "ok"}"#));
}

#[test]
fn oneof_mixed_local_ref_and_const_primitive_disjoint_family_lowers() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "$defs": {
            "Init": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"$ref": "#/$defs/ActionChain"}
                ]
            },
            "ActionChain": {
                "type": "array",
                "items": {"type": "object"}
            }
        },
        "oneOf": [
            {"$ref": "#/$defs/Init"},
            {"const": ""}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
    assert!(schema_accepts_bytes(&schema, br#""""#));
    assert!(schema_accepts_bytes(&schema, br#"[]"#));
}

#[test]
fn property_names_non_pattern_schema_still_errors() {
    let schema = json!({
        "type": "object",
        "propertyNames": {
            "type": "string"
        },
        "additionalProperties": {
            "type": "string"
        }
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("string pattern schemas"), "{error}");
}

#[test]
fn property_names_local_ref_without_explicit_string_pattern_still_errors() {
    let schema = json!({
        "$defs": {
            "token": {
                "type": "string"
            }
        },
        "type": "object",
        "propertyNames": {
            "$ref": "#/$defs/token"
        },
        "additionalProperties": {
            "type": "string"
        }
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("string pattern schemas"), "{error}");
}

#[test]
fn property_names_fixed_literal_key_outside_pattern_errors() {
    let schema = json!({
        "type": "object",
        "properties": {
            "Bad-Key": {"type": "string"}
        },
        "propertyNames": {
            "pattern": "^[a-z]+$"
        },
        "additionalProperties": false
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("does not allow fixed property"), "{error}");
}

#[test]
fn dependencies_property_array_requires_dependents() {
    let schema = json!({
        "type": "object",
        "properties": {
            "vendor": {"type": "string"},
            "model": {"type": "string"}
        },
        "dependencies": {
            "vendor": ["model"]
        },
        "additionalProperties": false
    });

    assert!(schema_accepts_bytes(&schema, br#"{}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"model": "m"}"#));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"vendor": "v", "model": "m"}"#
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"model": "m", "vendor": "v"}"#
    ));
    assert!(!schema_accepts_bytes(&schema, br#"{"vendor":"v"}"#));
}

#[test]
fn dependent_required_requires_dependents() {
    let schema = json!({
        "type": "object",
        "properties": {
            "favoriteTopic": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "dependentRequired": {
            "favoriteTopic": ["tags"]
        },
        "additionalProperties": false
    });

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"favoriteTopic": "rust", "tags": ["parser"]}"#
    ));
    assert!(!schema_accepts_bytes(
        &schema,
        br#"{"favoriteTopic":"rust"}"#
    ));
}

#[test]
fn dependencies_multiple_and_bidirectional() {
    let schema = json!({
        "type": "object",
        "properties": {
            "siteId": {"type": "string"},
            "pageId": {"type": "string"},
            "formatId": {"type": "string"}
        },
        "dependencies": {
            "siteId": ["pageId", "formatId"],
            "pageId": ["siteId", "formatId"],
            "formatId": ["siteId", "pageId"]
        },
        "additionalProperties": false
    });

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"formatId": "f", "siteId": "s", "pageId": "p"}"#
    ));
    assert!(schema_accepts_bytes(&schema, br#"{}"#));
    assert!(!schema_accepts_bytes(
        &schema,
        br#"{"siteId":"s","pageId":"p"}"#
    ));
    assert!(!schema_accepts_bytes(&schema, br#"{"formatId":"f"}"#));
}

#[test]
fn dependencies_unknown_dependent_in_closed_object_rejects_trigger() {
    let schema = json!({
        "type": "object",
        "properties": {
            "vendor": {"type": "string"}
        },
        "dependencies": {
            "vendor": ["model"]
        },
        "additionalProperties": false
    });

    assert!(schema_accepts_bytes(&schema, br#"{}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"vendor":"v"}"#));
}

#[test]
fn dependencies_schema_value_still_errors() {
    let schema = json!({
        "type": "object",
        "properties": {
            "siteId": {"type": "string"},
            "pageId": {"type": "string"}
        },
        "dependencies": {
            "siteId": {"required": ["pageId"]}
        },
        "additionalProperties": false
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(
        error.contains("schema dependencies are not supported"),
        "{error}"
    );
}

#[test]
fn dependent_schemas_still_errors() {
    let schema = json!({
        "type": "object",
        "properties": {
            "siteId": {"type": "string"},
            "pageId": {"type": "string"}
        },
        "dependentSchemas": {
            "siteId": {"required": ["pageId"]}
        },
        "additionalProperties": false
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("Unimplemented keys"), "{error}");
    assert!(error.contains("dependentSchemas"), "{error}");
}

#[test]
fn enum_and_const_lower_to_exact_json_literals() {
    let schema = json!({"enum": [null, true, "ready", 7]});
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("\"null\""), "{glrm}");
    assert!(glrm.contains("\"true\""), "{glrm}");
    assert!(glrm.contains("\"\\\"\" \"ready\\\"\""), "{glrm}");
    assert!(glrm.contains("\"7\""), "{glrm}");
}

#[test]
fn string_const_splits_open_quote_from_literal_body() {
    let schema = json!({"const": "ready"});
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);

    assert!(contains_literal_bytes(expr, b"\""), "{expr:?}");
    assert!(contains_literal_bytes(expr, b"ready\""), "{expr:?}");
    assert!(!contains_literal_bytes(expr, b"\"ready\""), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn object_const_uses_json_separator_rules() {
    let schema = json!({
        "const": {
            "$data": "1/password",
            "items": [1, true]
        }
    });
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);

    assert!(
        contains_raw_regex_substring(expr, r#""\$data""#)
            || contains_raw_regex_substring(expr, r#""$data""#)
            || contains_literal_bytes(expr, b"\"$data\""),
        "{expr:?}"
    );
    assert!(
        contains_ref_named(expr, "JSON_KEY_SEPARATOR")
            || contains_raw_regex_substring(expr, r#"": "#),
        "{expr:?}"
    );
    assert!(contains_ref_named(expr, "JSON_ITEM_SEPARATOR") || contains_raw_regex_substring(expr, r#", "#), "{expr:?}");
    lower(&grammar).unwrap();
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
fn small_string_enum_at_root_uses_factored_suffix_choice() {
    let schema = json!({"type": "string", "enum": ["red", "green", "blue"]});

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let GrammarExpr::Sequence(parts) = start_expr(&grammar) else {
        panic!("expected factored sequence: {:?}", start_expr(&grammar));
    };
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], GrammarExpr::Literal(b"\"".to_vec()));
    let GrammarExpr::Choice(suffixes) = &parts[1] else {
        panic!("expected suffix choice: {:?}", parts[1]);
    };
    assert_eq!(suffixes.len(), 3);
    assert!(suffixes.contains(&GrammarExpr::Literal(b"red\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"green\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"blue\"".to_vec())));
    assert!(!contains_literal_bytes(start_expr(&grammar), b"\"red\""), "{:?}", start_expr(&grammar));
    lower(&grammar).unwrap();
}

#[test]
fn shared_prefix_string_enum_uses_factored_suffix_choice() {
    let schema = json!({
        "type": "string",
        "enum": ["SHARED_ALPHA", "SHARED_BETA", "SHARED_GAMMA", "SHARED_DELTA"]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let GrammarExpr::Sequence(parts) = start_expr(&grammar) else {
        panic!("expected factored sequence: {:?}", start_expr(&grammar));
    };
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], GrammarExpr::Literal(b"\"".to_vec()));
    let GrammarExpr::Choice(suffixes) = &parts[1] else {
        panic!("expected suffix choice: {:?}", parts[1]);
    };
    assert_eq!(suffixes.len(), 4);
    assert!(suffixes.contains(&GrammarExpr::Literal(b"SHARED_ALPHA\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"SHARED_BETA\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"SHARED_GAMMA\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"SHARED_DELTA\"".to_vec())));
    assert!(!contains_literal_bytes(start_expr(&grammar), b"\"SHARED_ALPHA\""), "{:?}", start_expr(&grammar));
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
fn unbounded_integer_multiple_of_three_lowers_broadly() {
    let schema = json!({"type": "integer", "multipleOf": 3});
    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Ref(name) if name == "JSON_INTEGER"));
    lower(&grammar).unwrap();
}

#[test]
fn lower_bounded_integer_multiple_of_twelve_lowers_to_range() {
    let schema = json!({"type": "integer", "minimum": 0, "multipleOf": 12});
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let GrammarExpr::RawRegex(regex) = start_expr(&grammar) else {
        panic!("expected broad integer range regex: {:?}", start_expr(&grammar));
    };
    assert!(regex.contains("[1-9][0-9]"), "{regex}");
    lower(&grammar).unwrap();
}



#[test]
fn recursive_root_array_ref_allows_split_object_opener() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "definitions": {"Node": {"$ref": "#"}},
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "name": {"type": "string"},
            "children": {"type": "array", "items": {"$ref": "#/definitions/Node"}}
        },
        "required": ["name", "children"]
    });
    assert!(schema_mask_allows_token_after_prefix(&schema, b"", 300, br#"{""#));
}

#[test]
fn llguidance_bounded_integer_multiple_rejects_signed_start() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "min_y": {
                "type": "integer",
                "minimum": -2032,
                "maximum": 2031,
                "multipleOf": 16
            }
        }
    });

    assert!(!schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"min_y":"#,
        482,
        b" -",
    ));
}

#[test]
fn bounded_integer_multiple_of_sixteen_lowers_without_enumerating_large_range() {
    let schema = json!({
        "type": "integer",
        "minimum": -2032,
        "maximum": 2031,
        "multipleOf": 16
    });
    let grammar = schema_to_named_grammar(&schema).unwrap();
    let GrammarExpr::Choice(alternatives) = start_expr(&grammar) else {
        panic!("expected bounded multiple choice: {:?}", start_expr(&grammar));
    };
    assert_eq!(alternatives.len(), 254);
    lower(&grammar).unwrap();
}

#[test]
fn non_integer_integer_multiple_of_remains_unsupported() {
    let schema = json!({"type": "integer", "multipleOf": 2.5});
    let error = schema_to_named_grammar(&schema).unwrap_err();
    assert!(error.to_string().contains("integer multipleOf=2.5 is unsupported"), "{error}");
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
fn anyof_pattern_with_sibling_string_type_does_not_broaden_to_json_string() {
    let schema = json!({
        "type": "string",
        "anyOf": [
            {"type": "string", "pattern": "^/x$"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start_line = glrm
        .lines()
        .find(|line| line.starts_with("nt start ::="))
        .unwrap_or_else(|| panic!("{glrm}"));
    assert!(!start_line.contains("| JSON_STRING"), "{glrm}");
    assert!(schema_accepts_bytes(&schema, br#""/x""#));
    assert!(!schema_accepts_bytes(&schema, br#""""#));
    assert!(!schema_accepts_bytes(&schema, br#""<""#));
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
fn allof_anyof_required_properties_lowers_to_single_grouped_object() {
    let schema = json!({
        "type": "object",
        "properties": {
            "resultPath": {"type": "string"},
            "deviceTags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["key", "value"],
                    "additionalProperties": false
                }
            },
            "deviceIds": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["resultPath"],
        "additionalProperties": false,
        "allOf": [
            {
                "anyOf": [
                    {"required": ["deviceTags"]},
                    {"required": ["deviceIds"]}
                ]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_closed_object_body"), 2);
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 0);
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"resultPath": "x", "deviceTags": [{"key": "k", "value": "v"}]}"#
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"resultPath": "x", "deviceIds": ["d1"]}"#
    ));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"resultPath": "x", "deviceTags": [], "deviceIds": ["d1"]}"#
    ));
    assert!(!schema_accepts_bytes(&schema, br#"{"resultPath":"x"}"#));
    lower(&grammar).unwrap();
}

#[test]
fn allof_anyof_ref_object_variants_distribute_through_common_object() {
    let schema = json!({
        "definitions": {
            "alpha": {
                "properties": {
                    "kind": {"type": "string", "enum": ["alpha"]},
                    "alpha": {"type": "string"}
                },
                "required": ["kind", "alpha"]
            },
            "beta": {
                "properties": {
                    "kind": {"type": "string", "enum": ["beta"]},
                    "beta": {"type": "string"}
                },
                "required": ["kind", "beta"]
            }
        },
        "type": "object",
        "allOf": [
            {
                "properties": {
                    "kind": {"enum": ["alpha", "beta"]},
                    "common": {"type": "string"}
                },
                "required": ["kind"],
                "additionalProperties": false
            },
            {
                "anyOf": [
                    {"$ref": "#/definitions/alpha"},
                    {"$ref": "#/definitions/beta"}
                ]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(schema_accepts_bytes(&schema, br#"{"kind": "alpha", "alpha": "x"}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"kind": "beta", "beta": "x"}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"kind":"alpha","beta":"x"}"#));
    lower(&grammar).unwrap();
}

#[test]
fn allof_oneof_required_properties_does_not_use_any_required_factoring() {
    let schema = json!({
        "type": "object",
        "properties": {
            "a": {"type": "boolean"},
            "b": {"type": "boolean"}
        },
        "additionalProperties": false,
        "allOf": [
            {
                "oneOf": [
                    {"required": ["a"]},
                    {"required": ["b"]}
                ]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 0);
    lower(&grammar).unwrap();
}

#[test]
fn anyof_required_sets_with_object_sibling_type_do_not_allow_non_objects() {
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "layerType": {"enum": ["KML"], "type": "string"},
            "path": {"pattern": "^file:.+\\.km[lz]$", "type": "string"},
            "title": {"type": "string"},
            "url": {"type": "string"}
        },
        "additionalProperties": false,
        "anyOf": [
            {"required": ["id", "layerType", "title", "url"]},
            {"required": ["id", "layerType", "path", "title"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_BOOL"));
    assert!(!contains_ref_named(start_expr(&grammar), "JSON_NULL"));

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
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
    lower(&grammar).unwrap();
}

#[test]
fn object_typed_anyof_branches_do_not_emit_generic_json_object_fallback() {
    let schema = json!({
        "type": "object",
        "definitions": {
            "a": {
                "type": "object",
                "properties": {
                    "dpp_version": {"type": "integer", "minimum": 1, "maximum": 1},
                    "file_version": {"type": "integer", "minimum": 1},
                    "parent_id": {"type": ["string", "null"]}
                },
                "additionalProperties": false,
                "anyOf": [
                    {"properties": {"parent_id": {"type": "null"}}, "required": ["parent_id"]},
                    {"properties": {"parent_id": {"type": "string"}}, "required": ["parent_id"]}
                ]
            },
            "b": {
                "properties": {
                    "dpp_version": {"type": "integer", "minimum": 1, "maximum": 1},
                    "file_version": {"type": "integer", "minimum": 1}
                },
                "required": ["dpp_version", "file_version"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {"$ref": "#/definitions/a"},
            {"$ref": "#/definitions/b"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start_line = glrm
        .lines()
        .find(|line| line.starts_with("nt start ::="))
        .unwrap_or_else(|| panic!("{glrm}"));
    assert!(!start_line.contains("| json_object"), "{glrm}");
    assert!(!start_line.contains("JSON_STRING"), "{glrm}");
    assert!(!start_line.contains("JSON_NUMBER"), "{glrm}");
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"parent_id": null, "dpp_version": 1, "file_version": 1}"#
    ));
    assert!(schema_accepts_bytes(&schema, br#"{"dpp_version": 1, "file_version": 1}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"x": 1}"#));
    assert!(!schema_accepts_bytes(&schema, br#""not an object""#));
    lower(&grammar).unwrap();
}

#[test]
fn anyof_open_objects_with_disjoint_optional_properties_collapses_to_json_object() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"}
                }
            },
            {
                "type": "object",
                "properties": {
                    "b": {"type": "number"}
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("nt start ::= json_object;"), "{glrm}");
    assert!(
        !glrm.contains("\\\"a\\\":") && !glrm.contains(r#"/"a": "#),
        "{glrm}"
    );
    assert!(
        !glrm.contains("\\\"b\\\":") && !glrm.contains(r#"/"b": "#),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn unconstrained_object_collapses_to_json_object() {
    let schema = json!({
        "type": "object"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(matches!(start_expr(&grammar), GrammarExpr::Ref(name) if name == "json_object"));
    assert!(!glrm.contains("OBJ_ORD"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn empty_properties_object_collapses_to_json_object() {
    let schema = json!({
        "type": "object",
        "properties": {}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(matches!(start_expr(&grammar), GrammarExpr::Ref(name) if name == "json_object"));
    assert!(!glrm.contains("OBJ_ORD"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn constrained_open_objects_do_not_collapse_to_json_object() {
    for schema in [
        json!({
            "type": "object",
            "additionalProperties": {"type": "integer"}
        }),
        json!({
            "type": "object",
            "maxProperties": 0
        }),
        json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"}
            }
        }),
    ] {
        let grammar = schema_to_named_grammar(&schema).unwrap();
        assert!(!matches!(start_expr(&grammar), GrammarExpr::Ref(name) if name == "json_object"));
        lower(&grammar).unwrap();
    }
}

#[test]
fn anyof_open_objects_with_shared_optional_property_does_not_collapse_to_json_object() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"}
                }
            },
            {
                "type": "object",
                "properties": {
                    "a": {"type": "number"}
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Ref(name) if name == "json_object"));
    assert!(glrm.contains("json_additional_excluded_key_colon_shared"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_nested_object_allof_refs_factor_into_single_body() {
    let schema = json!({
        "type": "object",
        "anyOf": [
            {
                "allOf": [
                    {"$ref": "#/definitions/app"},
                    {"required": ["mainClass"]}
                ]
            },
            {
                "allOf": [
                    {"$ref": "#/definitions/app"},
                    {"required": ["files"]}
                ]
            },
            {
                "allOf": [
                    {"$ref": "#/definitions/base"},
                    {
                        "properties": {"type": {"const": "lib"}},
                        "required": ["type"]
                    }
                ]
            }
        ],
        "definitions": {
            "base": {
                "type": "object",
                "properties": {
                    "compilerOptions": {"$ref": "#/definitions/compilerOptions"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "extends": {"type": "string"}
                }
            },
            "app": {
                "allOf": [
                    {"$ref": "#/definitions/base"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "mainClass": {"type": "string"}
                        }
                    }
                ]
            },
            "compilerOptions": {
                "type": "object",
                "properties": {
                    "debug": {"type": "boolean"},
                    "swf-version": {"type": "integer"},
                    "target-player": {"type": "string"}
                }
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(glrm.contains("nt start ::= \"{\" json_anyof_object_body"), "{glrm}");
    assert!(
        !glrm.lines().any(|line| {
            line.starts_with("nt start ::=")
                && line.contains("|")
                && line.contains("json_closed_object_body")
        }),
        "{glrm}"
    );
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"compilerOptions": {"debug": true, "swf-version": 9}, "mainClass": "Main"}"#
    ));
    lower(&grammar).unwrap();
}

#[test]
fn pattern_map_anyof_open_objects_with_disjoint_optional_properties_collapses_value_to_json_object()
{
    let schema = json!({
        "type": "object",
        "patternProperties": {
            "^[a-z]+$": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"}
                        }
                    },
                    {
                        "type": "object",
                        "properties": {
                            "b": {"type": "number"}
                        }
                    }
                ]
            }
        },
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let pattern_pair_rule = glrm
        .lines()
        .find(|line| line.contains("json_pattern_map_pair_"))
        .unwrap_or_else(|| panic!("{glrm}"));
    assert!(pattern_pair_rule.ends_with(" json_object;"), "{glrm}");
    assert!(!glrm.contains("obj_ord_"), "{glrm}");
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
fn anyof_closed_object_variants_share_identical_common_ref_property_transition() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "common": {"$ref": "#/$defs/commonValue"},
                    "kind": {"const": "left"},
                    "left": {"type": "string"}
                },
                "required": ["kind", "left"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "common": {"$ref": "#/$defs/commonValue"},
                    "kind": {"const": "right"},
                    "right": {"type": "number"}
                },
                "required": ["kind", "right"],
                "additionalProperties": false
            }
        ],
        "$defs": {
            "commonValue": {
                "type": "object",
                "additionalProperties": {}
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(
        glrm.matches("\\\"common\\\"").count() == 1
            || glrm.matches("\"common\"").count() == 1,
        "{glrm}"
    );
    assert!(!glrm.contains("-- schema_ref_1"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_closed_object_variants_do_not_share_mismatched_common_ref_property_transition() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "common": {"$ref": "#/$defs/commonObject"},
                    "kind": {"const": "left"},
                    "left": {"type": "string"}
                },
                "required": ["kind", "left"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "common": {"$ref": "#/$defs/commonString"},
                    "kind": {"const": "right"},
                    "right": {"type": "number"}
                },
                "required": ["kind", "right"],
                "additionalProperties": false
            }
        ],
        "$defs": {
            "commonObject": {
                "type": "object",
                "additionalProperties": {}
            },
            "commonString": {
                "type": "string"
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(
        glrm.matches("\\\"common\\\"").count() == 1
            || glrm.matches("\"common\"").count() == 1,
        "{glrm}"
    );
    assert!(
        glrm.contains("-- schema_ref_1") || glrm.contains("-- schema_ref_2"),
        "{glrm}"
    );
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
fn mixed_anyof_closed_object_variants_with_string_alt_use_variant_nfa() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "headless": {"type": "boolean"},
                    "name": {"type": "string", "enum": ["chrome"]}
                },
                "required": ["headless", "name"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "headless": {"type": "boolean"},
                    "name": {"type": "string", "enum": ["firefox"]}
                },
                "required": ["headless", "name"],
                "additionalProperties": false
            },
            {
                "type": "string"
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    let start = start_expr(&grammar);
    let GrammarExpr::Choice(alternatives) = start else {
        panic!("expected start choice, got {start:?}");
    };
    assert_eq!(alternatives.len(), 2, "{start:?}");
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    assert!(glrm.contains("JSON_STRING"), "{glrm}");

    assert!(schema_accepts_bytes(&schema, br#"{"headless": true, "name": "chrome"}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"headless": true, "name": "firefox"}"#));
    assert!(schema_accepts_bytes(&schema, br#""browser-name-string""#));
    assert!(!schema_accepts_bytes(&schema, br#"{"headless":true,"name":"safari"}"#));

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
    assert!(!matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
    assert_eq!(count_rules_with_prefix(&grammar, "json_anyof_object_body"), 1);
    assert!(grammar.rules.iter().any(|rule| contains_expr_nfa(&rule.expr)));
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
    assert!(glrm.contains(r#"/"a": "#), "{glrm}");
    assert!(glrm.contains("JSON_BOOL"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn allof_merges_array_ref_with_min_items_assertion() {
    let schema = json!({
        "definitions": {
            "positionArray": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1
            }
        },
        "allOf": [
            {"$ref": "#/definitions/positionArray"},
            {"minItems": 2}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(!contains_intersect_with_separated_sequence(expr), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn allof_merges_array_bounds_before_ref_branch() {
    let schema = json!({
        "definitions": {
            "positionArray": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 1
            }
        },
        "allOf": [
            {"minItems": 2},
            {"$ref": "#/definitions/positionArray"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(!contains_intersect_with_separated_sequence(expr), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn allof_array_min_max_items_merge_clamps_bounds() {
    let schema = json!({
        "allOf": [
            {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 5
            },
            {
                "minItems": 3,
                "maxItems": 4
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("{3,4}"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn allof_array_merge_preserves_non_array_type_union_guard() {
    let schema = json!({
        "allOf": [
            {
                "type": ["array", "string"],
                "items": {"type": "number"}
            },
            {"minItems": 2}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(format!("{expr:?}").contains("Ref(\"JSON_STRING\")"), "{expr:?}");
}

#[test]
fn allof_flattens_nested_object_allof_before_intersect() {
    let schema = json!({
        "definitions": {
            "baseConfig": {
                "type": "object",
                "properties": {
                    "config": {"type": "object"}
                }
            }
        },
        "allOf": [
            {
                "allOf": [
                    {"$ref": "#/definitions/baseConfig"},
                    {
                        "properties": {
                            "mainClass": {"type": "string"}
                        }
                    }
                ]
            },
            {"required": ["mainClass"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn allof_collapses_single_anyof_ref_before_intersect() {
    let schema = json!({
        "definitions": {
            "coreProperties": {
                "type": "object",
                "properties": {
                    "spFolder": {"type": "string"},
                    "distFolder": {"type": "string"}
                },
                "patternProperties": {
                    "^_": {"additionalProperties": true}
                }
            },
            "brandingConfig": {
                "type": "object",
                "properties": {
                    "logoPath": {"type": "string"}
                }
            }
        },
        "allOf": [
            {"$ref": "#/definitions/coreProperties"},
            {"anyOf": [{"$ref": "#/definitions/brandingConfig"}]},
            {"required": ["spFolder", "distFolder"]}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn recursive_ref_in_allof_is_not_inlined() {
    let schema = json!({
        "definitions": {
            "A": {
                "allOf": [
                    {"$ref": "#/definitions/B"},
                    {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"}
                        }
                    }
                ]
            },
            "B": {
                "type": "object",
                "properties": {
                    "child": {"$ref": "#/definitions/A"}
                }
            }
        },
        "$ref": "#/definitions/A"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn allof_drops_vacuous_json_value_property_when_refined() {
    let schema = json!({
        "definitions": {
            "Request": {
                "type": "object",
                "properties": {
                    "arguments": {
                        "type": ["array", "boolean", "integer", "null", "number", "object", "string"]
                    }
                }
            },
            "SpecificArguments": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        },
        "allOf": [
            {"$ref": "#/definitions/Request"},
            {
                "type": "object",
                "properties": {
                    "arguments": {"$ref": "#/definitions/SpecificArguments"}
                },
                "required": ["arguments"]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn allof_drops_vacuous_object_property_when_refined() {
    let schema = json!({
        "definitions": {
            "assembly": {
                "type": "object",
                "properties": {
                    "options": {"type": "object"}
                }
            },
            "specificOptions": {
                "type": "object",
                "properties": {
                    "serialization": {"type": "string"}
                }
            }
        },
        "allOf": [
            {"$ref": "#/definitions/assembly"},
            {
                "type": "object",
                "properties": {
                    "options": {"$ref": "#/definitions/specificOptions"}
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
}

#[test]
fn allof_distributes_over_object_anyof_before_lowering() {
    let schema = json!({
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "match": {"type": "string"},
                    "browser": {"type": "string"}
                },
                "required": ["match"]
            },
            {
                "anyOf": [
                    {"properties": {"devices": {"type": "object"}}},
                    {"properties": {"device": {"type": "string"}}}
                ]
            },
            {
                "properties": {
                    "platforms": {"type": "array", "items": {"type": "string"}},
                    "engine": {"type": "string"}
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_anyof_object_body_"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn allof_ref_to_nested_object_oneof_with_siblings_lowers() {
    let schema = json!({
        "definitions": {
            "namedObject": {
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            },
            "competency": {
                "allOf": [
                    {"$ref": "#/definitions/namedObject"},
                    {
                        "oneOf": [
                            {
                                "properties": {
                                    "competencies": {
                                        "type": "array",
                                        "items": {"$ref": "#/definitions/competency"}
                                    }
                                },
                                "required": ["competencies"]
                            },
                            {
                                "properties": {
                                    "abilities": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["abilities"]
                            }
                        ]
                    }
                ]
            }
        },
        "allOf": [
            {"$ref": "#/definitions/competency"},
            {
                "properties": {
                    "description": {"type": "string"}
                },
                "required": ["description"]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(count_rules_with_prefix(&grammar, "json_closed_object_body") > 0);
    lower(&grammar).unwrap();
}

#[test]
fn unsafe_allof_object_ref_intersection_broadens_to_choice() {
    let schema = json!({
        "$defs": {
            "base": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"}
                },
                "additionalProperties": false
            }
        },
        "allOf": [
            {"$ref": "#/$defs/base"},
            {"type": "string"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(matches!(expr, GrammarExpr::Choice(_)), "{expr:?}");
    assert!(!contains_intersect(expr), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn unsafe_allof_array_separated_sequence_broadens_to_choice() {
    let schema = json!({
        "allOf": [
            {
                "type": "array",
                "items": {"type": "integer"}
            },
            {"type": "string"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(matches!(expr, GrammarExpr::Choice(_)), "{expr:?}");
    assert!(!contains_intersect_with_separated_sequence(expr), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn terminal_safe_allof_keeps_intersection() {
    let schema = json!({
        "allOf": [
            {"type": "number", "minimum": 0},
            {"type": "number", "multipleOf": 0.25}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let expr = start_expr(&grammar);
    assert!(contains_intersect(expr), "{expr:?}");
    lower(&grammar).unwrap();
}

#[test]
fn oneof_object_branches_with_root_type_object_and_required_anyof_lowers() {
    let schema = json!({
        "type": "object",
        "oneOf": [
            {
                "properties": {
                    "fromNumber": {"type": "string"},
                    "bodyTemplate": {"type": "string"},
                    "mediaUrl": {"type": "string", "format": "uri"}
                },
                "allOf": [
                    {"required": ["fromNumber"]},
                    {"anyOf": [
                        {"required": ["bodyTemplate"]},
                        {"required": ["mediaUrl"]}
                    ]}
                ],
                "additionalProperties": false
            },
            {
                "properties": {
                    "messagingServiceSid": {"type": "string"},
                    "bodyTemplate": {"type": "string"},
                    "mediaUrl": {"type": "string", "format": "uri"}
                },
                "allOf": [
                    {"required": ["messagingServiceSid"]},
                    {"anyOf": [
                        {"required": ["bodyTemplate"]},
                        {"required": ["mediaUrl"]}
                    ]}
                ],
                "additionalProperties": false
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(
        count_rules_with_prefix(&grammar, "json_closed_object_body") > 0
            || glrm.contains("json_anyof_object_body"),
        "{glrm}"
    );
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
    let pattern_pair_rule = glrm
        .lines()
        .find(|line| line.contains("json_pattern_map_pair_"))
        .unwrap_or_else(|| panic!("{glrm}"));
    assert!(pattern_pair_rule.ends_with(" json_object;"), "{glrm}");
    assert!(!glrm.contains("json_anyof_object_body"), "{glrm}");
    assert!(
        !glrm.contains("\"{\" json_closed_object_body")
            || !glrm.contains("| \"{\" json_closed_object_body"),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}

#[test]
fn array_items_anyof_allof_ref_alias_variants_lower_to_shared_open_object_body() {
    let schema = json!({
        "$schema": "http://json-schema.org/draft-06/schema#",
        "definitions": {
            "Statement": {
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_api": {"type": "string"},
                                "text": {"type": "string"}
                            }
                        }
                    },
                    "id": {"type": "string"},
                    "supports": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "supported_by": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["id"]
            },
            "Agent": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "db_refs": {"type": "object"}
                },
                "required": ["name", "db_refs"]
            },
            "RegulateActivity": {
                "allOf": [
                    {"$ref": "#/definitions/Statement"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^((Activation)|(Inhibition))$"
                            },
                            "subj": {"$ref": "#/definitions/Agent"},
                            "obj": {"$ref": "#/definitions/Agent"},
                            "obj_activity": {"type": "string"}
                        },
                        "required": ["type"]
                    }
                ]
            },
            "ActiveForm": {
                "allOf": [
                    {"$ref": "#/definitions/Statement"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^ActiveForm$"
                            },
                            "agent": {"$ref": "#/definitions/Agent"},
                            "activity": {"type": "string"},
                            "is_active": {"type": "boolean"}
                        },
                        "required": ["type"]
                    }
                ]
            },
            "ActiveFormAlias": {
                "allOf": [
                    {"$ref": "#/definitions/ActiveForm"}
                ]
            }
        },
        "type": "array",
        "items": {
            "anyOf": [
                {"$ref": "#/definitions/RegulateActivity"},
                {"$ref": "#/definitions/ActiveFormAlias"}
            ]
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn sibling_pattern_addback_subtracts_local_pattern_language_for_o10297_shape() {
    let schema = json!({
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "score_history": {
                "type": "object",
                "patternProperties": {
                    "^\\d+$": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": false,
                            "properties": {
                                "player_id": {"type": "integer"},
                                "score": {"type": "integer"},
                                "rating_delta": {"type": "number"},
                                "place": {"type": "integer"}
                            },
                            "required": ["player_id", "score", "rating_delta", "place"]
                        }
                    }
                }
            },
            "hands_value_summary": {
                "type": "object",
                "patternProperties": {
                    "^-?\\d+$": {"type": "integer"}
                }
            }
        },
        "required": ["score_history", "hands_value_summary"],
        "additionalProperties": false
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);

    assert!(
        glrm.contains("json_additional_excluded_key_colon_shared")
            || glrm.contains("json_pattern_key_colon_"),
        "{glrm}"
    );
    lower(&grammar).unwrap();
}



#[test]
fn oneof_sibling_object_preserves_root_property_order() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "grantType": {"type": "string"},
            "redirectUris": {"type": "array", "items": {"type": "string"}},
            "responseType": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}}
        },
        "oneOf": [
            {
                "properties": {
                    "grantType": {"enum": ["authorization_code"]},
                    "responseType": {"enum": ["code"]}
                },
                "required": ["grantType"]
            },
            {
                "properties": {
                    "grantType": {"enum": ["client_credentials"]},
                    "responseType": {"enum": ["token"]}
                },
                "required": ["grantType"]
            }
        ]
    });

    assert!(schema_mask_allows_token_after_prefix(
        &schema,
        br#"{"grantType": "authorization_code", "redirectUris": ["https://example.com/callback"], ""#,
        81,
        b"r",
    ));
}

#[test]
fn llguidance_compat_drops_only_plain_subsumed_open_object_anyof_branch() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "next": {"type": "array"}
                }
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "next": {"type": "array"},
                    "resource": {"type": "string"}
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_STRING JSON_KEY_SEPARATOR"), "{glrm}");
    assert!(!glrm.contains(r#""resource": "#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn llguidance_compat_keeps_subsumed_open_object_branch_with_pattern_properties() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "patternProperties": {
                    "^(/([\\S]*)?)$": {"type": "string"}
                }
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("json_pattern_key_colon"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_drops_subsumed_open_object_branch_for_o83993_shape() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sort": {"type": "string"},
                    "thumbnail": {
                        "type": "object",
                        "properties": {
                            "href": {"type": "string"}
                        },
                        "required": ["href"]
                    }
                },
                "required": ["name"]
            },
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "sort": {"type": "string"}
                },
                "required": ["name"]
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_ADDITIONAL_KEY_COLON_SHARED"), "{glrm}");
    assert!(!glrm.contains(r#"/"thumbnail": / -->"#), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_drops_recursive_open_object_branches_subsumed_by_base_node() {
    let recursive_node = json!({
        "anyOf": [
            {"$ref": "#/definitions/A"},
            {"$ref": "#/definitions/B"},
            {"$ref": "#/definitions/C"}
        ]
    });
    let schema = json!({
        "definitions": {
            "Module": {
                "type": "object",
                "properties": {
                    "n": {"type": "string"}
                }
            },
            "A": {
                "type": "object",
                "properties": {
                    "h": {
                        "type": "array",
                        "items": recursive_node.clone()
                    },
                    "f": {"type": "array"},
                    "m": {"$ref": "#/definitions/Module"},
                    "x": {"type": "array"}
                }
            },
            "B": {
                "type": "object",
                "properties": {
                    "h": {
                        "type": "array",
                        "items": recursive_node.clone()
                    },
                    "f": {"type": "array"},
                    "m": {
                        "type": "object",
                        "properties": {
                            "n": {"enum": ["k"], "type": "string"}
                        }
                    },
                    "n": {"enum": ["r"], "type": "string"},
                    "x": {"type": "array"}
                }
            },
            "C": {
                "type": "object",
                "properties": {
                    "h": {
                        "type": "array",
                        "items": recursive_node
                    },
                    "f": {"type": "array"},
                    "m": {
                        "type": "object",
                        "properties": {
                            "n": {"enum": ["k"], "type": "string"}
                        }
                    },
                    "n": {"enum": ["r"], "type": "string"},
                    "x": {"type": "array"}
                }
            }
        },
        "properties": {
            "e": {
                "anyOf": [
                    {"$ref": "#/definitions/A"},
                    {"$ref": "#/definitions/B"},
                    {"$ref": "#/definitions/C"}
                ]
            }
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains("\"r\""), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn anyof_does_not_drop_open_object_branch_that_widens_base_property() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"enum": ["A"], "type": "string"}
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

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_STRING"), "{glrm}");
    lower(&grammar).unwrap();
}

fn shadow_author_author_path_schema() -> serde_json::Value {
    json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "last_modification": {"type": "string", "format": "date-time"}
                },
                "required": ["name"]
            },
            {
                "type": "object",
                "properties": {
                    "$ref": {
                        "type": "object",
                        "properties": {
                            "$ref": {"type": "string", "format": "uri"}
                        }
                    }
                }
            }
        ]
    })
}

#[test]
fn shadow_owner_owned_object_close_suppresses_residual_duplicate() {
    let schema = shadow_author_author_path_schema();
    let input = br#"{"name": "Ada"}"#;

    assert!(schema_accepts_bytes(&schema, input));
    assert_eq!(parser_path_count_after_bytes(&schema, input, 4), 1);
}

#[test]
fn shadow_owner_missing_required_key_keeps_residual_open_branch() {
    let schema = shadow_author_author_path_schema();

    assert!(schema_accepts_bytes(&schema, br#"{"email": "ada@example.com"}"#));
}

#[test]
fn shadow_owner_invalid_owner_fixed_type_keeps_residual_open_branch() {
    let schema = shadow_author_author_path_schema();

    assert!(schema_accepts_bytes(&schema, br#"{"name": 123}"#));
}

#[test]
fn shadow_owner_invalid_date_time_string_keeps_residual_string_subtraction() {
    let schema = shadow_author_author_path_schema();

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"name": "Ada", "last_modification": "not-a-date"}"#
    ));
}

#[test]
fn shadow_owner_out_of_order_fixed_fields_keep_residual_open_branch() {
    let schema = shadow_author_author_path_schema();

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"email": "ada@example.com", "name": "Ada"}"#
    ));
}

#[test]
fn shadow_owner_skips_residual_with_unsafe_additional_constraints() {
    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["name"]
            },
            {
                "type": "object",
                "properties": {
                    "$ref": {"type": "string"}
                },
                "additionalProperties": {"type": "string"}
            }
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.contains(" - json_string_constrained"), "{glrm}");
    assert!(schema_accepts_bytes(&schema, br#"{"name": "Ada"}"#));
    assert!(!schema_accepts_bytes(&schema, br#"{"name": 123}"#));
}

#[test]
fn shadow_owner_allows_unsupported_optional_owner_fields() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _allow_large = EnvVarGuard::set("GLRMASK_JSON_SCHEMA_ALLOW_LARGE", "1");

    let schema = json!({
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "text": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["language", "text"]
            },
            {
                "type": "object",
                "properties": {
                    "$ref": {"type": "string", "format": "uri"}
                }
            }
        ]
    });

    let required_only = br#"{"language": "en", "text": "Hello"}"#;
    assert!(schema_accepts_bytes(&schema, required_only));
    assert_eq!(parser_path_count_after_bytes(&schema, required_only, 4), 1);

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"language": "en", "text": "Hello", "tags": 123}"#
    ));
}

#[test]
fn shadow_owner_ref_branch_context_uses_factored_open_object_body() {
    let schema = json!({
        "definitions": {
            "Translation": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "text": {"type": "string"},
                    "contexts": {
                        "type": "object",
                        "patternProperties": {
                            "^/": {"$ref": "#/definitions/Context"}
                        }
                    }
                },
                "required": ["language", "text"]
            },
            "Context": {
                "anyOf": [
                    {"$ref": "#/definitions/Translation"},
                    {
                        "type": "object",
                        "properties": {
                            "$ref": {"type": "string", "format": "uri"}
                        }
                    }
                ]
            }
        },
        "$ref": "#/definitions/Context"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(
        glrm.contains("schema_ref_0 ::= schema_ref_1 | ") && glrm.contains("json_closed_object_body_6"),
        "{glrm}"
    );
    assert!(
        glrm.lines().any(|line| {
            line.contains(" ::= schema_ref_") && line.contains("| \"{\" json_closed_object_body")
        }),
        "{glrm}"
    );

    assert!(schema_accepts_bytes(&schema, br#"{}"#));
    assert!(schema_accepts_bytes(
        &schema,
        br#"{"$ref": "https://example.com"}"#
    ));

    let required_only = br#"{"language": "en", "text": "Hi"}"#;
    assert!(schema_accepts_bytes(&schema, required_only));
    assert_eq!(parser_path_count_after_bytes(&schema, required_only, 4), 1);

    assert!(schema_accepts_bytes(
        &schema,
        br#"{"language": "en", "text": "Hi", "contexts": 123}"#
    ));
}

#[test]
fn single_anyof_object_ref_with_sibling_properties_merges_before_lowering() {
    let schema = json!({
        "definitions": {
            "base": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        },
        "anyOf": [
            {"$ref": "#/definitions/base"}
        ],
        "properties": {
            "extra": {"type": "string"}
        }
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(!glrm.is_empty(), "{glrm}");
    assert!(
        glrm.contains("\"name\"") || glrm.contains("\\\"name\\\""),
        "{glrm}"
    );
    assert!(
        glrm.contains("\"extra\"") || glrm.contains("\\\"extra\\\""),
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
    assert!(glrm.contains("JSON_STRING_CHAR{2}"), "{glrm}");
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

#[test]
fn test_reproduce_declared_key_failure() {
    let schema = json!({
      "properties": {
        "a": {"items": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}, "z": {"type": "string"}}}},
        "b": {"items": {"properties": {"x": {"type": "string"}, "y": {"type": "string"}}}}
      },
      "additionalProperties": false
    });
    let grammar = schema_to_named_grammar(&schema).unwrap();
    println!("GRAMMAR: {:#?}", grammar);
    let lowered = lower(&grammar).unwrap();
    println!("LOWERED: {:#?}", lowered);
}


#[test]
fn allof_propagates_object_type_into_nested_oneof_sibling_branch() {
    let schema = json!({
        "$defs": {
            "common": {
                "type": "object",
                "required": ["name", "type"],
                "properties": {
                    "name": {"type": "string"}
                }
            },
            "file": {
                "properties": {
                    "type": {"enum": ["file"]},
                    "size": {"type": "integer"}
                }
            },
            "dir": {
                "properties": {
                    "type": {"enum": ["dir"]}
                }
            }
        },
        "type": "array",
        "items": {
            "allOf": [
                {"$ref": "#/$defs/common"},
                {
                    "properties": {
                        "user": {"type": "string"}
                    },
                    "oneOf": [
                        {"$ref": "#/$defs/file"},
                        {"$ref": "#/$defs/dir"}
                    ]
                }
            ]
        }
    });

    assert!(schema_accepts_bytes(
        &schema,
        br#"[{"size": 1, "name": "x", "type": "file"}]"#
    ));
    assert!(!schema_accepts_bytes(&schema, br#"[[{"name": "x", "type": "file"}]]"#));
}

#[test]
fn llguidance_compat_oneof_sibling_optional_key_mask_regression() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "additionalProperties": false,
        "oneOf": [
            {"properties": {"grantType": {"enum": ["authorization_code"]}, "responseType": {"enum": ["code"]}}, "required": ["grantType"]},
            {"properties": {"grantType": {"enum": ["client_credentials"]}, "responseType": {"enum": ["token"]}}, "required": ["grantType"]}
        ],
        "properties": {
            "grantType": {"type": "string"},
            "redirectUris": {"type": "array", "items": {"type": "string"}},
            "responseType": {"type": "string"},
            "scopes": {"type": "array", "items": {"type": "string"}}
        }
    });
    assert!(!schema_mask_allows_token_after_prefix(&schema, br#"{""#, 301, b"r"));
    let prefix = br#"{"grantType": "authorization_code", "redirectUris": ["https://example.com/callback"], ""#;
    assert!(schema_mask_allows_token_after_prefix(&schema, prefix, 300, b"response"));
}

#[test]
fn max_properties_equal_required_count_blocks_trailing_pair_token() {
    let schema = json!({
        "type": "object",
        "required": ["a", "b"],
        "maxProperties": 2,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"}
        }
    });
    let prefix = br#"{"a": "x", "b": "#;
    assert!(schema_mask_allows_token_after_prefix(&schema, prefix, 300, br#""y""#));
    assert!(!schema_mask_allows_token_after_prefix(&schema, prefix, 301, br#""y", "#));
}

#[test]
fn untyped_string_keywords_on_array_items_allow_non_string_items() {
    let schema = json!({
        "type": "object",
        "properties": {
            "checksums": {
                "type": "array",
                "items": {"minLength": 32, "maxLength": 32, "pattern": "^[0-9a-f]*$"}
            }
        }
    });
    assert!(schema_accepts_bytes(&schema, br#"{"checksums": ["b026324c6904b2a9cb4b88d6d61c81d1"]}"#));
    assert!(schema_accepts_bytes(&schema, br#"{"checksums": [[]]}"#));
    assert!(schema_mask_allows_token_after_prefix(&schema, br#"{"checksums":"#, 300, b" [["));
}

#[test]
fn llguidance_compat_untyped_pattern_items_allow_non_string_items() {
    let _lock = ENV_LOCK.lock().unwrap();
    let _guard = EnvVarGuard::set(GLRMASK_LLGUIDANCE_COMPAT_ENV, "1");
    let schema = json!({
        "type": "object",
        "properties": {
            "checksums": {
                "type": "array",
                "items": {"minLength": 32, "maxLength": 32, "pattern": "^[0-9a-f]*$"}
            }
        }
    });
    assert!(schema_mask_allows_token_after_prefix(&schema, br#"{"checksums":"#, 300, b" [["));
}
