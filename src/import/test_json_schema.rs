//! Regression tests for JSON-schema and EBNF constraints.
//!
//! Cases that depend on external GPT-2 vocab fixtures or removed internal
//! assertions stay omitted.

use crate::import::ast::{GrammarExpr, NamedGrammar};
use crate::import::json_schema::{json_schema_to_grammar, schema_to_named_grammar};
use crate::runtime::{Constraint, ConstraintState};
use crate::Vocab;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Build a byte-level vocabulary: token 0 = [0x00], token 1 = [0x01], ..., 255 = [0xFF].
fn byte_vocab() -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = (0..=255u32).map(|b| (b, vec![b as u8])).collect();
    Vocab::new(entries, None)
}

/// Check whether a token id is set in a packed bitmask.
fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() {
        return false;
    }
    (mask[word] >> (id % 32)) & 1 != 0
}

fn assert_token_allowed(mask: &[u32], token_id: usize, message: &str) {
    assert!(token_allowed(mask, token_id), "{message}");
}

fn assert_token_disallowed(mask: &[u32], token_id: usize, message: &str) {
    assert!(!token_allowed(mask, token_id), "{message}");
}

fn schema_constraint(schema: &str) -> Constraint {
    schema_constraint_with_vocab(schema, &byte_vocab())
}

fn schema_constraint_with_vocab(schema: &str, vocab: &Vocab) -> Constraint {
    Constraint::from_json_schema(schema, vocab)
        .unwrap_or_else(|error| panic!("schema should compile: {error}"))
}

// Bypass Constraint::from_json_schema so debug-only import checks do not short-circuit
// this targeted regression reproduction.
fn schema_constraint_direct_compile(schema: &str, vocab: &Vocab) -> Constraint {
    let grammar = json_schema_to_grammar(schema)
        .unwrap_or_else(|error| panic!("schema should compile to grammar: {error}"));
    crate::compiler::compile::compile(&grammar, vocab)
}

fn named_grammar_from_schema(schema: &str) -> NamedGrammar {
    let value: serde_json::Value = serde_json::from_str(schema).expect("schema JSON should parse");
    schema_to_named_grammar(&value).expect("schema should convert to named grammar")
}

fn advance_byte_prefix(state: &mut ConstraintState<'_>, prefix: &[u8]) {
    for &byte in prefix {
        let mask = state.mask();
        assert_token_allowed(
            &mask,
            byte as usize,
            &format!("prefix byte {byte:?} should be allowed"),
        );
        state.commit_token(byte as u32).unwrap();
    }
}

fn advance_tokens(state: &mut ConstraintState<'_>, tokens: &[u32]) {
    for &token in tokens {
        state.commit_token(token).unwrap();
    }
}

fn max_parser_paths_over_prefix(constraint: &Constraint, prefix: &[u8]) -> usize {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1024);
    for &byte in prefix {
        let mask = state.mask();
        assert_token_allowed(
            &mask,
            byte as usize,
            &format!("prefix byte {byte:?} should be allowed"),
        );
        state.commit_token(byte as u32).unwrap();
        max_paths = max_paths.max(state.parser_path_count(1024));
    }
    max_paths
}

fn parser_paths_at_prefix_end(constraint: &Constraint, prefix: &[u8]) -> usize {
    let mut state = constraint.start();
    advance_byte_prefix(&mut state, prefix);
    state.parser_path_count(1024)
}

fn trace_parser_stacks_over_prefix(constraint: &Constraint, prefix: &[u8]) -> usize {
    let mut state = constraint.start();
    let mut max_paths = state.parser_path_count(1024);
    println!("start max_paths={max_paths} stacks={:?}", state.debug_parser_stacks());

    for (index, &byte) in prefix.iter().enumerate() {
        let mask = state.mask();
        assert_token_allowed(
            &mask,
            byte as usize,
            &format!("prefix byte {byte:?} should be allowed"),
        );
        state.commit_token(byte as u32).unwrap();
        let paths = state.parser_path_count(1024);
        max_paths = max_paths.max(paths);
        println!(
            "step={index} byte={byte:?} paths={paths} max_paths={max_paths} stacks={:?}",
            state.debug_parser_stacks()
        );
    }

    max_paths
}

fn read_fixture_schema(path: &Path) -> Option<String> {
    if !path.exists() {
        return None;
    }

    let fixture_text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("should read fixture: {error}"));
    let fixture: serde_json::Value = serde_json::from_str(&fixture_text)
        .unwrap_or_else(|error| panic!("should parse fixture: {error}"));
    Some(
        fixture
            .get("schema")
            .unwrap_or_else(|| panic!("fixture should contain schema"))
            .to_string(),
    )
}

/// Build a Constraint from a JSON schema (using the byte vocab) and assert
/// that every input is accepted byte-by-byte.
fn schema_accepts(schema: &str, inputs: &[&str]) {
    let c = schema_constraint(schema);
    for input in inputs {
        let mut s = c.start();
        for (index, byte) in input.bytes().enumerate() {
            let mask = s.mask();
            assert_token_allowed(
                &mask,
                byte as usize,
                &format!(
                    "Schema should accept {:?}: byte {:?} ({:#04x}) at position {} not in mask",
                    input,
                    byte as char,
                    byte,
                    index
                ),
            );
            s.commit_token(byte as u32).unwrap();
        }
    }
}

fn schema_rejects(schema: &str, inputs: &[&str]) {
    let c = schema_constraint(schema);
    for input in inputs {
        let mut s = c.start();
        let mut rejected = false;
        for (index, byte) in input.bytes().enumerate() {
            let mask = s.mask();
            if !token_allowed(&mask, byte as usize) {
                rejected = true;
                break;
            }
            s.commit_token(byte as u32)
                .unwrap_or_else(|error| panic!("schema should reject {:?} at byte {}: {error}", input, index));
        }
        if !rejected {
            assert!(
                !s.is_finished(),
                "Schema should reject {:?}: input remained accepted through the full payload",
                input
            );
        }
    }
}

/// Build a Constraint from EBNF (using the byte vocab) and return it.
fn ebnf_constraint(ebnf: &str) -> Constraint {
    let vocab = byte_vocab();
    Constraint::from_ebnf(ebnf, &vocab)
        .unwrap_or_else(|error| panic!("EBNF should compile: {error}"))
}

fn contains_literal(expr: &GrammarExpr, target: &[u8]) -> bool {
    match expr {
        GrammarExpr::Literal(bytes) => bytes == target,
        GrammarExpr::Sequence(parts) => parts.iter().any(|part| contains_literal(part, target)),
        GrammarExpr::Choice(options) => options.iter().any(|option| contains_literal(option, target)),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_literal(expr, target) || contains_literal(exclude, target)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => contains_literal(inner, target),
        _ => false,
    }
}

fn contains_repeat_range(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::RepeatRange { .. } => true,
        GrammarExpr::Sequence(parts) => parts.iter().any(contains_repeat_range),
        GrammarExpr::Choice(options) => options.iter().any(contains_repeat_range),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_repeat_range(expr) || contains_repeat_range(exclude)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_repeat_range(inner),
        GrammarExpr::SeparatedSequence { items, separator, .. } => {
            items.iter().any(|(item, _)| contains_repeat_range(item))
                || contains_repeat_range(separator)
        }
        GrammarExpr::Intersect { expr, intersect } => contains_repeat_range(expr) || contains_repeat_range(intersect),

        GrammarExpr::Ref(_) => false,
        GrammarExpr::Epsilon => false,
        GrammarExpr::Literal(_) => false,
        GrammarExpr::CharClass { .. } => false,
        GrammarExpr::RawRegex(_) => false,
        GrammarExpr::AnyByte => false,
    }
}

fn contains_nonempty_sepseq(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::SeparatedSequence { allow_empty, .. } => !*allow_empty,
        GrammarExpr::Sequence(parts) => parts.iter().any(contains_nonempty_sepseq),
        GrammarExpr::Choice(options) => options.iter().any(contains_nonempty_sepseq),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_nonempty_sepseq(expr) || contains_nonempty_sepseq(exclude)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => contains_nonempty_sepseq(inner),
        _ => false,
    }
}

fn contains_ref(expr: &GrammarExpr, target: &str) -> bool {
    match expr {
        GrammarExpr::Ref(name) => name == target,
        GrammarExpr::Sequence(parts) => parts.iter().any(|part| contains_ref(part, target)),
        GrammarExpr::Choice(options) => options.iter().any(|option| contains_ref(option, target)),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_ref(expr, target) || contains_ref(exclude, target)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => contains_ref(inner, target),
        _ => false,
    }
}

fn count_ref(expr: &GrammarExpr, target: &str) -> usize {
    match expr {
        GrammarExpr::Ref(name) => usize::from(name == target),
        GrammarExpr::Sequence(parts) => parts.iter().map(|part| count_ref(part, target)).sum(),
        GrammarExpr::Choice(options) => options.iter().map(|option| count_ref(option, target)).sum(),
        GrammarExpr::Exclude { expr, exclude } => count_ref(expr, target) + count_ref(exclude, target),
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => count_ref(inner, target),
        _ => 0,
    }
}

fn contains_literal_prefix(expr: &GrammarExpr, prefix: &[u8]) -> bool {
    match expr {
        GrammarExpr::Literal(bytes) => bytes.starts_with(prefix),
        GrammarExpr::Sequence(parts) => parts.iter().any(|part| contains_literal_prefix(part, prefix)),
        GrammarExpr::Choice(options) => options.iter().any(|option| contains_literal_prefix(option, prefix)),
        GrammarExpr::Exclude { expr, exclude } => {
            contains_literal_prefix(expr, prefix) || contains_literal_prefix(exclude, prefix)
        }
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner)
        | GrammarExpr::RepeatRange { expr: inner, .. } => contains_literal_prefix(inner, prefix),
        _ => false,
    }
}

#[test]
#[should_panic(expected = "byte should be allowed at index")]
fn test_uri_reg_name_chunking_regression_repro_commit_rejects_on_valid_prefix() {
        // Minimal schema that exercises the same URI host/path boundary behavior
        // seen in o1051 profiling.
        let schema = r#"
        {
            "type": "object",
            "properties": {
                "portrait": {
                    "type": "string",
                    "format": "uri",
                    "maxLength": 255
                }
            },
            "required": ["portrait"],
            "additionalProperties": false
        }
        "#;

        let vocab = byte_vocab();
        let constraint = schema_constraint_direct_compile(schema, &vocab);
        let mut state = constraint.start();

        let payload = br#"{"portrait":"https://www.example.com/h"}"#;
        for (idx, &byte) in payload.iter().enumerate() {
                let mask = state.mask();
                assert_token_allowed(
                        &mask,
                        byte as usize,
                        &format!("byte should be allowed at index {idx}: {byte:?}"),
                );
                state
                        .commit_token(byte as u32)
                        .unwrap_or_else(|error| panic!("regression reproduced at byte index {idx} ({byte:?}): {error}"));
        }
}

#[cfg(debug_assertions)]
#[test]
fn test_debug_ab_overlap_follow_check_passes_for_uri_schema() {
        let schema = r#"
        {
            "type": "object",
            "properties": {
                "portrait": {
                    "type": "string",
                    "format": "uri",
                    "maxLength": 255
                }
            },
            "required": ["portrait"],
            "additionalProperties": false
        }
        "#;

        // This must go through from_json_schema so the debug-only post-build check runs.
        // The check should not panic for this schema.
        let _ = schema_constraint(schema);
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn with_env_var<T>(key: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let old = std::env::var_os(key);
    match value {
        Some(val) => unsafe { std::env::set_var(key, val) },
        None => unsafe { std::env::remove_var(key) },
    }
    let result = f();
    match old {
        Some(val) => unsafe { std::env::set_var(key, val) },
        None => unsafe { std::env::remove_var(key) },
    }
    result
}

/// Adapted from `test_ebnf_ws_nullable`.
///
/// Whitespace rule is nullable via `(…)*`; after committing `{`, the `}`
/// should be immediately valid.
#[test]
fn test_ebnf_ws_nullable() {
    let c = ebnf_constraint(
        "root ::= '{' WS '}'\nWS ::= ( ' ' | '\\t' | '\\n' | '\\r' )*",
    );
    let mut s = c.start();

    // Commit '{'
    s.commit_token(b'{' as u32).unwrap();

    // '}' should be valid (WS is nullable)
    let mask = s.mask();
    assert!(
        token_allowed(&mask, b'}' as usize),
        "'}}' should be valid immediately after '{{' when WS is nullable"
    );
}

#[test]
fn test_bounded_array_uses_repeat_range_ast() {
    let schema = r#"{
        "type": "array",
        "items": { "type": "integer" },
        "minItems": 1,
        "maxItems": 3
    }"#;

    let named = named_grammar_from_schema(schema);
    assert!(
        named.rules.iter().any(|rule| contains_repeat_range(&rule.expr)),
        "bounded arrays should preserve a RepeatRange node instead of desugaring to an optional ladder"
    );
}

#[test]
fn test_shared_additional_properties_key_exclusions_are_on_by_default() {
    let schema = r#"{
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
    }"#;

    let grammar = with_env_var("GLRMASK_AP_SHARED_EXCLUSIONS", None, || named_grammar_from_schema(schema));
    assert!(grammar.rules.iter().any(|rule| rule.name == "AP_SHARED_KEY"));
}

#[test]
fn test_shared_additional_properties_key_exclusions_can_be_disabled_explicitly() {
    let schema = r#"{
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
    }"#;

    let grammar = with_env_var("GLRMASK_AP_SHARED_EXCLUSIONS", Some("0"), || named_grammar_from_schema(schema));
    assert!(grammar.rules.iter().all(|rule| rule.name != "AP_SHARED_KEY"));
}

#[test]
fn test_shared_additional_properties_key_exclusions_create_shared_terminal_and_allow_back_rules() {
    let schema = r#"{
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
    }"#;

    let grammar = with_env_var("GLRMASK_AP_SHARED_EXCLUSIONS", Some("1"), || named_grammar_from_schema(schema));

    let shared_rules: Vec<_> = grammar
        .rules
        .iter()
        .filter(|rule| rule.name == "AP_SHARED_KEY" && !rule.is_internal)
        .collect();
    assert_eq!(shared_rules.len(), 1, "expected exactly one non-internal shared additional-properties key terminal");
    let shared_rule_name = shared_rules[0].name.as_str();

    let ap_key_rules: Vec<_> = grammar
        .rules
        .iter()
        .filter(|rule| rule.name.contains("_ap_key_") && !rule.is_terminal)
        .collect();
    assert!(!ap_key_rules.is_empty(), "expected object-specific allow-back nonterminal rules");
    assert!(ap_key_rules.iter().all(|rule| contains_ref(&rule.expr, shared_rule_name)));
    assert!(ap_key_rules.iter().any(|rule| contains_literal_prefix(&rule.expr, b"a\"")));
    assert!(ap_key_rules.iter().any(|rule| contains_literal_prefix(&rule.expr, b"b\"")));
}

#[test]
fn test_pattern_key_terminal_is_extracted_for_unanchored_sides() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "string"}
        },
        "patternProperties": {
            "b": {"type": "string"}
        },
        "additionalProperties": false
    }"#;

    let grammar = named_grammar_from_schema(schema);
    let pp_key_rule = grammar
        .rules
        .iter()
        .find(|rule| rule.name.contains("_PP0_KEY"))
        .expect("expected pattern key terminal rule");

    assert!(pp_key_rule.is_terminal, "pattern key wrapper should be a terminal rule");
    assert!(!contains_ref(&pp_key_rule.expr, "JSON_STRING_MIDDLE"));
}

#[test]
fn test_pattern_key_terminal_anchor_controls_extracted_body_shape() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "string"}
        },
        "patternProperties": {
            "^b": {"type": "string"}
        },
        "additionalProperties": false
    }"#;

    let grammar = named_grammar_from_schema(schema);
    let pp_key_rule = grammar
        .rules
        .iter()
        .find(|rule| rule.name.contains("_PP0_KEY"))
        .expect("expected pattern key terminal rule");

    assert!(pp_key_rule.is_terminal, "pattern key wrapper should be a terminal rule");
    assert!(!contains_ref(&pp_key_rule.expr, "JSON_STRING_MIDDLE"));
}

/// Adapted from `test_ebnf_object_member_after_brace`.
///
/// After `{`, both `"` (starting a member) and `}` (empty object) should be valid.
#[test]
fn test_ebnf_object_member_after_brace() {
    let ebnf = "\
root ::= '{' WS member_opt WS '}'
member_opt ::= ( member ( ',' WS member )* )?
member ::= '\"name\"' WS ':' WS 'value'
WS ::= ( ' ' | '\\t' | '\\n' | '\\r' )*";
    let c = ebnf_constraint(ebnf);
    let mut s = c.start();

    s.commit_token(b'{' as u32).unwrap();
    let mask = s.mask();

    assert!(
        token_allowed(&mask, b'"' as usize),
        "'\"' should be valid after '{{' for starting a member"
    );
    assert!(
        token_allowed(&mask, b'}' as usize),
        "'}}' should be valid after '{{' for empty object"
    );
}

// ── JSON schema constraint tests ────────────────────────────────────────────

/// Adapted from `test_schema_simple_object`.
#[test]
fn test_schema_simple_object() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        }
    }"#;
    schema_accepts(
        schema,
        &["{}",
          r#"{"name": "test"}"#,
                    r#"{"name": "hello world"}"#],
    );
}

/// Adapted from `test_schema_additional_properties_true`.
#[test]
fn test_schema_additional_properties_true() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "additionalProperties": true
    }"#;
    schema_accepts(
        schema,
        &[
            "{}",
            r#"{"name": "test"}"#,
            r#"{"foo": "bar"}"#,
            r#"{"name": "test", "extra": 123}"#,
            r#"{"x": null, "y": true, "z": [1, 2, 3]}"#,
        ],
    );
}

/// Adapted from `test_schema_additional_properties_schema`.
#[test]
fn test_schema_additional_properties_schema() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "additionalProperties": {"type": "string"}
    }"#;
    schema_accepts(
        schema,
        &[
            "{}",
            r#"{"name": "test"}"#,
            r#"{"foo": "bar"}"#,
            r#"{"name": "test", "extra": "value"}"#,
        ],
    );
}

/// Adapted from `test_schema_dependencies`.
#[test]
fn test_schema_dependencies() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "dependencies": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
        },
        "required": ["name", "version"]
    }"#;
    schema_accepts(
        schema,
        &[
            r#"{"name": "pkg", "version": "1.0.0"}"#,
            r#"{"name": "pkg", "version": "1.0.0", "dependencies": {}}"#,
            r#"{"name": "pkg", "version": "1.0.0", "dependencies": {"lodash": "^4.0.0"}}"#,
        ],
    );
}

/// Adapted from `test_schema_nested_objects`.
#[test]
fn test_schema_nested_objects() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {"type": "string"}
                }
            }
        }
    }"#;
    schema_accepts(
        schema,
        &[
            "{}",
            r#"{"outer": {}}"#,
            r#"{"outer": {"inner": "value"}}"#,
        ],
    );
}

#[test]
fn test_diag_replace_comma() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "size": {"type": "number"},
            "value": {"type": "number"},
            "lastSetValue": {"type": "number"}
        }
    }"#;
    let c = schema_constraint(schema);
    let mut s = c.start();

    // Commit each byte, checking mask and commit separately
    let prefix = br#"{"id": 1"#;
    for (i, &byte) in prefix.iter().enumerate() {
        let mask = s.mask();
        let allowed = token_allowed(&mask, byte as usize);
        eprintln!("step {i}: byte {} ({}), mask_allows={allowed}", byte, byte as char);
        assert!(allowed, "mask should allow byte {} at step {i}", byte as char);
        s.commit_token(byte as u32).unwrap();
    }

    // Now check the comma
    let mask = s.mask();
    let comma_allowed_mask = token_allowed(&mask, b',' as usize);
    eprintln!("mask allows comma: {comma_allowed_mask}");

    // Try commit directly (skip mask check)
    let commit_result = s.commit_token(b',' as u32);
    eprintln!("commit result for comma: {commit_result:?}");
    
    if !comma_allowed_mask {
        // Show what the mask DOES allow
        let mut allowed_bytes = Vec::new();
        for b in 0..=255u8 {
            if token_allowed(&mask, b as usize) {
                allowed_bytes.push(b);
            }
        }
        let as_chars: Vec<String> = allowed_bytes.iter().map(|&b| {
            if b.is_ascii_graphic() || b == b' ' { format!("'{}'", b as char) }
            else { format!("0x{b:02x}") }
        }).collect();
        eprintln!("allowed bytes after prefix: {:?}", as_chars);
    }
}

/// Minimal reproducer: after a number value in a 2-property object, the mask
/// must allow comma.
#[test]
fn test_replace_comma_after_number_value() {
    let schema = r#"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}}}"#;
    let c = schema_constraint(schema);
    let mut s = c.start();
    advance_byte_prefix(&mut s, br#"{"a": 1"#);
    let mask = s.mask();
    assert_token_allowed(&mask, b',' as usize, "comma must be allowed after number value in object");
}

#[test]
fn test_schema_object_after_comma_requires_key_quote() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "id": {"type": "number"},
            "size": {"type": "number"},
            "value": {"type": "number"},
            "lastSetValue": {"type": "number"}
        }
    }"#;
    let c = schema_constraint(schema);

    let mut s = c.start();
    advance_byte_prefix(&mut s, br#"{"id": 1, "#);

    let mask = s.mask();
    assert_token_allowed(
        &mask,
        b'"' as usize,
        "object member after comma+space should start with a quote",
    );
    assert_token_disallowed(
        &mask,
        b'1' as usize,
        "object member after comma+space must not allow a digit",
    );
}

#[test]
fn test_o56012_fixture_after_comma_requires_key_quote() {
    let fixture_path = Path::new(
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/maskbench/data/Github_hard---o56012.json",
    );
    let Some(schema) = read_fixture_schema(fixture_path) else {
        return;
    };
    let example = concat!(
        "{\"id\": 2, \"name\": \"RGB Light\", \"roomID\": 0, \"type\": \"rgb_driver\", ",
        "\"remoteGatewayId\": 0, \"remoteDeviceID\": 0, \"properties\": {\"UIMessageSendTime\": ",
        "\"2022-01-01 00:00:00\", \"associationMode\": \"0\", \"bScaler\": \"1\", ",
        "\"buttonType\": \"0\", \"classConfigure\": \"0\", \"classGeneric\": \"0\", ",
        "\"classSupport\": \"0\", \"classVersion\": \"0\", \"color\": \"#FFFFFF\", ",
        "\"currentProgram\": \"0\", \"currentProgramID\": \"0\", \"dead\": \"0\", ",
        "\"deviceControlType\": \"0\", \"deviceIcon\": \"0\", \"disabled\": \"0\", ",
        "\"emailNotificationID\": \"0\", \"emailNotificationType\": \"0\", \"endPoint\": \"0\", ",
        "\"favoriteProgram\": \"0\", \"gScaler\": \"1\", \"isBatteryOperated\": \"0\", ",
        "\"isLight\": \"1\", \"lastColorSet\": \"#FFFFFF\", \"lastUsedPrograms\": \"0\", ",
        "\"liliOffCommand\": \"0\", \"liliOnCommand\": \"0\", \"log\": \"0\", ",
        "\"logTemp\": \"0\", \"meterSupport\": \"0\", \"mode\": \"0\", ",
        "\"needConfigure\": \"0\", \"nodeID\": \"0\", \"parametersTemplate\": \"0\", ",
        "\"parentID\": \"0\", \"pollingRetryError\": \"0\", \"pollingTime\": \"0\", ",
        "\"pollingTimeNext\": \"0\", \"pollingTimeSec\": \"0\", \"productInfo\": \"0\", ",
        "\"programsSortOrder\": \"0\", \"pushNotificationID\": \"0\", \"pushNotificationType\": \"0\", ",
        "\"rScaler\": \"1\", \"rememberColor\": \"0\", \"requestNodeNeighborState\": \"0\", ",
        "\"requestNodeNeighborStateTimeStemp\": \"0\", \"saveLogs\": \"0\", \"sensorSupport\": \"0\", ",
        "\"showChildren\": \"0\", \"showEnergy\": \"0\", \"smsNotificationID\": \"0\", ",
        "\"smsNotificationType\": \"0\", \"sortOrder\": \"0\", \"unit\": \"0\", ",
        "\"unitMeter\": \"0\", \"unitSensor\": \"0\", \"useTemplate\": \"0\", ",
        "\"userDescription\": \"0\", \"value\": \"0\", \"valueMeter\": \"0\", ",
        "\"valueSensor\": \"0\", \"zwaveCompany\": \"0\", \"zwaveInfo\": \"0\", ",
        "\"zwaveVersion\": \"0\", \"parameters\": [{\"id\": 1, \"size\": 1, \"value\": 1, ",
        "\"lastSetValue\": 1}], \"associationView\": [{\"groupID\": 1, \"devices\": [1, 2, 3]}], ",
        "\"associationSet\": [{\"groupID\": 1, \"devices\": [1, 2, 3]}]}, \"actions\": {\"firmwareUpdate\": 1, ",
        "\"pollingTimeSec\": 1, \"requestNodeNeighborUpdate\": 0, \"resetMeter\": 0, \"setB\": 1, ",
        "\"setColor\": 4, \"setG\": 1, \"setR\": 1, \"setValue\": 1, \"setW\": 1, ",
        "\"silentSetColor\": 4, \"startProgram\": 1, \"turnOff\": 0, \"turnOn\": 0}, ",
        "\"created\": 1643723400, \"modified\": 1643723400, \"sortOrder\": 0}"
    );
    let prefix_marker = "\"parameters\": [{\"id\": 1, ";
    let prefix_end = example
        .find(prefix_marker)
        .unwrap_or_else(|| panic!("example should contain target prefix"))
        + prefix_marker.len();

    let c = schema_constraint(&schema);
    let mut s = c.start();
    advance_byte_prefix(&mut s, &example.as_bytes()[..prefix_end]);

    let mask = s.mask();
    assert_token_allowed(&mask, b'"' as usize, "expected a key-opening quote");
    assert_token_disallowed(
        &mask,
        b'1' as usize,
        "o56012 native path must not allow a digit after comma+space in parameters item"
    );
}

/// Adapted from `test_schema_array`.
#[test]
fn test_schema_array() {
    let schema = r#"{
        "type": "array",
        "items": {"type": "string"}
    }"#;
    schema_accepts(schema, &["[]", r#"["a"]"#, r#"["a", "b", "c"]"#]);
}

/// Adapted from `test_schema_anyof`.
#[test]
fn test_schema_anyof() {
    let schema = r#"{
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"}
        ]
    }"#;
    schema_accepts(schema, &[r#""hello""#, "42", "3.14", "true", "false"]);
}

/// Adapted from `test_schema_enum`.
#[test]
fn test_schema_enum() {
    let schema = r#"{
        "enum": ["red", "green", "blue"]
    }"#;
    schema_accepts(schema, &[r#""red""#, r#""green""#, r#""blue""#]);
}

/// Adapted from `test_schema_const`.
#[test]
fn test_schema_const() {
    let schema = r#"{
        "const": "fixed_value"
    }"#;
    schema_accepts(schema, &[r#""fixed_value""#]);
}

/// Adapted from the original minimal-vocab const regression.
///
/// Uses a minimal custom vocabulary (only the bytes needed for `"x"`).
#[test]
fn test_schema_const_with_minimal_vocab() {
    let schema = r#"{
        "const": "x"
    }"#;
    let entries = vec![
        (b'"' as u32, vec![b'"']),
        (b'x' as u32, vec![b'x']),
        (b' ' as u32, vec![b' ']),
        (b'\n' as u32, vec![b'\n']),
        (b'\r' as u32, vec![b'\r']),
        (b'\t' as u32, vec![b'\t']),
    ];
    let vocab = Vocab::new(entries, None);
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut s = c.start();
    for byte in b"\"x\"" {
        let mask = s.mask();
        assert_token_allowed(
            &mask,
            *byte as usize,
            &format!("Byte {:?} ({:#04x}) should be valid", *byte as char, byte),
        );
        s.commit_token(*byte as u32).unwrap()
    }
}

#[test]
fn test_schema_const_object_native_spacing() {
    let schema = r#"{
        "const": {"name": "x", "count": 2}
    }"#;
    schema_accepts(schema, &[r#"{"name": "x", "count": 2}"#]);
}

#[test]
fn test_schema_const_array_native_spacing() {
    let schema = r#"{
        "const": [1, 2, 3]
    }"#;
    schema_accepts(schema, &["[1, 2, 3]"]);
}

/// Adapted from `test_json_schema_mask_generation`.
#[test]
#[ignore = "current JSON Schema key-prefix continuation still rejects sparse multibyte whitespace+quote vocab after '{\\n  \"' prefix"]
fn test_json_schema_allows_name_after_brace_newline_space_quote_prefix() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "additionalProperties": true
    }"#;
    let vocab = Vocab::new(
        vec![
            (90u32, b"{".to_vec()),
            (198u32, b"\n".to_vec()),
            (220u32, b" ".to_vec()),
            (366u32, b" \"".to_vec()),
            (3672u32, b"name".to_vec()),
        ],
        None,
    );
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut s = c.start();

    advance_tokens(&mut s, &[90u32, 198u32, 220u32, 366u32]);

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 3672),
        "token 'name' should be allowed after the object prefix '{{\\n  \"'"
    );
}

/// Adapted from `test_newsletter_schema_disallows_quote_colon_minus`.
#[test]
fn test_json_schema_name_prefix_disallows_quote_colon_minus_token() {
    let schema = r#"{
        "type": "object",
        "title": "Newsletter Subscription",
        "properties": {
            "name": {"type": "string", "minLength": 8, "maxLength": 80},
            "email": {"type": "string", "maxLength": 120},
            "lists": {"type": "string", "enum": ["Daily New", "Promotion"]}
        },
        "additionalProperties": false,
        "required": ["name", "email", "lists"],
        "x-guidance": {
            "item_separator": ", ",
            "key_separator": ": ",
            "whitespace_flexible": false,
            "whitespace_pattern": null,
            "coerce_one_of": false,
            "lenient": false
        }
    }"#;
    let vocab = Vocab::new(
        vec![
            (1u32, b"{".to_vec()),
            (2u32, b"\"".to_vec()),
            (3u32, b"name".to_vec()),
            (4u32, b"\":-".to_vec()),
        ],
        None,
    );
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut s = c.start();

    advance_tokens(&mut s, &[1u32, 2u32, 3u32]);

    let mask = s.mask();
    assert_token_disallowed(
        &mask,
        4,
        "token '\":-' must not be allowed after the key prefix '\"name'",
    );
}

/// Adapted from `test_conversion_simple_object`.
///
/// Checks that a simple object schema with string and integer properties
/// compiles to a grammar containing rules for those types.
#[test]
fn test_conversion_simple_object() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }"#;
    let named = named_grammar_from_schema(schema);
    assert!(
        !named.rules.is_empty(),
        "Should produce non-empty rules"
    );
    assert_eq!(named.start, "start");
}

#[test]
fn test_email_format_uses_plain_bounded_string_lowering() {
    let schema = r#"{
        "type": "string",
        "format": "email",
        "maxLength": 5
    }"#;
    let named = named_grammar_from_schema(schema);

    assert!(
        named
            .rules
            .iter()
            .any(|rule| rule.name.starts_with("JSON_STRING_BOUNDED_")),
        "email strings should reuse bounded-string helper terminals"
    );
    assert!(
        named.rules.iter().all(|rule| rule.name != "JSON_FORMAT_STRING"),
        "email strings should not lower through the regex format terminal path"
    );
}

#[test]
fn test_closed_object_min_properties_uses_nonempty_sepseq() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"}
        },
        "minProperties": 1,
        "additionalProperties": false
    }"#;
    let named = named_grammar_from_schema(schema);

    assert!(
        named.rules.iter().any(|rule| contains_nonempty_sepseq(&rule.expr)),
        "closed-object minProperties lowering should force a nonempty SeparatedSequence"
    );
}

/// Adapted from `test_conversion_any_of`.
///
/// Checks that an anyOf schema produces a non-empty grammar.
#[test]
fn test_conversion_any_of() {
    let schema = r#"{
        "anyOf": [
            {"type": "string"},
            {"type": "number"}
        ]
    }"#;
    let named = named_grammar_from_schema(schema);
    assert!(
        !named.rules.is_empty(),
        "anyOf schema should produce non-empty rules"
    );
}

#[test]
fn test_closed_object_anyof_ordered_builtin_examples() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "image": {"type": "string"},
                    "plugin": {"type": "string"},
                    "sync": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "alpha": {"type": "object"},
                    "context": {"type": "string"},
                    "image": {"type": "string"},
                    "plugin": {"type": "string"},
                    "sync": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "delta": {"type": "object"},
                    "image": {"type": "string"},
                    "plugin": {"type": "string"},
                    "sync": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "image": {"type": "string"},
                    "plugin": {"type": "string"},
                    "sync": {"type": "string"},
                    "zeta": {"type": "object"}
                },
                "additionalProperties": false
            }
        ]
    }"#;

    schema_accepts(
        schema,
        &[
            r#"{"context": "a", "image": "b", "plugin": "c", "sync": "d"}"#,
            r#"{"alpha": {}, "context": "a", "image": "b", "plugin": "c", "sync": "d"}"#,
            r#"{"context": "a", "delta": {}, "image": "b", "plugin": "c", "sync": "d"}"#,
            r#"{"context": "a", "image": "b", "plugin": "c", "sync": "d", "zeta": {}}"#,
        ],
    );
}

#[test]
fn test_closed_object_anyof_preserves_ordered_language() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "context": {"type": "string"},
                    "image": {"type": "string"},
                    "plugin": {"type": "string"},
                    "sync": {"type": "string"}
                },
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "alpha": {"type": "object"},
                    "context": {"type": "string"},
                    "image": {"type": "string"},
                    "plugin": {"type": "string"},
                    "sync": {"type": "string"}
                },
                "additionalProperties": false
            }
        ]
    }"#;

    schema_rejects(
        schema,
        &[
            r#"{"image": "b", "context": "a", "plugin": "c", "sync": "d"}"#,
            r#"{"context": "a", "plugin": "c", "image": "b", "sync": "d"}"#,
        ],
    );
}

#[test]
fn test_closed_object_anyof_prunes_required_subset_branch() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "root": {
                "type": "object",
                "properties": {
                    "children": {
                        "type": "object",
                        "patternProperties": {
                            "^k$": {
                                "type": "object",
                                "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                    }
                },
                "anyOf": [
                    {},
                    {"required": ["children"]}
                ],
                "additionalProperties": false
            }
        },
        "required": ["root"],
        "additionalProperties": false
    }"#;

    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(
        &constraint,
        br#"{"root": {"children": {"k": {}}}}"#,
    );
    assert!(
        max_paths <= 2,
        "anyOf should prune the stricter closed-object branch instead of keeping depth-driven duplicate alternatives; got max_paths={max_paths}"
    );
}

#[test]
fn test_overlapping_oneof_is_rejected_without_explicit_coercion() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "oneOf": [
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
                    "a": {"type": "string"},
                    "b": {"type": "string"}
                },
                "additionalProperties": false
            }
        ]
    }"#;

    let err = json_schema_to_grammar(schema).expect_err("overlapping oneOf should be rejected");
    assert!(
        err.to_string().contains("oneOf constraints are not supported"),
        "expected llguidance-style oneOf rejection, got: {err}"
    );
}

#[test]
fn test_closed_object_single_variant_preserves_ordered_language() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "structureTests": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["image"],
        "additionalProperties": false
    }"#;

    schema_accepts(
        schema,
        &[
            r#"{"image": "x"}"#,
            r#"{"image": "x", "structureTests": []}"#,
        ],
    );
    schema_rejects(
        schema,
        &[
            r#"{}"#,
            r#"{"structureTests": []}"#,
            r#"{"structureTests": [], "image": "x"}"#,
        ],
    );
}

#[test]
fn test_closed_object_single_variant_does_not_emit_exact_dfa_states() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "image": {"type": "string"},
            "structureTests": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["image"],
        "additionalProperties": false
    }"#;
    let named = named_grammar_from_schema(schema);

    assert!(
        named.rules.iter().all(|rule| !rule.name.starts_with("obj_ord_q_")),
        "closed-object lowering should not emit exact closed-object DFA states"
    );
}

#[test]
fn test_disjoint_oneof_objects_compile_without_explicit_leniency() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "help": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"const": "email"}
                },
                "required": ["label", "type"],
                "additionalProperties": false
            },
            {
                "type": "object",
                "properties": {
                    "help": {"type": "string"},
                    "label": {"type": "string"},
                    "type": {"const": "time"}
                },
                "required": ["label", "type"],
                "additionalProperties": false
            }
        ]
    }"#;

    schema_accepts(
        schema,
        &[
            r#"{"label": "x", "type": "email"}"#,
            r#"{"label": "x", "type": "time"}"#,
            r#"{"help": "h", "label": "x", "type": "email"}"#,
        ],
    );
    schema_rejects(
        schema,
        &[
            r#"{}"#,
            r#"{"label": "x"}"#,
            r#"{"help": "h", "label": "x"}"#,
        ],
    );
}

#[test]
fn test_lenient_oneof_is_coerced_to_anyof() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "x-guidance": {
            "lenient": true
        },
        "oneOf": [
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
                    "a": {"type": "string"},
                    "b": {"type": "string"}
                },
                "additionalProperties": false
            }
        ]
    }"#;

    schema_accepts(
        schema,
        &[
            r#"{"a": "x"}"#,
            r#"{"a": "x", "b": "y"}"#,
        ],
    );
}

#[test]
fn test_explicit_coerce_oneof_is_treated_as_anyof() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "x-guidance": {
            "coerce_one_of": true
        },
        "oneOf": [
            {"type": "string"},
            {"type": "integer"}
        ]
    }"#;

    schema_accepts(schema, &[r#""value""#, r#"17"#]);
    schema_rejects(schema, &[r#"true"#]);
}

#[test]
fn test_ref_oneof_with_required_pattern_discriminator_compiles() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r##"{
        "definitions": {
            "ecu": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "pattern": "^ecu$"
                    },
                    "name": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": false
            },
            "string_table": {
                "type": "object",
                "properties": {
                    "file_type": {
                        "type": "string",
                        "pattern": "^string_table$"
                    },
                    "table_name": {"type": "string"}
                },
                "required": ["file_type", "table_name"],
                "additionalProperties": false
            }
        },
        "oneOf": [
            {"$ref": "#/definitions/ecu"},
            {"$ref": "#/definitions/string_table"}
        ]
    }"##;

    schema_accepts(
        schema,
        &[
            r#"{"name": "engine"}"#,
            r#"{"file_type": "string_table", "table_name": "codes"}"#,
        ],
    );
    schema_rejects(
        schema,
        &[
            r#"{"file_type": "ecu", "table_name": "codes"}"#,
            r#"{"file_type": "string_table", "name": "engine"}"#,
        ],
    );
}

#[test]
fn test_ref_oneof_with_allof_const_discriminator_compiles() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r##"{
        "definitions": {
            "item": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"}
                },
                "required": ["id"],
                "additionalProperties": false
            },
            "plate": {
                "allOf": [
                    {"$ref": "#/definitions/item"},
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
                    {"$ref": "#/definitions/item"},
                    {
                        "properties": {
                            "kind": {"const": "tipbox"}
                        },
                        "required": ["kind"]
                    }
                ]
            }
        },
        "oneOf": [
            {"$ref": "#/definitions/plate"},
            {"$ref": "#/definitions/tipbox"}
        ]
    }"##;

    schema_accepts(
        schema,
        &[
            r#"{"id": "x", "kind": "plate"}"#,
            r#"{"id": "x", "kind": "tipbox"}"#,
        ],
    );
    schema_rejects(
        schema,
        &[
            r#"{"id": "x"}"#,
            r#"{"id": "x", "kind": "other"}"#,
        ],
    );
}

#[test]
fn test_o47674_top_level_oneof_matches_llguidance_permissiveness() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema_path = Path::new(
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/maskbench/data/Github_hard---o47674.json",
    );
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(schema_path).expect("o47674 fixture should be readable"),
    )
    .expect("o47674 fixture should parse");
    let schema = fixture
        .get("schema")
        .expect("fixture should contain schema")
        .to_string();

    let _constraint = schema_constraint(&schema);
}

#[test]
fn test_minimized_o47674_shape_still_reaches_eighteen_paths() {
    let schema = r#"{
        "type": "object",
        "patternProperties": {
            "[a-z_0-9\\.\\-]*": {
                "type": "object",
                "properties": {
                    "uuid": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "command": {"type": "array"},
                    "results": {
                        "patternProperties": {
                            "[a-z_0-9\\.\\-]*": {
                                "type": "object",
                                "properties": {
                                    "uuid": {"type": "string"},
                                    "parent_id": {"type": "string"},
                                    "type": {"enum": ["string", "hex_string"]},
                                    "display": {"enum": ["string", "hex_string"]},
                                    "start_pos": {"type": "integer"},
                                    "length": {"type": "integer", "minimum": 1},
                                    "mask": {"type": "string"},
                                    "levels": {"type": "object"},
                                    "rpn": {"type": "string"},
                                    "units": {"type": "string"}
                                },
                                "required": ["uuid"],
                                "anyOf": [
                                    {"required": ["parent_id"]},
                                    {"required": ["type", "display", "start_pos", "length", "units"]}
                                ],
                                "additionalProperties": false
                            }
                        }
                    }
                },
                "required": ["uuid"],
                "anyOf": [
                    {"required": ["parent_id"]},
                    {"required": ["command", "results"]}
                ],
                "additionalProperties": false
            }
        }
    }"#;

    // Print the grammar
    let glrm = crate::dump_json_schema_grammar_glrm(&schema).unwrap();
    println!("Grammar:\n{}", glrm);

    let prefix = br#"{"op":"#;
    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(&constraint, prefix);
    assert_eq!(max_paths, 1);

    let prefix = br#"{"op": "#;
    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(&constraint, prefix);
    assert_eq!(max_paths, 2);

    let prefix = br#"{"op": {""#;
    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(&constraint, prefix);
    assert_eq!(max_paths, 3);

    let prefix = br#"{"op": {"uuid": "u", "#;
    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(&constraint, prefix);
    assert_eq!(max_paths, 6);

    let prefix = br#"{"op": {"uuid": "u", "command": [], "results": {"r1": {"uuid": "u", "#;
    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(&constraint, prefix);
    assert_eq!(
        max_paths, 18,
        "simplified o47674-shaped schema should still reach 18 live parser paths"
    );
}

#[test]
fn test_o29389_top_level_oneof_matches_llguidance_permissiveness() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema_path = Path::new(
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/maskbench/data/Github_hard---o29389.json",
    );
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(schema_path).expect("o29389 fixture should be readable"),
    )
    .expect("o29389 fixture should parse");
    let schema = fixture
        .get("schema")
        .expect("fixture should contain schema")
        .to_string();

    let _constraint = schema_constraint(&schema);
}

#[test]
fn test_nested_dynamic_object_prefix_stays_single_path() {
    let schema = r#"{
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "object",
                    "additionalProperties": {"type": "boolean"}
                },
                "patches": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string"},
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            }
        }
    }"#;
    let prefix = br#"{"kas": {"layers": {"meta-kas": true, "meta-openembedded": true}, "patches": {"kas-patch": {"repo": "r", "path": "p"}}"#;

    let constraint = schema_constraint(schema);
    let max_paths = max_parser_paths_over_prefix(&constraint, prefix);

    assert_eq!(
        max_paths, 1,
        "pure dynamic-key object lowering should avoid close-brace path waves"
    );
}

#[test]
fn test_minimized_dynamic_object_prefix_has_two_paths() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "p": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
        }
    }"#;
    let prefix = br#"{"p": {"q": "y"}"#;

    let constraint = schema_constraint(schema);
    let max_paths = trace_parser_stacks_over_prefix(&constraint, prefix);

    assert_eq!(
        max_paths, 2,
        "optional named-property plus dynamic-object lowering should still expose the reduced two-path ambiguity"
    );
}

#[test]
fn test_closed_object_anyof_shared_key_value_variants_preserve_language() {
    let _guard = env_lock().lock().expect("env lock should not be poisoned");
    let schema = r#"{
        "type": "object",
        "properties": {
            "uuid": {"type": "string"},
            "name": {"type": "string"},
            "parent_id": {"type": ["string", "null"]}
        },
        "required": ["uuid"],
        "additionalProperties": false,
        "anyOf": [
            {
                "properties": {
                    "parent_id": {"type": "null"}
                },
                "required": ["parent_id"]
            },
            {
                "properties": {
                    "parent_id": {"type": "string"}
                },
                "required": ["name", "parent_id"]
            }
        ]
    }"#;

    schema_accepts(
        schema,
        &[
            r#"{"uuid": "u", "parent_id": null}"#,
            r#"{"uuid": "u", "name": "n", "parent_id": "p"}"#,
        ],
    );
    schema_rejects(
        schema,
        &[
            r#"{"uuid": "u"}"#,
            r#"{"uuid": "u", "parent_id": "p"}"#,
        ],
    );
}

/// Adapted from `test_conversion_enum`.
///
/// Checks that an enum schema produces grammar rules containing the
/// literal enum values.
#[test]
fn test_conversion_enum() {
    let schema = r#"{
        "enum": ["red", "green", "blue"]
    }"#;
    let named = named_grammar_from_schema(schema);
    assert!(!named.rules.is_empty());

    // After opening-quote separation, string literals are split:
    // literal_expr(b"\"") + literal_expr(b"red\"") etc.
    let has_red = named.rules.iter().any(|r| contains_literal(&r.expr, b"red\""));
    let has_green = named.rules.iter().any(|r| contains_literal(&r.expr, b"green\""));
    let has_blue = named.rules.iter().any(|r| contains_literal(&r.expr, b"blue\""));

    assert!(has_red, "Grammar should contain literal body for red");
    assert!(has_green, "Grammar should contain literal body for green");
    assert!(has_blue, "Grammar should contain literal body for blue");
}

/// Adapted from the original `$ref` conversion regression.
///
/// Checks that `$ref` and `$defs` are resolved correctly, producing a valid grammar.
#[test]
fn test_conversion_resolves_ref() {
    let schema = r##"{
        "$defs": {
            "person": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        },
        "type": "array",
        "items": {"$ref": "#/$defs/person"}
    }"##;
    let named = named_grammar_from_schema(schema);
    assert!(!named.rules.is_empty());
}

#[test]
fn test_conversion_merges_property_key_and_colon() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"],
        "additionalProperties": true
    }"#;
    let named = named_grammar_from_schema(schema);

    assert!(
        named.rules.iter().any(|r| r.name == "JSON_KEY_COLON_BODY"),
        "CFA-style lowering should include a shared JSON_KEY_COLON_BODY terminal rule"
    );
    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"name\"")),
        "Known object properties should use merged key+close-quote body literals (without colon-space)"
    );
}

#[test]
fn test_conversion_supports_definitions_ref() {
    let schema = r##"{
        "definitions": {
            "Point": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"}
                },
                "required": ["x", "y"]
            }
        },
        "$ref": "#/definitions/Point"
    }"##;
    let named = named_grammar_from_schema(schema);

    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"x\"")),
        "Resolved definitions ref should contribute merged body literal for x"
    );
    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"y\"")),
        "Resolved definitions ref should contribute merged body literal for y"
    );
}

#[test]
fn test_conversion_allof_merges_object_properties() {
    let schema = r#"{
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string"}
                },
                "required": ["a"]
            },
            {
                "type": "object",
                "properties": {
                    "b": {"type": "integer"}
                },
                "required": ["b"],
                "additionalProperties": false
            }
        ]
    }"#;
    let named = named_grammar_from_schema(schema);

    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"a\"")),
        "allOf merge should preserve property a body literal"
    );
    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"b\"")),
        "allOf merge should preserve property b body literal"
    );
}

#[test]
fn test_prefix_items_follow_optional_tuple_semantics() {
    let schema = r#"{
        "type": "array",
        "prefixItems": [
            {"const": 1},
            {"const": 2}
        ]
    }"#;
    let vocab = Vocab::new(
        vec![
            (0, b"[]".to_vec()),
            (1, b"[1]".to_vec()),
            (2, b"[1, 2]".to_vec()),
        ],
        None,
    );
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut state = c.start();
    let mask = state.mask();

    assert!(
        token_allowed(&mask, 0),
        "prefixItems should allow omitting trailing tuple positions by default"
    );
    assert!(
        token_allowed(&mask, 1),
        "prefixItems should allow consuming only the first tuple position by default"
    );
    assert!(
        token_allowed(&mask, 2),
        "prefixItems should allow the full tuple payload"
    );

    state.commit_token(1).unwrap();
    assert!(state.is_finished(), "[1] should finish successfully");

    let mut state = c.start();
    state.commit_token(2).unwrap();
    assert!(state.is_finished(), "[1,2] should finish successfully");
}

#[test]
fn test_date_format_rejects_impossible_february_day_prefix() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "end_date": {
                "type": ["string", "null"],
                "format": "date"
            }
        }
    }"#;
    let c = schema_constraint(schema);
    let mut state = c.start();
    advance_byte_prefix(&mut state, br#"{"end_date": "2020-02-"#);
    let mask = state.mask();
    assert_token_allowed(
        &mask,
        b'2' as usize,
        "day prefix '2' should remain valid because 20-29 can still complete",
    );
    assert_token_disallowed(
        &mask,
        b'3' as usize,
        "day prefix '3' should be rejected for February because only 30/31 remain",
    );
}

#[test]
fn test_false_schema_property_is_omitted_when_additional_properties_are_forbidden() {
    let schema = r#"{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "additionalProperties": false,
            "paging": {
                "type": "object",
                "properties": {
                    "uri": {"type": "string"}
                },
                "required": ["uri"]
            },
            "results": {}
        },
        "required": ["paging"]
    }"#;
    let c = schema_constraint(schema);
    let mut state = c.start();
    advance_byte_prefix(&mut state, b"{\"");
    let mask = state.mask();
    assert_token_disallowed(
        &mask,
        b'a' as usize,
        "false-schema property should not contribute the impossible additionalProperties key",
    );
    assert_token_allowed(
        &mask,
        b'p' as usize,
        "real declared keys should remain available",
    );
}

#[test]
fn test_dotted_required_property_name_restricts_key_prefix() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "PersonController.personal": {
                "type": "object",
                "properties": {
                    "model": {},
                    "request": {}
                },
                "required": ["model", "request"],
                "additionalProperties": false
            }
        },
        "required": ["PersonController.personal"],
        "additionalProperties": false
    }"#;
    let c = schema_constraint(schema);
    let mut state = c.start();
    advance_byte_prefix(&mut state, b"{\"");
    let mask = state.mask();

    assert_token_allowed(
        &mask,
        b'P' as usize,
        "declared dotted key should remain available",
    );
    assert_token_disallowed(
        &mask,
        b'!' as usize,
        "undeclared key prefixes should be rejected when additionalProperties is false",
    );
}

#[test]
fn test_dotted_required_property_name_restricts_token_vocab_prefix() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "PersonController.personal": {
                "type": "object",
                "properties": {
                    "model": {},
                    "request": {}
                },
                "required": ["model", "request"],
                "additionalProperties": false
            }
        },
        "required": ["PersonController.personal"],
        "additionalProperties": false
    }"#;
    let vocab = Vocab::new(
        vec![
            (1u32, b"{".to_vec()),
            (2u32, b"\"".to_vec()),
            (3u32, b"PersonController.personal".to_vec()),
            (4u32, b"!".to_vec()),
        ],
        None,
    );
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut state = c.start();
    advance_tokens(&mut state, &[1, 2]);

    let mask = state.mask();
    assert!(
        token_allowed(&mask, 3),
        "declared dotted key token should remain available"
    );
    assert!(
        !token_allowed(&mask, 4),
        "undeclared key token should be rejected when additionalProperties is false"
    );
}

#[test]
fn test_required_only_untyped_object_delays_extra_keys_until_required_keys_are_satisfied() {
    let schema = r#"{
        "type": "array",
        "items": {
            "host": {"type": "string"},
            "port": {"type": "integer"},
            "required": ["host", "port"]
        }
    }"#;
    let c = schema_constraint(schema);

    let mut key_state = c.start();
    advance_byte_prefix(&mut key_state, b"[{\"");
    let key_mask = key_state.mask();
    assert_token_allowed(
        &key_mask,
        b'h' as usize,
        "the required host key should remain available at object start",
    );
    assert_token_allowed(
        &key_mask,
        b'p' as usize,
        "the required port key should remain available at object start",
    );
    assert_token_disallowed(
        &key_mask,
        b'!' as usize,
        "free-form object keys should be delayed until the required keys are satisfied",
    );

    let mut value_state = c.start();
    advance_byte_prefix(&mut value_state, b"[{\"host\": \"\"");
    let value_mask = value_state.mask();
    assert_token_disallowed(
        &value_mask,
        b'}' as usize,
        "object closure should remain invalid until the required port key appears",
    );

    let mut extra_key_state = c.start();
    advance_byte_prefix(&mut extra_key_state, b"[{\"host\": \"\", \"port\": 1, \"");
    let extra_key_mask = extra_key_state.mask();
    assert_token_allowed(
        &extra_key_mask,
        b'!' as usize,
        "free-form object keys should become available once all required keys are satisfied",
    );
}

#[test]
fn test_pattern_length_constraints_bound_string_content() {
    let schema = r##"{
        "type": "object",
        "properties": {
            "clientId": {
                "type": "string",
                "minLength": 12,
                "maxLength": 12,
                "pattern": "^[0-9a-fA-F]+$"
            },
            "secret": {
                "type": "string",
                "minLength": 30,
                "pattern": "^[ !\"#$%&\\'()*+,\\-./0-9:;<=>?@A-Z\\[\\\\\\]\\^_`a-z{\\|}]+$"
            }
        }
    }"##;
    let c = schema_constraint(schema);

    let mut client_id_state = c.start();
    advance_byte_prefix(&mut client_id_state, br#"{"clientId": "0123456789ab"#);
    let client_id_mask = client_id_state.mask();
    assert_token_allowed(
        &client_id_mask,
        b'"' as usize,
        "closing quote should be allowed once the fixed-length hex string is complete",
    );
    assert_token_disallowed(
        &client_id_mask,
        b'c' as usize,
        "extra hex characters should be rejected once maxLength is reached",
    );

    let mut secret_state = c.start();
    advance_byte_prefix(&mut secret_state, br#"{"secret": "abcdefghijklmnopqrstuvwxyz012"#);
    let secret_mask = secret_state.mask();
    assert_token_disallowed(
        &secret_mask,
        b'"' as usize,
        "closing quote should remain invalid before minLength is reached",
    );
}

/// Minimized from Snowplow `sp_367_Normalized`.
///
/// With GPT-2-like vocabularies, this small 3-field object can still trigger a
/// disproportionate compile-time blowup driven by the `host` hostname/ip
/// formats interacting with other required bounded strings.
#[test]
fn test_snowplow_host_name_username_schema_accepts_hostname_object() {
    let schema = r#"{
        "type": "object",
        "additionalProperties": false,
        "properties": {
            "host": {
                "type": "string",
                "anyOf": [
                    {"format": "hostname"},
                    {"format": "ipv4"},
                    {"format": "ipv6"}
                ]
            },
            "name": {
                "type": "string",
                "maxLength": 255
            },
            "username": {
                "type": "string",
                "maxLength": 64
            }
        },
        "required": ["host", "name", "username"]
    }"#;

    schema_accepts(
        schema,
        &[r#"{"host": "example.com", "name": "x", "username": "u"}"#],
    );
}

#[test]
fn test_hostname_format_accepts_max_label_length() {
    let schema = r#"{
        "type": "string",
        "format": "hostname"
    }"#;
    let valid = format!(r#""{}.com""#, "a".repeat(63));
    schema_accepts(schema, &[valid.as_str()]);
}

#[test]
fn test_hostname_format_rejects_label_length_above_63() {
    let schema = r#"{
        "type": "string",
        "format": "hostname"
    }"#;
    let invalid = format!(r#""{}.com""#, "a".repeat(64));
    schema_rejects(schema, &[invalid.as_str()]);
}

#[test]
fn test_date_or_null_schema_rejects_empty_string_span_token() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "start_date": {
                "type": ["string", "null"],
                "format": "date"
            }
        }
    }"#;
    let vocab = Vocab::new(vec![(13538u32, b" \"\"".to_vec())], None);
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut state = c.start();
    state.commit_bytes(br#"{"start_date":"#).unwrap();
    let mask = state.mask();
    assert!(
        !token_allowed(&mask, 13538),
        "span token b' \"\"' must be rejected after '{{\"start_date\":' because empty string is not a valid date"
    );
}

#[ignore]
#[test]
fn test_pattern_with_min_length_rejects_empty_string_span_token() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "minLength": 1,
                "maxLength": 5000,
                "pattern": "^$|(^(?:\\S+\\s+){0,99}\\S+$)"
            }
        }
    }"#;
    let vocab = Vocab::new(vec![(13538u32, b" \"\"".to_vec())], None);
    let c = schema_constraint_with_vocab(schema, &vocab);
    let mut state = c.start();
    state.commit_bytes(br#"{"question":"#).unwrap();
    let mask = state.mask();
    assert!(
        !token_allowed(&mask, 13538),
        "span token b' \"\"' must be rejected after '{{\"question\":' because minLength=1 removes the pattern's empty-string branch"
    );
}

#[ignore]
#[test]
fn test_group_wrapped_anchored_pattern_rejects_leading_space() {
    let schema = r#"{
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "minLength": 1,
                "maxLength": 5000,
                "pattern": "^$|(^(?:\\S+\\s+){0,99}\\S+$)"
            }
        }
    }"#;
    let c = schema_constraint(schema);
    let mut state = c.start();
    state.commit_bytes(br#"{"question": ""#).unwrap();
    let mask = state.mask();
    assert_token_disallowed(
        &mask,
        b' ' as usize,
        "a leading space must be rejected after '{{\"question\": \"' because the anchored branch starts with \\S",
    );
    assert_token_allowed(
        &mask,
        b'W' as usize,
        "a non-whitespace leading character should remain allowed after '{{\"question\": \"'",
    );
}

#[test]
fn test_large_max_length_string_respects_min_length() {
    let schema = r#"{
        "type": "string",
        "minLength": 8,
        "maxLength": 2048
    }"#;

    let vocab = Vocab::new(
        vec![
            (1u32, br#""Pass""#.to_vec()),
            (2u32, br#""Password123""#.to_vec()),
        ],
        None,
    );
    let c = schema_constraint_with_vocab(schema, &vocab);
    let state = c.start();
    let mask = state.mask();
    assert_token_disallowed(
        &mask,
        1,
        "short string token must be rejected when minLength=8",
    );
    assert_token_allowed(
        &mask,
        2,
        "long-enough string token should remain allowed when minLength=8",
    );
}

include!("/tmp/cfa_rust_test_o9788.rs");
include!("/tmp/minimize_o9788_bug.rs");
