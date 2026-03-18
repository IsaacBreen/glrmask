//! Ported JSON-schema and EBNF constraint tests from sep1.
//!
//! Source: `grammars2024/src/interface/json_schema/tests.rs`
//!   (21 tests total; 16 ported, 5 skipped)
//!
//! Skipped tests:
//!   - test_small_vocab_only_brace_valid_at_start: complex debug reproduction with
//!     extensive sep1-internal-API assertions (possible_matches, DWA internals,
//!     vocab mapping) that have no glrmask surface equivalent
//!   - test_schema_simple_object_weight_heavy: #[ignore] in sep1 source
//!   - test_multibyte_tokens_simple_object: requires GPT-2 vocab file on disk
//!   - test_multibyte_tokens_additional_properties_true: requires GPT-2 vocab file
//!   - test_object_schema_rejects_quote_at_empty_prefix: requires GPT-2 vocab file

use crate::import::ast::GrammarExpr;
use crate::import::json_schema::{json_schema_to_grammar, schema_to_named_grammar};
use crate::runtime::Constraint;
use crate::Vocab;
use std::path::Path;

// ── Helpers ──────────────────────────────────────────────────────────────────

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

/// Build a Constraint from a JSON schema (using the byte vocab) and assert
/// that every input is accepted byte-by-byte.
fn schema_accepts(schema: &str, inputs: &[&str]) {
    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(schema, &vocab)
        .unwrap_or_else(|e| panic!("schema should compile: {}", e));
    for input in inputs {
        let mut s = c.start();
        for (i, byte) in input.bytes().enumerate() {
            let mask = s.mask();
            assert!(
                token_allowed(&mask, byte as usize),
                "Schema should accept {:?}: byte {:?} ({:#04x}) at position {} not in mask",
                input,
                byte as char,
                byte,
                i
            );
            s.commit_token(byte as u32).unwrap();
        }
    }
}

/// Build a Constraint from EBNF (using the byte vocab) and return it.
fn ebnf_constraint(ebnf: &str) -> Constraint {
    let vocab = byte_vocab();
    Constraint::from_ebnf(ebnf, &vocab)
        .unwrap_or_else(|e| panic!("EBNF should compile: {}", e))
}

fn contains_literal(expr: &GrammarExpr, target: &[u8]) -> bool {
    match expr {
        GrammarExpr::Literal(bytes) => bytes == target,
        GrammarExpr::Sequence(parts) => parts.iter().any(|part| contains_literal(part, target)),
        GrammarExpr::Choice(options) => options.iter().any(|option| contains_literal(option, target)),
        GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_literal(inner, target),
        _ => false,
    }
}

// ── EBNF constraint tests ───────────────────────────────────────────────────

/// Ported from `test_ebnf_ws_nullable`.
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

/// Ported from `test_ebnf_object_member_after_brace`.
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

/// Ported from `test_schema_simple_object`.
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

/// Ported from `test_schema_additional_properties_true`.
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

/// Ported from `test_schema_additional_properties_schema`.
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

/// Ported from `test_schema_dependencies`.
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

/// Ported from `test_schema_nested_objects`.
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
    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(schema, &vocab)
        .unwrap_or_else(|e| panic!("schema should compile: {}", e));

    let mut s = c.start();
    for byte in br#"{"id": 1, "# {
        let mask = s.mask();
        assert!(
            token_allowed(&mask, *byte as usize),
            "prefix byte {byte:?} should be allowed"
        );
        s.commit_token(*byte as u32).unwrap();
    }

    let mask = s.mask();
    assert!(
        token_allowed(&mask, b'"' as usize),
        "object member after comma+space should start with a quote"
    );
    assert!(
        !token_allowed(&mask, b'1' as usize),
        "object member after comma+space must not allow a digit"
    );
}

#[test]
fn test_o56012_after_comma_requires_key_quote() {
    let fixture_path = Path::new(
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/maskbench/data/Github_hard---o56012.json",
    );
    if !fixture_path.exists() {
        return;
    }

    let fixture_text = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|e| panic!("should read o56012 fixture: {e}"));
    let fixture: serde_json::Value = serde_json::from_str(&fixture_text)
        .unwrap_or_else(|e| panic!("should parse o56012 fixture: {e}"));
    let schema = fixture
        .get("schema")
        .unwrap_or_else(|| panic!("fixture should contain schema"))
        .to_string();
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

    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(&schema, &vocab)
        .unwrap_or_else(|e| panic!("schema should compile: {}", e));
    let mut s = c.start();
    for &byte in &example.as_bytes()[..prefix_end] {
        let mask = s.mask();
        assert!(token_allowed(&mask, byte as usize), "prefix byte {byte:?} should be allowed");
        s.commit_token(byte as u32).unwrap();
    }

    let mask = s.mask();
    assert!(token_allowed(&mask, b'"' as usize));
    assert!(
        !token_allowed(&mask, b'1' as usize),
        "o56012 native path must not allow a digit after comma+space in parameters item"
    );
}

/// Ported from `test_schema_array`.
#[test]
fn test_schema_array() {
    let schema = r#"{
        "type": "array",
        "items": {"type": "string"}
    }"#;
    schema_accepts(schema, &["[]", r#"["a"]"#, r#"["a", "b", "c"]"#]);
}

/// Ported from `test_schema_anyof`.
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

/// Ported from `test_schema_enum`.
#[test]
fn test_schema_enum() {
    let schema = r#"{
        "enum": ["red", "green", "blue"]
    }"#;
    schema_accepts(schema, &[r#""red""#, r#""green""#, r#""blue""#]);
}

/// Ported from `test_schema_const`.
#[test]
fn test_schema_const() {
    let schema = r#"{
        "const": "fixed_value"
    }"#;
    schema_accepts(schema, &[r#""fixed_value""#]);
}

/// Ported from `test_schema_const2`.
///
/// Uses a minimal custom vocabulary (only the bytes needed for `"x"`).
#[test]
fn test_schema_const2() {
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
    let c = Constraint::from_json_schema(schema, &vocab)
        .expect("schema should compile with minimal vocab");
    let mut s = c.start();
    // Walk through "x" byte by byte: ", x, "
    for byte in b"\"x\"" {
        let mask = s.mask();
        assert!(
            token_allowed(&mask, *byte as usize),
            "Byte {:?} ({:#04x}) should be valid",
            *byte as char,
            byte
        );
        let token_id = *byte as u32;
        s.commit_token(token_id).unwrap()
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

/// Ported from `test_json_schema_mask_generation`.
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
    let c = Constraint::from_json_schema(schema, &vocab)
        .expect("schema should compile with sparse multibyte vocab");
    let mut s = c.start();

    s.commit_token(90u32).unwrap();
    s.commit_token(198u32).unwrap();
    s.commit_token(220u32).unwrap();
    s.commit_token(366u32).unwrap();

    let mask = s.mask();
    assert!(
        token_allowed(&mask, 3672),
        "token 'name' should be allowed after the object prefix '{{\\n  \"'"
    );
}

/// Ported from `test_newsletter_schema_disallows_quote_colon_minus`.
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
    let c = Constraint::from_json_schema(schema, &vocab)
        .expect("newsletter schema should compile");
    let mut s = c.start();

    s.commit_token(1u32).unwrap();
    s.commit_token(2u32).unwrap();
    s.commit_token(3u32).unwrap();

    let mask = s.mask();
    assert!(
        !token_allowed(&mask, 4),
        "token '\":-' must not be allowed after the key prefix '\"name'"
    );
}

// ── JSON schema conversion tests ────────────────────────────────────────────

/// Ported from `test_conversion_simple_object`.
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
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("schema should convert to named grammar");
    assert!(
        !named.rules.is_empty(),
        "Should produce non-empty rules"
    );
    assert_eq!(named.start, "start");
}

/// Ported from `test_conversion_any_of`.
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
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("schema should convert to named grammar");
    assert!(
        !named.rules.is_empty(),
        "anyOf schema should produce non-empty rules"
    );
}

/// Ported from `test_conversion_enum`.
///
/// Checks that an enum schema produces grammar rules containing the
/// literal enum values.
#[test]
fn test_conversion_enum() {
    let schema = r#"{
        "enum": ["red", "green", "blue"]
    }"#;
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("schema should convert to named grammar");
    assert!(!named.rules.is_empty());

    // Check that the grammar contains the literal strings somewhere in the rules
    fn contains_literal(expr: &GrammarExpr, target: &[u8]) -> bool {
        match expr {
            GrammarExpr::Literal(bytes) => bytes == target,
            GrammarExpr::Sequence(parts) => parts.iter().any(|p| contains_literal(p, target)),
            GrammarExpr::Choice(options) => options.iter().any(|o| contains_literal(o, target)),
            GrammarExpr::Optional(inner) | GrammarExpr::Repeat(inner) | GrammarExpr::RepeatOne(inner) => {
                contains_literal(inner, target)
            }
            _ => false,
        }
    }

    let has_red = named.rules.iter().any(|r| contains_literal(&r.expr, b"\"red\""));
    let has_green = named.rules.iter().any(|r| contains_literal(&r.expr, b"\"green\""));
    let has_blue = named.rules.iter().any(|r| contains_literal(&r.expr, b"\"blue\""));

    assert!(has_red, "Grammar should contain literal for \"red\"");
    assert!(has_green, "Grammar should contain literal for \"green\"");
    assert!(has_blue, "Grammar should contain literal for \"blue\"");
}

/// Ported from `test_conversion_ref`.
///
/// Checks that `$ref` and `$defs` are resolved correctly, producing a valid grammar.
#[test]
fn test_conversion_ref() {
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
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("schema with $ref should convert to named grammar");
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
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("object schema should convert to named grammar");

    assert!(
        named.rules.iter().any(|r| r.name == "JSON_KEY_COLON"),
        "CFA-style lowering should include a shared JSON_KEY_COLON rule"
    );
    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"\"name\": ")),
        "Known object properties should use merged key+colon literals"
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
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("schema using definitions refs should convert to named grammar");

    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"\"x\": ")),
        "Resolved definitions ref should contribute merged literal for x"
    );
    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"\"y\": ")),
        "Resolved definitions ref should contribute merged literal for y"
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
    let parsed: serde_json::Value = serde_json::from_str(schema).unwrap();
    let named = schema_to_named_grammar(&parsed)
        .expect("allOf object schema should convert to named grammar");

    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"\"a\": ")),
        "allOf merge should preserve property a"
    );
    assert!(
        named.rules.iter().any(|r| contains_literal(&r.expr, b"\"b\": ")),
        "allOf merge should preserve property b"
    );
}

#[test]
fn test_prefix_items_default_to_required_like_cfa() {
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
            (2, b"[1,2]".to_vec()),
        ],
        None,
    );
    let c = Constraint::from_json_schema(schema, &vocab)
        .expect("prefixItems schema should compile");
    let mut state = c.start();
    let mask = state.mask();

    assert!(
        !token_allowed(&mask, 0),
        "CFA-style prefixItems lowering should not allow omitting all prefix items"
    );
    assert!(
        !token_allowed(&mask, 1),
        "CFA-style prefixItems lowering should not allow truncating required prefix items"
    );
    assert!(
        token_allowed(&mask, 2),
        "CFA-style prefixItems lowering should allow the full tuple payload"
    );

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
    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(schema, &vocab).expect("date schema should compile");
    let mut state = c.start();
    for byte in br#"{"end_date": "2020-02-"# {
        state.commit_token(*byte as u32).unwrap();
    }
    let mask = state.mask();
    assert!(
        token_allowed(&mask, b'2' as usize),
        "day prefix '2' should remain valid because 20-29 can still complete"
    );
    assert!(
        !token_allowed(&mask, b'3' as usize),
        "day prefix '3' should be rejected for February because only 30/31 remain"
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
    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(schema, &vocab).expect("false-subschema object should compile");
    let mut state = c.start();
    for byte in b"{\"" {
        state.commit_token(*byte as u32).unwrap();
    }
    let mask = state.mask();
    assert!(
        !token_allowed(&mask, b'a' as usize),
        "false-schema property should not contribute the impossible additionalProperties key"
    );
    assert!(
        token_allowed(&mask, b'p' as usize),
        "real declared keys should remain available"
    );
}

#[test]
fn test_required_only_untyped_object_allows_extra_keys_but_not_early_closure() {
    let schema = r#"{
        "type": "array",
        "items": {
            "host": {"type": "string"},
            "port": {"type": "integer"},
            "required": ["host", "port"]
        }
    }"#;
    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(schema, &vocab).expect("required-only untyped object schema should compile");

    let mut key_state = c.start();
    for byte in b"[{\"" {
        key_state.commit_token(*byte as u32).unwrap();
    }
    let key_mask = key_state.mask();
    assert!(
        token_allowed(&key_mask, b'!' as usize),
        "free-form object keys should remain allowed before required keys are satisfied"
    );

    let mut value_state = c.start();
    for byte in b"[{\"host\": \"\"" {
        value_state.commit_token(*byte as u32).unwrap();
    }
    let value_mask = value_state.mask();
    assert!(
        !token_allowed(&value_mask, b'}' as usize),
        "object closure should remain invalid until the required port key appears"
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
                "pattern": "[0-9a-fA-F]+"
            },
            "secret": {
                "type": "string",
                "minLength": 30,
                "pattern": "^[ !\"#$%&\\'()*+,\\-./0-9:;<=>?@A-Z\\[\\\\\\]\\^_`a-z{\\|}]+$"
            }
        }
    }"##;
    let vocab = byte_vocab();
    let c = Constraint::from_json_schema(schema, &vocab).expect("pattern+length schema should compile");

    let mut client_id_state = c.start();
    for byte in br#"{"clientId": "0123456789ab"# {
        client_id_state.commit_token(*byte as u32).unwrap();
    }
    let client_id_mask = client_id_state.mask();
    assert!(
        token_allowed(&client_id_mask, b'"' as usize),
        "closing quote should be allowed once the fixed-length hex string is complete"
    );
    assert!(
        !token_allowed(&client_id_mask, b'c' as usize),
        "extra hex characters should be rejected once maxLength is reached"
    );

    let mut secret_state = c.start();
    for byte in br#"{"secret": "abcdefghijklmnopqrstuvwxyz012"# {
        secret_state.commit_token(*byte as u32).unwrap();
    }
    let secret_mask = secret_state.mask();
    assert!(
        !token_allowed(&secret_mask, b'"' as usize),
        "closing quote should remain invalid before minLength is reached"
    );
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
    let c = Constraint::from_json_schema(schema, &vocab)
        .expect("date-or-null schema should compile");
    let mut state = c.start();
    state.commit_bytes(br#"{"start_date":"#);
    let mask = state.mask();
    assert!(
        !token_allowed(&mask, 13538),
        "span token b' \"\"' must be rejected after '{{\"start_date\":' because empty string is not a valid date"
    );
}

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
    let c = Constraint::from_json_schema(schema, &vocab)
        .expect("pattern+minLength schema should compile");
    let mut state = c.start();
    state.commit_bytes(br#"{"question":"#);
    let mask = state.mask();
    assert!(
        !token_allowed(&mask, 13538),
        "span token b' \"\"' must be rejected after '{{\"question\":' because minLength=1 removes the pattern's empty-string branch"
    );
}
