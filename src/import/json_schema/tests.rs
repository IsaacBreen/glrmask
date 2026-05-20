use serde_json::json;

use super::schema_to_named_grammar;
use crate::grammar::ast::{lower, GrammarExpr, NamedGrammar};
use crate::grammar::glrm::to_glrm;

fn start_expr(grammar: &NamedGrammar) -> &GrammarExpr {
    &grammar
        .rules
        .iter()
        .find(|rule| rule.name == grammar.start)
        .expect("start rule exists")
        .expr
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

#[test]
fn closed_object_lowers_to_separated_sequence() {
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
    assert!(contains_separated_sequence(start_expr(&grammar)));
    lower(&grammar).unwrap();
}

#[test]
fn open_object_has_repeat_tail_and_excludes_fixed_keys() {
    let schema = json!({
        "type": "object",
        "properties": {"kind": {"const": "event"}},
        "required": ["kind"],
        "additionalProperties": {"type": "string"}
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("JSON_STRING - \"\\\"kind\\\"\""), "{glrm}");
    assert!(glrm.contains("+?"), "{glrm}");
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
fn string_length_and_pattern_lower_as_intersection() {
    let schema = json!({
        "type": "string",
        "minLength": 2,
        "maxLength": 8,
        "pattern": "^[A-Za-z]+$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("& /[A-Za-z]+/"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn string_format_lowers_as_regex_intersection() {
    let schema = json!({
        "type": "string",
        "format": "uuid"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("[0-9A-Fa-f]{8}"), "{glrm}");
    lower(&grammar).unwrap();
}

#[test]
fn unknown_format_errors() {
    let schema = json!({
        "type": "string",
        "format": "made-up"
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("Unknown format"), "{error}");
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
fn oneof_lowers_as_documented_union() {
    let schema = json!({
        "oneOf": [
            {"const": "left"},
            {"const": "right"}
        ]
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    assert!(matches!(start_expr(&grammar), GrammarExpr::Choice(_)));
}

#[test]
fn oneof_allows_sibling_assertions() {
    let schema = json!({
        "oneOf": [
            {"type": "string", "pattern": "^left+$"},
            {"type": "string", "pattern": "^right+$"}
        ],
        "minLength": 4
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
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
