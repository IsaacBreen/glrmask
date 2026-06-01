use serde_json::json;
use std::{env, ffi::OsString, sync::Mutex};

use super::ast::StringSchema;
use super::lower_exact_subtractions_enabled;
use super::schema_to_named_grammar;
use super::string::{property_name_matches_pattern, string_value_satisfies_schema};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::{Action, GLRTable, TableAmbiguityKind};
use crate::grammar::ast::{lower, GrammarExpr, NamedGrammar};
use crate::grammar::glrm::to_glrm;
use crate::Vocab;

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

fn byte_vocab() -> Vocab {
    let mut entries = (0u32..=255)
        .map(|byte| (byte, vec![byte as u8]))
        .collect::<Vec<_>>();
    entries.push((256, b"<|endoftext|>".to_vec()));
    Vocab::new(entries, Some(256))
}

fn schema_accepts_bytes(schema: &serde_json::Value, input: &[u8]) -> bool {
    let grammar = schema_to_named_grammar(schema).expect("schema should import");
    let lowered = lower(&grammar).expect("schema grammar should lower");
    let constraint = crate::compiler::compile_owned(lowered, &byte_vocab());
    let mut state = constraint.start();
    state.commit_bytes(input).is_ok() && state.is_complete()
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

fn contains_ref_with_prefix(expr: &GrammarExpr, prefix: &str) -> bool {
    match expr {
        GrammarExpr::Ref(name) => name.starts_with(prefix),
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_ref_with_prefix(inner, prefix),
        GrammarExpr::RepeatRange { expr, .. } => contains_ref_with_prefix(expr, prefix),
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
        GrammarExpr::Intersect { expr, intersect } => {
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

fn contains_intersect_with_separated_sequence(expr: &GrammarExpr) -> bool {
    match expr {
        GrammarExpr::Intersect { expr, intersect } => {
            contains_separated_sequence(expr)
                || contains_separated_sequence(intersect)
                || contains_intersect_with_separated_sequence(expr)
                || contains_intersect_with_separated_sequence(intersect)
        }
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_intersect_with_separated_sequence(inner),
        GrammarExpr::RepeatRange { expr, .. } => contains_intersect_with_separated_sequence(expr),
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

fn contains_literal_bytes(expr: &GrammarExpr, bytes: &[u8]) -> bool {
    match expr {
        GrammarExpr::Literal(literal) => literal == bytes,
        GrammarExpr::Grouped(inner)
        | GrammarExpr::Optional(inner)
        | GrammarExpr::Repeat(inner)
        | GrammarExpr::RepeatOne(inner) => contains_literal_bytes(inner, bytes),
        GrammarExpr::RepeatRange { expr, .. } => contains_literal_bytes(expr, bytes),
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
        GrammarExpr::Intersect { expr, intersect } => {
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
    assert!(!matches!(parts[1], GrammarExpr::Optional(_)));
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
    assert!(glrm.contains("\\\"line1\\\": "), "{glrm}");
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
    assert!(glrm.contains("\\\"line1\\\": "), "{glrm}");
    assert!(glrm.contains("ok"), "{glrm}");
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
    assert!(glrm.contains("\\\"missing\\\": "), "{glrm}");
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
    assert!(glrm.contains("\\\"line1\\\": "), "{glrm}");
    assert!(glrm.contains("ok"), "{glrm}");
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
fn huge_shared_additional_exclusion_set_uses_expanded_literal_addback() {
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

    assert!(contains_literal_bytes(&excluded_rule.expr, b"\"open_only\": "));
    assert!(!contains_literal_bytes(&excluded_rule.expr, b"\"closed_only\": "));

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
    assert!(glrm.contains("JSON_STRING_CHAR{0,64}"), "{glrm}");
    assert!(glrm.contains("json_string_constrained"), "{glrm}");
    assert!(!glrm.contains("json_string_char_exact_50"), "{glrm}");
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
    assert!(glrm.contains("0[13578]"), "{glrm}");
    assert!(glrm.contains("02-(?:0[1-9]|1[0-9]|2[0-8])"), "{glrm}");
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
fn string_pattern_is_intersected_with_format() {
    let schema = json!({
        "type": "string",
        "format": "uuid",
        "pattern": "^abc$"
    });

    let grammar = schema_to_named_grammar(&schema).unwrap();
    let glrm = to_glrm(&grammar);
    assert!(glrm.contains("/\"(?:abc)\"/"), "{glrm}");
    assert!(glrm.contains("[0-9A-Fa-f]{8}"), "{glrm}");
    assert!(glrm.contains(" & /\"(?:[0-9A-Fa-f]{8}"), "{glrm}");
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
fn conditional_keywords_are_ignored_for_broad_lowering() {
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

    let grammar = schema_to_named_grammar(&schema).unwrap();
    lower(&grammar).unwrap();
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
fn oneof_mixed_ref_and_inline_errors() {
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
            {"type": "string"},
            {"$ref": "#/definitions/input"}
        ]
    });

    let error = schema_to_named_grammar(&schema).unwrap_err().to_string();
    assert!(error.contains("mixed $ref and inline"), "{error}");
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

    assert!(contains_literal_bytes(expr, b"\"$data\": "), "{expr:?}");
    assert!(contains_ref_named(expr, "JSON_ITEM_SEPARATOR"), "{expr:?}");
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
fn snowplow_style_string_enum_uses_factored_suffix_choice() {
    let schema = json!({
        "type": "string",
        "enum": ["INVALID_SCHEMAVER", "INVALID_IGLUURI", "INVALID_DATA_PAYLOAD", "INVALID_SCHEMA"]
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
    assert!(suffixes.contains(&GrammarExpr::Literal(b"INVALID_SCHEMAVER\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"INVALID_IGLUURI\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"INVALID_DATA_PAYLOAD\"".to_vec())));
    assert!(suffixes.contains(&GrammarExpr::Literal(b"INVALID_SCHEMA\"".to_vec())));
    assert!(!contains_literal_bytes(start_expr(&grammar), b"\"INVALID_SCHEMAVER\""), "{:?}", start_expr(&grammar));
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
    assert!(!glrm.contains("\\\"a\\\":"), "{glrm}");
    assert!(!glrm.contains("\\\"b\\\":"), "{glrm}");
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
    assert!(glrm.contains("\\\"a\\\":"), "{glrm}");
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
    assert!(glrm.contains("\\\"a\\\""), "{glrm}");
    assert!(glrm.contains("\\\"b\\\""), "{glrm}");
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
    assert!(contains_intersect(expr), "{expr:?}");
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
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
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
        glrm.lines()
            .any(|line| line.contains("json_pattern_key_colon_")
                && line.contains(" - json_pattern_key_colon_")),
        "{glrm}"
    );
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
    assert!(!glrm.contains("\"thumbnail\""), "{glrm}");
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
    assert!(glrm.contains("json_anyof_object_body"), "{glrm}");
    assert!(
        !glrm.lines().any(|line| {
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
    assert!(!glrm.contains(" & "), "{glrm}");
    assert!(glrm.contains("\\\"name\\\""), "{glrm}");
    assert!(glrm.contains("\\\"extra\\\""), "{glrm}");
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
