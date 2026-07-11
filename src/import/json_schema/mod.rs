mod array;
mod ast;
mod combinators;
mod config;
mod error;
mod load;
mod lower;
mod number;
mod object;
mod preflight;
pub(crate) mod string;

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::collections::HashMap;
use std::env;

use serde_json::{Map, Value};

use crate::GlrMaskError;
use crate::grammar::ast::resolved_named_terminal_exprs;
use crate::grammar::exact_subtraction_lowering::lower_exact_subtractions;
use crate::grammar::named_simplify::simplify_named_grammar;
use crate::grammar::terminal_choice_promotion::promote_choice_terminals_exact;
use crate::import::ast::NamedGrammar;

use self::config::JsonSchemaConfig;
use self::load::{load_document_with_features, scan_document_features};
use self::lower::lower_document;

const JSON_PATTERN_SINGLETONS_DEFAULT: bool = true;

fn json_pattern_singletons_enabled() -> bool {
    match env::var("GLRMASK_JSON_PATTERN_SINGLETONS") {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            other => panic!(
                "invalid GLRMASK_JSON_PATTERN_SINGLETONS={other:?}; expected one of 1/0, true/false, yes/no, or on/off"
            ),
        },
        Err(_) => JSON_PATTERN_SINGLETONS_DEFAULT,
    }
}

fn is_pattern_partition(partition: &str) -> bool {
    partition == lower::JSON_PATTERN_LEXER_PARTITION
        || partition.starts_with("json_pattern_")
}

fn finalize_lexer_partitions_with_options(
    grammar: &mut NamedGrammar,
    pattern_singletons: bool,
) -> crate::Result<()> {
    let previous_partitions = std::mem::take(&mut grammar.lexer_partitions);
    let resolved_terminals = resolved_named_terminal_exprs(grammar)?;
    let mut pattern_partitions = HashMap::new();

    grammar.default_lexer_partition = None;
    for rule in grammar
        .rules
        .iter()
        .filter(|rule| rule.is_terminal && !rule.is_internal)
    {
        let previous = previous_partitions.get(&rule.name).map(String::as_str);
        let partition = if previous == Some(lower::JSON_LITERAL_LEXER_PARTITION) {
            lower::JSON_LITERAL_LEXER_PARTITION.to_string()
        } else if previous.is_some_and(is_pattern_partition) {
            if pattern_singletons {
                let terminal_expr = resolved_terminals
                    .get(&rule.name)
                    .expect("resolved emitting JSON terminal expression");
                pattern_partitions
                    .entry(terminal_expr.clone())
                    .or_insert_with(|| format!("json_pattern_{}", rule.name))
                    .clone()
            } else {
                lower::JSON_PATTERN_LEXER_PARTITION.to_string()
            }
        } else {
            lower::JSON_OTHER_LEXER_PARTITION.to_string()
        };
        grammar.lexer_partitions.insert(rule.name.clone(), partition);
    }
    let literals = grammar.emitted_anonymous_literals();
    grammar.set_literal_lexer_partition(lower::JSON_LITERAL_LEXER_PARTITION, literals);
    Ok(())
}

pub(crate) fn finalize_lexer_partitions(grammar: &mut NamedGrammar) -> crate::Result<()> {
    finalize_lexer_partitions_with_options(grammar, json_pattern_singletons_enabled())
}

pub(crate) fn prepare_named_grammar(grammar: &mut NamedGrammar) -> crate::Result<()> {
    if simplify_grammar_enabled() {
        simplify_named_grammar(grammar);
    }
    if lower_exact_subtractions_enabled() {
        lower_exact_subtractions(grammar)?;
    }
    if promote_literal_choices_enabled() {
        promote_choice_terminals_exact(grammar, false);
    }
    finalize_lexer_partitions(grammar)?;
    Ok(())
}

pub(crate) fn prepare_named_grammar_for_dump(grammar: &mut NamedGrammar) -> crate::Result<()> {
    if simplify_grammar_enabled() {
        simplify_named_grammar(grammar);
    }
    if promote_literal_choices_enabled() {
        promote_choice_terminals_exact(grammar, false);
    }
    finalize_lexer_partitions(grammar)?;
    Ok(())
}

#[cfg(test)]
mod lexer_partition_policy_tests {
    use super::{
        finalize_lexer_partitions_with_options,
        lower,
        JSON_PATTERN_SINGLETONS_DEFAULT,
    };
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};

    fn terminal(name: &str, expr: GrammarExpr) -> NamedRule {
        NamedRule {
            name: name.to_string(),
            expr,
            is_terminal: true,
            is_internal: false,
        }
    }

    #[test]
    fn final_pattern_singletons_follow_resolved_terminal_identity() {
        assert!(JSON_PATTERN_SINGLETONS_DEFAULT);
        let mut grammar = NamedGrammar {
            rules: vec![
                terminal("A", GrammarExpr::RawRegex("[a-z]+".to_string())),
                terminal(
                    "B",
                    GrammarExpr::Grouped(Box::new(GrammarExpr::RawRegex(
                        "[a-z]+".to_string(),
                    ))),
                ),
                terminal("C", GrammarExpr::RawRegex("[0-9]+".to_string())),
                terminal("L", GrammarExpr::Literal(b"literal".to_vec())),
                terminal("O", GrammarExpr::RawRegex("-?[0-9]+".to_string())),
            ],
            start: "A".to_string(),
            ignore: None,
            lexer_partitions: [
                ("A".to_string(), lower::JSON_PATTERN_LEXER_PARTITION.to_string()),
                ("B".to_string(), lower::JSON_PATTERN_LEXER_PARTITION.to_string()),
                ("C".to_string(), lower::JSON_PATTERN_LEXER_PARTITION.to_string()),
                ("L".to_string(), lower::JSON_LITERAL_LEXER_PARTITION.to_string()),
                ("O".to_string(), lower::JSON_OTHER_LEXER_PARTITION.to_string()),
            ]
            .into_iter()
            .collect(),
            lexer_literal_partitions: Default::default(),
            default_lexer_partition: None,
        };

        finalize_lexer_partitions_with_options(&mut grammar, true).unwrap();
        assert_eq!(grammar.lexer_partitions["A"], grammar.lexer_partitions["B"]);
        assert_ne!(grammar.lexer_partitions["A"], grammar.lexer_partitions["C"]);
        assert_eq!(
            grammar.lexer_partitions["L"],
            lower::JSON_LITERAL_LEXER_PARTITION
        );
        assert_eq!(grammar.lexer_partitions["O"], lower::JSON_OTHER_LEXER_PARTITION);

        let singleton_partitions = grammar.lexer_partitions.clone();
        finalize_lexer_partitions_with_options(&mut grammar, true).unwrap();
        assert_eq!(grammar.lexer_partitions, singleton_partitions);

        finalize_lexer_partitions_with_options(&mut grammar, false).unwrap();
        assert_eq!(grammar.lexer_partitions["A"], lower::JSON_PATTERN_LEXER_PARTITION);
        assert_eq!(grammar.lexer_partitions["B"], lower::JSON_PATTERN_LEXER_PARTITION);
        assert_eq!(grammar.lexer_partitions["C"], lower::JSON_PATTERN_LEXER_PARTITION);
    }
}

/// Convert a JSON Schema value into the project's named grammar AST.
///
/// The implementation intentionally has two phases:
///
/// 1. [`load_document`] parses serde_json data into a typed schema AST.
/// 2. [`lower_document`] lowers that schema AST into `GrammarExpr` rules.
///
/// Unsupported schema keywords are rejected while loading so the lowering phase
/// is not forced to carry partially-understood JSON values.
pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some();
    let total_started_at = profile_enabled.then(std::time::Instant::now);
    let config = JsonSchemaConfig::from_env();
    // This scan is also reused by typed loading, avoiding separate walks for
    // oneOf coercion, definitions, local aliases, and references.
    let document_features = scan_document_features(schema);
    // Coercion is default-on, but the large majority of schemas do not contain
    // oneOf. Preserve the source Value unless there is an actual rewrite.
    let imported_schema: Cow<'_, Value> = if config.coerce_one_of_to_any_of
        && document_features.has_one_of
    {
        Cow::Owned(coerce_one_of_to_any_of_schema(schema))
    } else {
        Cow::Borrowed(schema)
    };
    let preflight_started_at = profile_enabled.then(std::time::Instant::now);
    preflight::check_schema_preflight(imported_schema.as_ref()).map_err(GlrMaskError::from)?;
    let preflight_ms = preflight_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let load_started_at = profile_enabled.then(std::time::Instant::now);
    let document = load_document_with_features(imported_schema.as_ref(), &document_features)
        .map_err(GlrMaskError::from)?;
    let load_ms = load_started_at
        .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let lower_started_at = profile_enabled.then(std::time::Instant::now);
    let grammar = lower_document(&document, config).map_err(GlrMaskError::from)?;
    if let Some(total_started_at) = total_started_at {
        eprintln!(
            "[glrmask/profile][json_schema_import] preflight_ms={:.3} load_ms={:.3} lower_ms={:.3} total_ms={:.3}",
            preflight_ms,
            load_ms,
            lower_started_at
                .map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(grammar)
}


fn coerce_one_of_to_any_of_schema(schema: &Value) -> Value {
    coerce_one_of_to_any_of_schema_node(schema)
}

fn coerce_one_of_to_any_of_schema_node(node: &Value) -> Value {
    let Value::Object(object) = node else {
        return node.clone();
    };

    let mut out = Map::new();
    for (key, value) in object {
        if key == "oneOf" {
            continue;
        }
        out.insert(key.clone(), coerce_one_of_child(key, value));
    }

    let Some(Value::Array(one_of)) = object.get("oneOf") else {
        if let Some(value) = object.get("oneOf") {
            out.insert("oneOf".to_string(), value.clone());
        }
        return Value::Object(out);
    };

    let coerced = Value::Array(
        one_of
            .iter()
            .map(coerce_one_of_to_any_of_schema_node)
            .collect(),
    );
    if out.contains_key("anyOf") {
        match out.get_mut("allOf") {
            Some(Value::Array(all_of)) => all_of.push(Value::Object(Map::from_iter([(
                "anyOf".to_string(),
                coerced,
            )]))),
            _ => {
                out.insert(
                    "allOf".to_string(),
                    Value::Array(vec![Value::Object(Map::from_iter([(
                        "anyOf".to_string(),
                        coerced,
                    )]))]),
                );
            }
        }
    } else {
        out.insert("anyOf".to_string(), coerced);
    }
    Value::Object(out)
}

fn coerce_one_of_child(key: &str, value: &Value) -> Value {
    match key {
        "const" | "default" | "enum" | "examples" => value.clone(),
        "$defs" | "definitions" | "dependentSchemas" | "dependencies"
        | "patternProperties" | "properties" => coerce_one_of_schema_map(value),
        "additionalItems" | "additionalProperties" | "contains" | "contentSchema"
        | "else" | "if" | "items" | "not" | "propertyNames" | "then"
        | "unevaluatedItems" | "unevaluatedProperties" => coerce_one_of_schema_or_tuple(value),
        "allOf" | "anyOf" | "prefixItems" => coerce_one_of_schema_array(value),
        _ => coerce_one_of_extension_value(value),
    }
}

fn coerce_one_of_schema_map(value: &Value) -> Value {
    let Value::Object(object) = value else {
        return value.clone();
    };
    Value::Object(Map::from_iter(object.iter().map(|(key, child)| {
        (key.clone(), coerce_one_of_to_any_of_schema_node(child))
    })))
}

fn coerce_one_of_schema_array(value: &Value) -> Value {
    let Value::Array(items) = value else {
        return value.clone();
    };
    Value::Array(items.iter().map(coerce_one_of_to_any_of_schema_node).collect())
}

fn coerce_one_of_schema_or_tuple(value: &Value) -> Value {
    match value {
        Value::Object(_) => coerce_one_of_to_any_of_schema_node(value),
        Value::Array(_) => coerce_one_of_schema_array(value),
        _ => value.clone(),
    }
}

fn coerce_one_of_extension_value(value: &Value) -> Value {
    match value {
        Value::Object(_) => coerce_one_of_to_any_of_schema_node(value),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|child| match child {
                    Value::Object(_) => coerce_one_of_to_any_of_schema_node(child),
                    _ => child.clone(),
                })
                .collect(),
        ),
        _ => value.clone(),
    }
}

/// The new importer deliberately does not depend on the old post-import grammar
/// simplification pass.
pub(crate) fn simplify_grammar_enabled() -> bool {
    false
}

/// Exact terminal subtraction lowering is disabled by default.
///
/// Set `GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS=1` (or any non-empty,
/// non-falsey value) to enable exact-subtraction lowering in downstream import
/// and compile paths.
///
/// Note: JSON Schema GLRM dumps preserve exact subtraction syntax and do not
/// apply this lowering pass.
pub(crate) fn lower_exact_subtractions_enabled() -> bool {
    match env::var("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS") {
        Ok(value) => {
            let trimmed = value.trim();
            !trimmed.is_empty()
                && !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
        }
        Err(_) => false,
    }
}

/// Split fixed JSON literal terminals at shared structural boundaries. Default OFF.
///
/// Set `GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS=1` (or any non-empty,
/// non-falsey value) to enable the split key and string-literal terminals.
pub(crate) const GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS_ENV: &str =
    "GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS";

#[cfg(test)]
std::thread_local! {
    static SPLIT_LITERAL_TERMINALS_TEST_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn swap_split_literal_terminals_test_override(
    value: Option<bool>,
) -> Option<bool> {
    SPLIT_LITERAL_TERMINALS_TEST_OVERRIDE.with(|override_value| override_value.replace(value))
}

pub(crate) fn split_literal_terminals_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) =
        SPLIT_LITERAL_TERMINALS_TEST_OVERRIDE.with(std::cell::Cell::get)
    {
        return value;
    }

    // Process-fixed knob (the toggle test exercises it via child processes, like
    // `unanchored_pattern_split_mode`). Cache it so the per-string-terminal call
    // sites in object/string/lower do not re-read the environment.
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| match env::var(GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS_ENV) {
        Ok(value) => {
            let trimmed = value.trim();
            !trimmed.is_empty()
                && !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
        }
        Err(_) => false,
    })
}

/// Fold additional-property excluded-key add-backs into the shared terminal
/// instead of emitting one parser alternative per excluded key. Default ON.
/// Disable with GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK=0 (or false/no/off/empty).
pub(crate) fn share_additional_addback_choices_enabled() -> bool {
    match env::var("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK") {
        Ok(value) => {
            let trimmed = value.trim();
            !trimmed.is_empty()
                && !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
        }
        Err(_) => true,
    }
}

/// Literal-choice promotion was an optimization knob in the old importer.  The
/// simple importer leaves choices as written.
pub(crate) fn promote_literal_choices_enabled() -> bool {
    false
}
