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

use std::env;

use serde_json::{Map, Value};

use crate::GlrMaskError;
use crate::import::ast::NamedGrammar;

use self::config::JsonSchemaConfig;
use self::load::load_document;
use self::lower::lower_document;

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
    let config = JsonSchemaConfig::from_env();
    let imported_schema = if config.coerce_one_of_to_any_of {
        coerce_one_of_to_any_of_schema(schema)
    } else {
        schema.clone()
    };
    preflight::check_schema_preflight(&imported_schema).map_err(GlrMaskError::from)?;
    let document = load_document(&imported_schema).map_err(GlrMaskError::from)?;
    lower_document(&document, config).map_err(GlrMaskError::from)
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

/// Split fixed JSON literal terminals at shared structural boundaries. Default ON.
///
/// Set `GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS=0` (or false/no/off/empty)
/// to restore the previous fused key and string-literal terminals.
pub(crate) const GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS_ENV: &str =
    "GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS";

pub(crate) fn split_literal_terminals_enabled() -> bool {
    match env::var(GLRMASK_JSON_SCHEMA_SPLIT_LITERAL_TERMINALS_ENV) {
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
