use std::env;

use derivre::{RegexAst, RegexBuilder};
use serde_json::{Map, Value};

use super::error::{ImportResult, SchemaImportError};
use super::string::{JsonStringCompatMode, json_string_compat_mode, preprocess_ascii_shorthand};

const DEFAULT_MAX_NODES: usize = 100_000;
const ALLOW_LARGE_ENV: &str = "GLRMASK_JSON_SCHEMA_ALLOW_LARGE";
const MAX_NODES_ENV: &str = "GLRMASK_JSON_SCHEMA_MAX_NODES";

#[derive(Debug, Default)]
struct SchemaSizeMetrics {
    nodes: usize,
    objects: usize,
    arrays: usize,
    refs: usize,
    definitions: usize,
    any_of_branches: usize,
    one_of_branches: usize,
    all_of_branches: usize,
}

pub(crate) fn check_schema_preflight(schema: &Value) -> ImportResult<()> {
    check_pattern_properties_disjointness(schema)?;
    check_schema_size(schema)
}

pub(crate) fn check_schema_size(schema: &Value) -> ImportResult<()> {
    if env_flag_enabled(ALLOW_LARGE_ENV) {
        return Ok(());
    }

    let max_nodes = max_nodes_limit()?;
    let mut metrics = SchemaSizeMetrics::default();
    collect_metrics(schema, &mut metrics);

    if metrics.nodes > max_nodes {
        return Err(SchemaImportError::new(format!(
            "schema too large: nodes={} limit={} objects={} arrays={} refs={} definitions={} anyOf_branches={} oneOf_branches={} allOf_branches={}. Set {ALLOW_LARGE_ENV}=1 to bypass or raise {MAX_NODES_ENV}.",
            metrics.nodes,
            max_nodes,
            metrics.objects,
            metrics.arrays,
            metrics.refs,
            metrics.definitions,
            metrics.any_of_branches,
            metrics.one_of_branches,
            metrics.all_of_branches,
        )));
    }

    Ok(())
}

const PATTERN_PROPERTIES_CHECK_LIMIT: u64 = 10_000;

fn check_pattern_properties_disjointness(schema: &Value) -> ImportResult<()> {
    if !matches!(json_string_compat_mode(), JsonStringCompatMode::LlGuidanceNative) {
        return Ok(());
    }
    walk_schema_for_pattern_properties(schema)
}

fn walk_schema_for_pattern_properties(value: &Value) -> ImportResult<()> {
    match value {
        Value::Object(object) => {
            if let Some(pattern_properties) = object.get("patternProperties") {
                check_pattern_properties_object(pattern_properties)?;
            }
            for child in object.values() {
                walk_schema_for_pattern_properties(child)?;
            }
        }
        Value::Array(items) => {
            for item in items {
                walk_schema_for_pattern_properties(item)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
    Ok(())
}

fn check_pattern_properties_object(value: &Value) -> ImportResult<()> {
    let Some(pattern_properties) = value.as_object() else {
        return Ok(());
    };
    if pattern_properties.len() < 2 {
        return Ok(());
    }

    let patterns = pattern_properties.keys().collect::<Vec<_>>();
    let mut builder = RegexBuilder::new();
    let refs = patterns
        .iter()
        .map(|pattern| {
            let normalized = preprocess_ascii_shorthand(pattern);
            builder
                .mk_regex_for_serach(normalized.as_str())
                .map_err(|error| {
                    SchemaImportError::new(format!(
                        "invalid patternProperties regex {pattern:?}: {error}"
                    ))
                })
        })
        .collect::<ImportResult<Vec<_>>>()?;

    for (left_index, left_pattern) in patterns.iter().enumerate() {
        for (right_index, right_pattern) in patterns.iter().enumerate() {
            if left_index >= right_index {
                continue;
            }
            let intersection = builder
                .mk(&RegexAst::And(vec![
                    RegexAst::ExprRef(refs[left_index]),
                    RegexAst::ExprRef(refs[right_index]),
                ]))
                .map_err(|error| {
                    SchemaImportError::new(format!(
                        "can't determine if patternProperty regexes /{}/ and /{}/ are disjoint: {}",
                        left_pattern, right_pattern, error
                    ))
                })?;
            let mut regex = builder
                .to_regex_limited(intersection, PATTERN_PROPERTIES_CHECK_LIMIT)
                .map_err(|_| {
                    SchemaImportError::new(format!(
                        "can't determine if patternProperty regexes /{}/ and /{}/ are disjoint",
                        left_pattern, right_pattern
                    ))
                })?;
            if !regex.always_empty() {
                return Err(SchemaImportError::new(format!(
                    "patternProperty regexes /{}/ and /{}/ are not disjoint",
                    left_pattern, right_pattern
                )));
            }
        }
    }

    Ok(())
}

fn max_nodes_limit() -> ImportResult<usize> {
    match env::var(MAX_NODES_ENV) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Ok(DEFAULT_MAX_NODES);
            }
            let parsed = trimmed.parse::<usize>().map_err(|_| {
                SchemaImportError::new(format!(
                    "{MAX_NODES_ENV} must be a positive integer node limit, got {raw:?}"
                ))
            })?;
            if parsed == 0 {
                return Err(SchemaImportError::new(format!(
                    "{MAX_NODES_ENV} must be a positive integer node limit, got {raw:?}"
                )));
            }
            Ok(parsed)
        }
        Err(_) => Ok(DEFAULT_MAX_NODES),
    }
}

fn env_flag_enabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty()
                && !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
        })
        .unwrap_or(false)
}

fn collect_metrics(value: &Value, metrics: &mut SchemaSizeMetrics) {
    metrics.nodes = metrics.nodes.saturating_add(1);
    match value {
        Value::Object(object) => collect_object_metrics(object, metrics),
        Value::Array(items) => {
            metrics.arrays = metrics.arrays.saturating_add(1);
            for item in items {
                collect_metrics(item, metrics);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn collect_object_metrics(object: &Map<String, Value>, metrics: &mut SchemaSizeMetrics) {
    metrics.objects = metrics.objects.saturating_add(1);
    for (key, value) in object {
        match key.as_str() {
            "$ref" => {
                metrics.refs = metrics.refs.saturating_add(1);
            }
            "$defs" | "definitions" => {
                metrics.definitions = metrics.definitions.saturating_add(object_len(value));
            }
            "anyOf" => {
                metrics.any_of_branches = metrics.any_of_branches.saturating_add(array_len(value));
            }
            "oneOf" => {
                metrics.one_of_branches = metrics.one_of_branches.saturating_add(array_len(value));
            }
            "allOf" => {
                metrics.all_of_branches = metrics.all_of_branches.saturating_add(array_len(value));
            }
            _ => {}
        }
        collect_metrics(value, metrics);
    }
}

fn object_len(value: &Value) -> usize {
    value.as_object().map_or(0, Map::len)
}

fn array_len(value: &Value) -> usize {
    value.as_array().map_or(0, Vec::len)
}
