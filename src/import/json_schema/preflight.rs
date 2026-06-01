use std::env;

use serde_json::{Map, Value};

use super::error::{ImportResult, SchemaImportError};

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
