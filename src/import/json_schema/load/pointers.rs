//! Local pointer and local alias utilities for JSON Schema loading.
//!
//! This module is deliberately small: it understands local JSON Pointer spelling,
//! `$id`/`id` aliases that are local to the document, and recursive discovery of
//! `$ref` strings in raw JSON values.

use serde_json::{Map, Value};

pub(super) fn collect_all_ref_pointers(value: &Value, refs: &mut std::collections::BTreeSet<String>) {
    if let Some(obj) = value.as_object() {
        if let Some(r) = obj.get("$ref").and_then(Value::as_str) {
            refs.insert(r.to_string());
        }
        for val in obj.values() {
            collect_all_ref_pointers(val, refs);
        }
    } else if let Some(arr) = value.as_array() {
        for val in arr {
            collect_all_ref_pointers(val, refs);
        }
    }
}

pub(super) fn local_id_alias(object: &Map<String, Value>, location: &str) -> Option<String> {
    let alias = object
        .get("$id")
        .or_else(|| object.get("id"))
        .and_then(Value::as_str)?;
    if alias.starts_with("#") {
        return Some(alias.to_string());
    }
    if location == "#" && alias.ends_with("#") {
        return Some(alias.to_string());
    }
    None
}

pub(super) fn escape_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}
