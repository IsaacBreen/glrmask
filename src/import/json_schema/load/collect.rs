//! Collection of local definition and local reference targets.
//!
//! This pass records schema nodes that may need their own grammar rules later.
//! It is still part of loading: it calls `load_schema_at` to build typed schema
//! nodes, but it does not allocate grammar names.

use serde_json::Value;

use super::super::diagnostics::ImportResult;
use super::super::schema::SchemaDefinition;
use super::pointers::{escape_pointer_segment, local_id_alias};
use super::typed::load_schema_at;

pub(super) fn collect_definitions(
    value: &Value,
    location: &str,
    out: &mut Vec<SchemaDefinition>,
) -> ImportResult<()> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };

    for container_key in ["$defs", "definitions"] {
        if let Some(defs) = object.get(container_key).and_then(Value::as_object) {
            for (name, schema_value) in defs {
                let pointer = format!("{location}/{}/{}", escape_pointer_segment(container_key), escape_pointer_segment(name));
                out.push(SchemaDefinition {
                    pointer: pointer.clone(),
                    schema: load_schema_at(schema_value, &pointer)?,
                });
                collect_definitions(schema_value, &pointer, out)?;
            }
        }
    }

    for (key, child) in object {
        if matches!(key.as_str(), "$defs" | "definitions") {
            continue;
        }
        let child_location = format!("{location}/{}", escape_pointer_segment(key));
        if matches!(key.as_str(), "properties" | "patternProperties") {
            if let Some(children) = child.as_object() {
                for (name, schema_value) in children {
                    let schema_location = format!(
                        "{child_location}/{}",
                        escape_pointer_segment(name)
                    );
                    collect_definitions(schema_value, &schema_location, out)?;
                }
                continue;
            }
        }
        collect_definitions(child, &child_location, out)?;
    }
    Ok(())
}

pub(super) fn collect_ref_targets(
    value: &Value,
    location: &str,
    out: &mut Vec<SchemaDefinition>,
) -> ImportResult<()> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };

    if let Some(alias) = local_id_alias(object, location) {
        out.push(SchemaDefinition {
            pointer: alias,
            schema: load_schema_at(value, location)?,
        });
    }

    for map_key in ["properties", "patternProperties"] {
        let child_location = format!("{location}/{}", escape_pointer_segment(map_key));
        if let Some(children) = object.get(map_key).and_then(Value::as_object) {
            for (name, schema_value) in children {
                let schema_location = format!(
                    "{child_location}/{}",
                    escape_pointer_segment(name)
                );
                out.push(SchemaDefinition {
                    pointer: schema_location.clone(),
                    schema: load_schema_at(schema_value, &schema_location)?,
                });
                collect_ref_targets(schema_value, &schema_location, out)?;
            }
        }
    }

    for defs_key in ["$defs", "definitions"] {
        let child_location = format!("{location}/{}", escape_pointer_segment(defs_key));
        if let Some(children) = object.get(defs_key).and_then(Value::as_object) {
            for (name, schema_value) in children {
                let schema_location = format!(
                    "{child_location}/{}",
                    escape_pointer_segment(name)
                );
                collect_ref_targets(schema_value, &schema_location, out)?;
            }
        }
    }

    for schema_key in [
        "additionalProperties",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
    ] {
        if let Some(child) = object.get(schema_key) {
            let child_location = format!("{location}/{}", escape_pointer_segment(schema_key));
            collect_ref_targets(child, &child_location, out)?;
        }
    }

    if let Some(items) = object.get("items") {
        let child_location = format!("{location}/items");
        match items {
            Value::Array(children) => {
                for (index, schema_value) in children.iter().enumerate() {
                    let schema_location = format!("{child_location}/{index}");
                    collect_ref_targets(schema_value, &schema_location, out)?;
                }
            }
            Value::Bool(_) | Value::Object(_) => collect_ref_targets(items, &child_location, out)?,
            _ => {}
        }
    }

    for array_key in ["prefixItems", "anyOf", "oneOf", "allOf"] {
        let child_location = format!("{location}/{}", escape_pointer_segment(array_key));
        if let Some(children) = object.get(array_key).and_then(Value::as_array) {
            for (index, schema_value) in children.iter().enumerate() {
                let schema_location = format!("{child_location}/{index}");
                collect_ref_targets(schema_value, &schema_location, out)?;
            }
        }
    }

    Ok(())
}

