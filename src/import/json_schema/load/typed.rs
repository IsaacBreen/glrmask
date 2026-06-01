//! Typed schema loading entry point.
//!
//! This file orchestrates the loader.  It delegates raw-keyword parsing to
//! `keywords`, reference-target collection to `collect`, pointer spelling to
//! `pointers`, and schema-shape predicates to `shape`.

use serde_json::{Map, Value};

use super::collect::{collect_definitions, collect_ref_targets};
use super::keywords::{
    load_array_keywords, load_enum_values, load_number_keywords, load_object_keywords,
    load_string_keywords, load_types, validate_supported_keys,
};
use super::pointers::collect_all_ref_pointers;
use super::shape::{one_of_mixes_ref_and_inline_branches, singleton_all_of_ref_without_siblings};
use super::super::diagnostics::{ImportResult, SchemaImportError};
use super::super::schema::{
    Schema, SchemaAssertions, SchemaDefinition, SchemaDocument, SchemaKind, SchemaType,
};

pub(crate) fn load_document(root: &Value) -> ImportResult<SchemaDocument> {
    let mut definitions = Vec::new();
    collect_definitions(root, "#", &mut definitions)?;
    let mut ref_targets = Vec::new();
    collect_ref_targets(root, "#", &mut ref_targets)?;

    let mut ref_pointers = std::collections::BTreeSet::new();
    collect_all_ref_pointers(root, &mut ref_pointers);

    for r in ref_pointers {
        if r == "#" {
            continue;
        }
        if r.starts_with("#/") {
            let pointer_path = &r[1..];
            let exists = definitions.iter().any(|d| d.pointer == r)
                || ref_targets.iter().any(|d| d.pointer == r);
            if !exists {
                if let Some(resolved_val) = root.pointer(pointer_path) {
                    let schema = load_schema_at(resolved_val, &r)?;
                    ref_targets.push(SchemaDefinition {
                        pointer: r.clone(),
                        schema,
                    });
                }
            }
        }
    }

    Ok(SchemaDocument {
        root: load_schema_at(root, "#")?,
        definitions,
        ref_targets,
    })
}


pub(super) fn load_schema_at(value: &Value, location: &str) -> ImportResult<Schema> {
    match value {
        Value::Bool(true) => Ok(Schema::any(location)),
        Value::Bool(false) => Ok(Schema::never(location)),
        Value::Object(object) => load_object_schema(object, location),
        _ => Err(SchemaImportError::at(location, "schema must be a boolean or object")),
    }
}

fn load_object_schema(object: &Map<String, Value>, location: &str) -> ImportResult<Schema> {
    validate_supported_keys(object, location)?;

    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        let siblings = load_assertions(object, location)?;
        if siblings.is_empty() {
            return Ok(Schema { location: location.to_string(), kind: SchemaKind::Ref(reference.to_string()) });
        }
        return Ok(Schema::assertions(
            location,
            SchemaAssertions {
                all_of: vec![
                    Schema { location: location.to_string(), kind: SchemaKind::Ref(reference.to_string()) },
                    Schema::assertions(format!("{location}/<ref-siblings>"), siblings),
                ],
                ..SchemaAssertions::default()
            },
        ));
    }

    let assertions = load_assertions(object, location)?;
    if let Some(reference) = singleton_all_of_ref_without_siblings(&assertions) {
        return Ok(Schema { location: location.to_string(), kind: SchemaKind::Ref(reference.to_string()) });
    }

    Ok(Schema::assertions(location, assertions))
}

fn load_assertions(object: &Map<String, Value>, location: &str) -> ImportResult<SchemaAssertions> {
    let mut assertions = SchemaAssertions::default();
    assertions.types = load_types(object, location)?;
    assertions.const_value = object.get("const").cloned();
    assertions.enum_values = load_enum_values(object, location)?;
    assertions.any_of = load_schema_array(object, "anyOf", location)?;
    assertions.one_of = load_schema_array(object, "oneOf", location)?;
    if one_of_mixes_ref_and_inline_branches(&assertions.one_of) {
        return Err(SchemaImportError::at(
            location,
            "oneOf constraints with mixed $ref and inline branches are not supported",
        ));
    }
    assertions.all_of = load_schema_array(object, "allOf", location)?;
    assertions.not = load_schema_member(object, "not", location)?;

    if should_load_object_assertion(object, assertions.types.as_deref()) {
        assertions.object = Some(load_object_keywords(object, location)?);
    }
    if should_load_array_assertion(object, assertions.types.as_deref()) {
        assertions.array = Some(load_array_keywords(object, location)?);
    }
    if should_load_string_assertion(object, assertions.types.as_deref()) {
        assertions.string = Some(load_string_keywords(object, location)?);
    }
    if should_load_number_assertion(object, assertions.types.as_deref()) {
        assertions.number = Some(load_number_keywords(object, location)?);
    }

    Ok(assertions)
}


fn load_schema_array(
    object: &Map<String, Value>,
    key: &str,
    location: &str,
) -> ImportResult<Vec<Schema>> {
    let Some(value) = object.get(key) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        return Err(SchemaImportError::at(location, format!("{key} must be an array")));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, child)| load_schema_at(child, &format!("{location}/{key}/{index}")))
        .collect()
}

fn load_schema_member(
    object: &Map<String, Value>,
    key: &str,
    location: &str,
) -> ImportResult<Option<Schema>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    load_schema_at(value, &format!("{location}/{key}")).map(Some)
}

fn should_load_object_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {
    type_mentions(types, SchemaType::Object)
        || [
            "properties",
            "required",
            "patternProperties",
            "additionalProperties",
            "minProperties",
        ]
            .iter()
            .any(|key| object.contains_key(*key))
}

fn should_load_array_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {
    type_mentions(types, SchemaType::Array)
        || ["items", "prefixItems", "minItems", "maxItems"]
            .iter()
            .any(|key| object.contains_key(*key))
}

fn should_load_string_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {
    type_mentions(types, SchemaType::String)
        || ["minLength", "maxLength", "pattern", "format"]
            .iter()
            .any(|key| object.contains_key(*key))
}

fn should_load_number_assertion(object: &Map<String, Value>, types: Option<&[SchemaType]>) -> bool {
    type_mentions(types, SchemaType::Number)
        || type_mentions(types, SchemaType::Integer)
        || ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"]
            .iter()
            .any(|key| object.contains_key(*key))
}

fn type_mentions(types: Option<&[SchemaType]>, wanted: SchemaType) -> bool {
    types.is_some_and(|types| types.contains(&wanted))
}

