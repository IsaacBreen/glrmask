use serde_json::{Map, Value};

use super::ast::{
    AdditionalProperties, ArraySchema, NumberSchema, ObjectSchema, PatternPropertySchema,
    PropertySchema, Schema, SchemaAssertions, SchemaDefinition, SchemaDocument, SchemaKind,
    SchemaType, StringSchema,
};
use super::error::{ImportResult, SchemaImportError};

fn singleton_all_of_ref_without_siblings(assertions: &SchemaAssertions) -> Option<&str> {
    if assertions.all_of.len() != 1 {
        return None;
    }

    let mut siblings = assertions.clone();
    siblings.all_of.clear();
    if !siblings.is_empty() {
        return None;
    }

    match &assertions.all_of[0].kind {
        SchemaKind::Ref(reference) => Some(reference.as_str()),
        _ => None,
    }
}

fn one_of_mixes_ref_and_inline_branches(branches: &[Schema]) -> bool {
    branches.len() > 1
        && branches
            .iter()
            .any(|branch| matches!(branch.kind, SchemaKind::Ref(_)))
        && branches
            .iter()
            .any(|branch| {
                !matches!(branch.kind, SchemaKind::Ref(_))
                    && !schema_is_null_only_inline_branch(branch)
            })
}

fn schema_is_null_only_inline_branch(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    matches!(assertions.types.as_deref(), Some([SchemaType::Null]))
        && assertions.const_value.is_none()
        && assertions.enum_values.is_none()
        && assertions.object.is_none()
        && assertions.array.is_none()
        && assertions.string.is_none()
        && assertions.number.is_none()
        && assertions.any_of.is_empty()
        && assertions.one_of.is_empty()
        && assertions.all_of.is_empty()
}

fn schema_is_object_shaped_inline_branch(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    assertions.object.is_some()
}

fn schema_is_array_shaped_inline_branch(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    assertions.array.is_some()
}

fn one_of_can_normalize_mixed_local_refs(branches: &[Schema]) -> bool {
    branches.len() > 1
        && branches.iter().any(|branch| matches!(branch.kind, SchemaKind::Ref(_)))
        && branches.iter().all(|branch| match &branch.kind {
            SchemaKind::Ref(reference) => reference.starts_with('#'),
            _ => {
                schema_is_null_only_inline_branch(branch)
                    || schema_is_object_shaped_inline_branch(branch)
                    || schema_is_array_shaped_inline_branch(branch)
            }
        })
}

fn schema_has_explicit_single_primitive_inline_type(schema: &Schema) -> bool {
    let SchemaKind::Assertions(assertions) = &schema.kind else {
        return false;
    };

    matches!(
        assertions.types.as_deref(),
        Some([SchemaType::String])
            | Some([SchemaType::Number])
            | Some([SchemaType::Integer])
            | Some([SchemaType::Boolean])
    ) && assertions.object.is_none()
        && assertions.array.is_none()
        && assertions.any_of.is_empty()
        && assertions.one_of.is_empty()
        && assertions.all_of.is_empty()
}

fn one_of_can_defer_local_ref_disjoint_family_proof(branches: &[Schema]) -> bool {
    branches.len() > 1
        && branches.iter().any(|branch| matches!(branch.kind, SchemaKind::Ref(_)))
        && branches.iter().all(|branch| match &branch.kind {
            SchemaKind::Ref(reference) => reference.starts_with('#'),
            _ => {
                schema_has_explicit_single_primitive_inline_type(branch)
                    || schema_is_null_only_inline_branch(branch)
                    || schema_is_object_shaped_inline_branch(branch)
                    || schema_is_array_shaped_inline_branch(branch)
            }
        })
}

fn normalize_mixed_ref_one_of_branches(branches: &mut [Schema]) {
    for branch in branches {
        if matches!(branch.kind, SchemaKind::Ref(_)) {
            let wrapped_ref = branch.clone();
            *branch = Schema::assertions(
                format!("{}/<mixed-ref-oneof>", branch.location),
                SchemaAssertions {
                    all_of: vec![wrapped_ref],
                    ..SchemaAssertions::default()
                },
            );
        }
    }
}

fn collect_all_ref_pointers(value: &Value, refs: &mut std::collections::BTreeSet<String>) {
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

fn local_id_alias(object: &Map<String, Value>, location: &str) -> Option<String> {
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

fn collect_definitions(
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

fn collect_ref_targets(
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

fn load_schema_at(value: &Value, location: &str) -> ImportResult<Schema> {
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
        // Normalize the narrow safe subset where local $ref branches are mixed
        // with object-shaped, array-shaped, or null-only inline branches.
        // Explicit primitive inline branches are only allowed through when
        // lowering can later prove they are disjoint from resolved local refs.
        if one_of_can_normalize_mixed_local_refs(&assertions.one_of) {
            normalize_mixed_ref_one_of_branches(&mut assertions.one_of);
        } else if one_of_can_defer_local_ref_disjoint_family_proof(&assertions.one_of) {
            // Lowering has the resolved local-ref target map needed to prove
            // this mixed local-ref disjoint-family subset safe.
        } else {
            return Err(SchemaImportError::at(
                location,
                "oneOf constraints with mixed $ref and inline branches are not supported",
            ));
        }
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

fn validate_supported_keys(object: &Map<String, Value>, location: &str) -> ImportResult<()> {
    let unsupported = object
        .keys()
        .filter(|key| is_unsupported_validation_key(key))
        .cloned()
        .collect::<Vec<_>>();
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(SchemaImportError::at(location, format!("Unimplemented keys: {unsupported:?}")))
    }
}

fn is_unsupported_validation_key(key: &str) -> bool {
    matches!(
        key,
        "propertyNames"
            | "uniqueItems"
            | "contains"
            | "minContains"
            | "maxContains"
            | "dependencies"
            | "dependentRequired"
            | "dependentSchemas"
            | "unevaluatedProperties"
            | "unevaluatedItems"
    )
}

fn load_types(object: &Map<String, Value>, location: &str) -> ImportResult<Option<Vec<SchemaType>>> {
    let Some(value) = object.get("type") else {
        return Ok(None);
    };

    let mut types = Vec::new();
    match value {
        Value::String(name) => types.push(parse_type_name(name, location)?),
        Value::Array(values) => {
            for (index, item) in values.iter().enumerate() {
                let Some(name) = item.as_str() else {
                    return Err(SchemaImportError::at(location, format!("type[{index}] must be a string")));
                };
                let schema_type = parse_type_name(name, location)?;
                if !types.contains(&schema_type) {
                    types.push(schema_type);
                }
            }
        }
        _ => return Err(SchemaImportError::at(location, "type must be a string or string array")),
    }
    Ok(Some(types))
}

fn parse_type_name(name: &str, location: &str) -> ImportResult<SchemaType> {
    match name {
        "null" => Ok(SchemaType::Null),
        "boolean" => Ok(SchemaType::Boolean),
        "object" => Ok(SchemaType::Object),
        "array" => Ok(SchemaType::Array),
        "string" => Ok(SchemaType::String),
        "number" => Ok(SchemaType::Number),
        "integer" => Ok(SchemaType::Integer),
        _ => Err(SchemaImportError::at(location, format!("unsupported JSON Schema type {name:?}"))),
    }
}

fn load_enum_values(object: &Map<String, Value>, location: &str) -> ImportResult<Option<Vec<Value>>> {
    let Some(value) = object.get("enum") else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(SchemaImportError::at(location, "enum must be an array"));
    };
    Ok(Some(values.clone()))
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

fn load_object_keywords(
    object: &Map<String, Value>,
    location: &str,
) -> ImportResult<ObjectSchema> {
    let mut schema = ObjectSchema::default();

    if let Some(properties) = object.get("properties") {
        let Some(properties) = properties.as_object() else {
            return Err(SchemaImportError::at(location, "properties must be an object"));
        };
        for (name, property_value) in properties {
            let child_location = format!("{location}/properties/{}", escape_pointer_segment(name));
            schema.properties.push(PropertySchema {
                name: name.clone(),
                schema: load_schema_at(property_value, &child_location)?,
            });
        }
    }

    if let Some(required) = object.get("required") {
        let Some(required) = required.as_array() else {
            return Err(SchemaImportError::at(location, "required must be an array"));
        };
        for (index, value) in required.iter().enumerate() {
            let Some(name) = value.as_str() else {
                return Err(SchemaImportError::at(location, format!("required[{index}] must be a string")));
            };
            schema.required.insert(name.to_string());
        }
    }

    if let Some(pattern_properties) = object.get("patternProperties") {
        let Some(pattern_properties) = pattern_properties.as_object() else {
            return Err(SchemaImportError::at(location, "patternProperties must be an object"));
        };
        for (pattern, property_value) in pattern_properties {
            let child_location = format!("{location}/patternProperties/{}", escape_pointer_segment(pattern));
            schema.pattern_properties.push(PatternPropertySchema {
                pattern: pattern.clone(),
                schema: load_schema_at(property_value, &child_location)?,
            });
        }
    }

    if let Some(additional) = object.get("additionalProperties") {
        schema.additional_properties = match additional {
            Value::Bool(true) => AdditionalProperties::AllowAny,
            Value::Bool(false) => AdditionalProperties::Deny,
            _ => AdditionalProperties::Schema(Box::new(load_schema_at(
                additional,
                &format!("{location}/additionalProperties"),
            )?)),
        };
    }

    schema.min_properties = read_usize_keyword(object, "minProperties", location)?.unwrap_or(0);
    schema.max_properties = read_usize_keyword(object, "maxProperties", location)?;

    Ok(schema)
}

fn load_array_keywords(
    object: &Map<String, Value>,
    location: &str,
) -> ImportResult<ArraySchema> {
    let mut schema = ArraySchema::default();
    let mut tuple_items_loaded = false;

    if let Some(items) = object.get("items") {
        match items {
            Value::Array(values) => {
                tuple_items_loaded = true;
                for (index, item) in values.iter().enumerate() {
                    schema.prefix_items.push(load_schema_at(item, &format!("{location}/items/{index}"))?);
                }
            }
            _ => {
                schema.items = Box::new(load_schema_at(items, &format!("{location}/items"))?);
            }
        }
    }

    if let Some(prefix_items) = object.get("prefixItems") {
        if tuple_items_loaded {
            return Err(SchemaImportError::at(
                location,
                "cannot use tuple-form items together with prefixItems",
            ));
        }
        let Some(prefix_items) = prefix_items.as_array() else {
            return Err(SchemaImportError::at(location, "prefixItems must be an array"));
        };
        if !tuple_items_loaded {
            schema.prefix_items.clear();
        }
        for (index, item) in prefix_items.iter().enumerate() {
            schema.prefix_items.push(load_schema_at(item, &format!("{location}/prefixItems/{index}"))?);
        }
    }

    if let Some(additional_items) = object.get("additionalItems") {
        if !schema.prefix_items.is_empty() {
            schema.items = Box::new(load_schema_at(
                additional_items,
                &format!("{location}/additionalItems"),
            )?);
        }
    }

    schema.min_items = read_usize_keyword(object, "minItems", location)?.unwrap_or(0);
    schema.max_items = read_usize_keyword(object, "maxItems", location)?;
    Ok(schema)
}

fn load_string_keywords(object: &Map<String, Value>, location: &str) -> ImportResult<StringSchema> {
    Ok(StringSchema {
        min_length: read_usize_keyword(object, "minLength", location)?.unwrap_or(0),
        max_length: read_usize_keyword(object, "maxLength", location)?,
        pattern: read_string_keyword(object, "pattern", location)?,
        format: read_string_keyword(object, "format", location)?,
    })
}

fn load_number_keywords(
    object: &Map<String, Value>,
    location: &str,
) -> ImportResult<NumberSchema> {
    let mut number = NumberSchema {
        integer: false,
        minimum: read_f64_keyword(object, "minimum", location)?,
        maximum: read_f64_keyword(object, "maximum", location)?,
        exclusive_minimum: false,
        exclusive_maximum: false,
        multiple_of: read_f64_keyword(object, "multipleOf", location)?,
    };

    if let Some(value) = object.get("exclusiveMinimum") {
        match value {
            Value::Bool(flag) => number.exclusive_minimum = *flag,
            Value::Number(_) => {
                number.minimum = read_f64_keyword(object, "exclusiveMinimum", location)?;
                number.exclusive_minimum = true;
            }
            _ => return Err(SchemaImportError::at(location, "exclusiveMinimum must be a bool or number")),
        }
    }

    if let Some(value) = object.get("exclusiveMaximum") {
        match value {
            Value::Bool(flag) => number.exclusive_maximum = *flag,
            Value::Number(_) => {
                number.maximum = read_f64_keyword(object, "exclusiveMaximum", location)?;
                number.exclusive_maximum = true;
            }
            _ => return Err(SchemaImportError::at(location, "exclusiveMaximum must be a bool or number")),
        }
    }

    if number.multiple_of.is_some_and(|value| value <= 0.0) {
        return Err(SchemaImportError::at(location, "multipleOf must be positive"));
    }

    Ok(number)
}

fn read_usize_keyword(object: &Map<String, Value>, key: &str, location: &str) -> ImportResult<Option<usize>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    let Some(number) = value.as_u64() else {
        return Err(SchemaImportError::at(location, format!("{key} must be a non-negative integer")));
    };
    usize::try_from(number)
        .map(Some)
        .map_err(|_| SchemaImportError::at(location, format!("{key} is too large")))
}

fn read_f64_keyword(object: &Map<String, Value>, key: &str, location: &str) -> ImportResult<Option<f64>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    value
        .as_f64()
        .ok_or_else(|| SchemaImportError::at(location, format!("{key} must be a number")))
        .map(Some)
}

fn read_string_keyword(object: &Map<String, Value>, key: &str, location: &str) -> ImportResult<Option<String>> {
    let Some(value) = object.get(key) else {
        return Ok(None);
    };
    value
        .as_str()
        .map(|value| Some(value.to_string()))
        .ok_or_else(|| SchemaImportError::at(location, format!("{key} must be a string")))
}

pub(crate) fn escape_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}
