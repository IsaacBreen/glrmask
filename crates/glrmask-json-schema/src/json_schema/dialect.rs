//! Dialect normalization at schema positions, before validation or typed loading.
//! An absent `$schema` selects Draft 2020-12. Unknown extension payloads and
//! descriptive annotations are opaque, never recursively interpreted as schemas.
use serde_json::{Map, Value};
use super::error::{ImportResult, SchemaImportError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dialect { Draft4, Draft6, Draft7, Draft2019, Draft2020 }
impl Dialect {
    fn modern(self) -> bool { matches!(self, Self::Draft2019 | Self::Draft2020) }
    fn conditional(self) -> bool { !matches!(self, Self::Draft4 | Self::Draft6) }
}
fn malformed(path: &str, message: &str) -> SchemaImportError {
    SchemaImportError::at(path, format!("malformed schema: {message}"))
}
fn unsupported(path: &str, message: &str) -> SchemaImportError {
    SchemaImportError::at(path, format!("unsupported feature: {message}"))
}
fn pointer(path: &str, key: &str) -> String {
    format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"))
}
fn declaration(value: &Value, path: &str) -> ImportResult<Dialect> {
    let uri = value.as_str().ok_or_else(|| malformed(path, "$schema must be a string"))?;
    match uri.trim_end_matches('#') {
        "http://json-schema.org/draft-04/schema" => Ok(Dialect::Draft4),
        "http://json-schema.org/draft-06/schema" => Ok(Dialect::Draft6),
        "http://json-schema.org/draft-07/schema" => Ok(Dialect::Draft7),
        "https://json-schema.org/draft/2019-09/schema" => Ok(Dialect::Draft2019),
        "https://json-schema.org/draft/2020-12/schema" => Ok(Dialect::Draft2020),
        _ => Err(unsupported(path, &format!("unknown JSON Schema dialect {uri:?}"))),
    }
}
/// Return a schema-only tree with draft-specific assertions normalized. This
/// must precede scans/preflight so annotation data cannot trigger assertions.
pub fn prepare_document(root: &Value) -> ImportResult<Value> {
    let dialect = root.get("$schema").map(|v| declaration(v, "#/$schema"))
        .transpose()?.unwrap_or(Dialect::Draft2020);
    let prepared = prepare(root, "#", dialect, true, false)?;
    check_reference_targets(&prepared, root, &prepared, "#")?;
    Ok(prepared)
}
// Referencing an annotation payload is legal in JSON Schema, but that payload
// has not been normalized as a schema. Fail closed instead of losing its rules.
fn check_reference_targets(value: &Value, original: &Value, prepared: &Value, path: &str) -> ImportResult<()> {
    if let Some(object) = value.as_object() {
        if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
            if let Some(pointer) = reference.strip_prefix('#') {
                if !pointer.is_empty() && original.pointer(pointer).is_some() && prepared.pointer(pointer).is_none() {
                    return Err(unsupported(&pointer_path(path), "reference targets an annotation or ignored keyword payload"));
                }
            }
        }
        for (key, child) in object {
            if !matches!(key.as_str(), "const" | "enum") {
                check_reference_targets(child, original, prepared, &pointer(path, key))?;
            }
        }
    } else if let Some(items) = value.as_array() {
        for (i, child) in items.iter().enumerate() {
            check_reference_targets(child, original, prepared, &pointer(path, &i.to_string()))?;
        }
    }
    Ok(())
}
fn pointer_path(path: &str) -> String { pointer(path, "$ref") }
fn prepare(value: &Value, path: &str, dialect: Dialect, root: bool, draft4_bool: bool) -> ImportResult<Value> {
    if value.is_boolean() {
        if dialect == Dialect::Draft4 && !draft4_bool {
            return Err(malformed(path, "Draft 4 schemas must be objects"));
        }
        return Ok(value.clone());
    }
    let object = value.as_object().ok_or_else(|| malformed(path, "schema must be an object or boolean"))?;
    if let Some(v) = object.get("$schema") {
        let declared = declaration(v, &pointer(path, "$schema"))?;
        if !root && declared != dialect {
            return Err(unsupported(&pointer(path, "$schema"), "mixed dialect resources"));
        }
    }
    // In Draft 4/6/7 a reference object ignores all assertion siblings.
    if !dialect.modern() && object.contains_key("$ref") {
        let reference = object["$ref"].as_str().ok_or_else(|| malformed(path, "$ref must be a string"))?;
        let mut result = Map::new();
        result.insert("$ref".into(), Value::String(reference.into()));
        // Definitions remain addressable even though reference siblings do not
        // contribute assertions in these drafts.
        if let Some(definitions) = object.get("definitions") {
            let entries = definitions.as_object().ok_or_else(|| malformed(path, "definitions must be an object"))?;
            let mut normalized = Map::new();
            for (name, schema) in entries {
                normalized.insert(name.clone(), prepare(schema, &pointer(&pointer(path, "definitions"), name), dialect, false, false)?);
            }
            result.insert("definitions".into(), Value::Object(normalized));
        }
        return Ok(Value::Object(result));
    }
    if dialect.modern() {
        if let Some(vocabulary) = object.get("$vocabulary") {
            let entries = vocabulary.as_object().ok_or_else(|| malformed(&pointer(path, "$vocabulary"), "must be an object"))?;
            for (uri, required) in entries {
                let required = required.as_bool().ok_or_else(|| malformed(&pointer(path, "$vocabulary"), "vocabulary requirements must be booleans"))?;
                if !required { continue; }
                let known = ["core", "applicator", "validation", "meta-data", "content", "unevaluated", "format-annotation"]
                    .iter().any(|name| uri == &format!("https://json-schema.org/draft/{}/vocab/{name}", if dialect == Dialect::Draft2020 { "2020-12" } else { "2019-09" }));
                if !known { return Err(unsupported(&pointer(&pointer(path, "$vocabulary"), uri), "required vocabulary (including format assertion) is not implemented")); }
            }
        }
    }
    validate_scalar_keywords(object, path, dialect)?;
    let mut out = Map::new();
    for (key, child) in object {
        let child_path = pointer(path, key);
        match key.as_str() {
            "$schema" | "$vocabulary" | "$comment" | "title" | "description" | "default" | "examples" | "readOnly" | "writeOnly" | "deprecated" | "contentEncoding" | "contentMediaType" | "contentSchema" => {},
            "format" => {
                if !child.is_string() { return Err(malformed(&child_path, "format must be a string")); }
                // Draft 2019-09 and 2020-12 standard dialects use annotations.
                // Earlier drafts permit optional format enforcement; this
                // implementation consistently chooses annotation-only there too.
            },
            "id" if dialect != Dialect::Draft4 => {},
            "$id" if dialect == Dialect::Draft4 => {},
            "$anchor" if !dialect.modern() => {},
            "$id" | "id" | "$anchor" | "$ref" => {
                if !child.is_string() { return Err(malformed(&child_path, "identifier/reference must be a string")); }
                if key != "$ref" && key != "$anchor" && !root && !child.as_str().unwrap().starts_with('#') {
                    return Err(unsupported(&child_path, "nested resource identifiers require URI scope resolution"));
                }
                if key == "$anchor" {
                    return Err(unsupported(&child_path, "named anchors"));
                }
                out.insert(key.clone(), child.clone());
            },
            "$dynamicRef" | "$dynamicAnchor" if dialect == Dialect::Draft2020 => return Err(unsupported(&child_path, "dynamic references")),
            "$recursiveRef" | "$recursiveAnchor" if dialect == Dialect::Draft2019 => return Err(unsupported(&child_path, "recursive references")),
            "properties" | "patternProperties" | "definitions" | "$defs" | "dependentSchemas" => {
                if (key == "dependentSchemas" || key == "$defs") && !dialect.modern() { continue; }
                let entries = child.as_object().ok_or_else(|| malformed(&child_path, "must be a map of schemas"))?;
                let mut children = Map::new();
                for (name, schema) in entries {
                    children.insert(name.clone(), prepare(schema, &pointer(&child_path, name), dialect, false, false)?);
                }
                out.insert(key.clone(), Value::Object(children));
            },
            "dependencies" if dialect.modern() => {},
            "dependencies" => {
                let entries = child.as_object().ok_or_else(|| malformed(&child_path, "must be an object"))?;
                let mut children = Map::new();
                for (name, schema) in entries {
                    children.insert(name.clone(), if schema.is_array() { schema.clone() } else { prepare(schema, &pointer(&child_path, name), dialect, false, false)? });
                }
                out.insert(key.clone(), Value::Object(children));
            },
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                if key == "prefixItems" && dialect != Dialect::Draft2020 { continue; }
                let entries = child.as_array().ok_or_else(|| malformed(&child_path, "must be an array of schemas"))?;
                if entries.is_empty() { return Err(malformed(&child_path, "schema array must not be empty")); }
                let children = entries.iter().enumerate().map(|(i, v)| prepare(v, &pointer(&child_path, &i.to_string()), dialect, false, false)).collect::<ImportResult<Vec<_>>>()?;
                out.insert(key.clone(), Value::Array(children));
            },
            "items" if child.is_array() => {
                if dialect == Dialect::Draft2020 { return Err(malformed(&child_path, "Draft 2020-12 items must be a schema; use prefixItems for tuples")); }
                let entries = child.as_array().unwrap();
                if entries.is_empty() { return Err(malformed(&child_path, "tuple items must not be empty")); }
                let children = entries.iter().enumerate().map(|(i, v)| prepare(v, &pointer(&child_path, &i.to_string()), dialect, false, false)).collect::<ImportResult<Vec<_>>>()?;
                // Retain the legacy tuple spelling so JSON pointers stay valid.
                out.insert(key.clone(), Value::Array(children));
            },
            "additionalItems" if dialect == Dialect::Draft2020 || !object.get("items").is_some_and(Value::is_array) => {},
            "if" | "then" | "else" if !dialect.conditional() => {},
            "contains" | "propertyNames" if dialect == Dialect::Draft4 => {},
            "unevaluatedProperties" | "unevaluatedItems" if !dialect.modern() => {},
            "items" | "additionalItems" | "additionalProperties" | "contains" | "propertyNames" | "not" | "if" | "then" | "else" | "unevaluatedProperties" | "unevaluatedItems" => {
                out.insert(key.clone(), prepare(child, &child_path, dialect, false, matches!(key.as_str(), "additionalItems" | "additionalProperties"))?);
            },
            "exclusiveMinimum" | "exclusiveMaximum" => {
                if dialect == Dialect::Draft4 {
                    let exclusive = child.as_bool().ok_or_else(|| malformed(&child_path, "Draft 4 exclusive bounds must be booleans"))?;
                    let bound_key = if key == "exclusiveMinimum" { "minimum" } else { "maximum" };
                    if !object.contains_key(bound_key) { return Err(malformed(&child_path, "Draft 4 exclusive bound requires corresponding bound")); }
                    if exclusive { out.insert(key.clone(), object[bound_key].clone()); }
                } else {
                    if !child.is_number() { return Err(malformed(&child_path, "exclusive bound must be numeric in this dialect")); }
                    out.insert(key.clone(), child.clone());
                }
            },
            "const" if dialect == Dialect::Draft4 => {},
            "dependentRequired" | "minContains" | "maxContains" if !dialect.modern() => {},
            "minContains" | "maxContains" | "minProperties" | "maxProperties" | "minItems" | "maxItems" | "minLength" | "maxLength" => {
                let count = if let Some(count) = child.as_u64() { count } else {
                    let count = child.as_f64().ok_or_else(|| malformed(&child_path, "count must be a nonnegative integer"))?;
                    if count < 0.0 || count.fract() != 0.0 { return Err(malformed(&child_path, "count must be a nonnegative integer")); }
                    if count >= u64::MAX as f64 { return Err(SchemaImportError::at(&child_path, "resource limit: count exceeds supported integer range")); }
                    count as u64
                };
                out.insert(key.clone(), Value::from(count));
            },
            "type" | "const" | "enum" | "required" | "dependentRequired" | "uniqueItems" | "pattern" | "minimum" | "maximum" | "multipleOf" => { out.insert(key.clone(), child.clone()); },
            // Unknown keywords are annotations, including object/array values.
            _ => {},
        }
    }
    Ok(Value::Object(out))
}

fn validate_scalar_keywords(object: &Map<String, Value>, path: &str, dialect: Dialect) -> ImportResult<()> {
    for (key, value) in object {
        let at = pointer(path, key);
        match key.as_str() {
            "type" => {
                let names: Vec<&Value> = if let Some(values) = value.as_array() {
                    if values.is_empty() { return Err(malformed(&at, "type array must not be empty")); }
                    values.iter().collect()
                } else { vec![value] };
                let mut seen = std::collections::BTreeSet::new();
                for name in names {
                    let name = name.as_str().ok_or_else(|| malformed(&at, "type must name JSON types"))?;
                    if !matches!(name, "null" | "boolean" | "object" | "array" | "number" | "integer" | "string") {
                        return Err(malformed(&at, "unknown JSON type"));
                    }
                    if !seen.insert(name) { return Err(malformed(&at, "type array must contain unique names")); }
                }
            },
            "enum" => {
                if !value.as_array().is_some_and(|values| !values.is_empty()) { return Err(malformed(&at, "enum must be a nonempty array")); }
            },
            "minProperties" | "maxProperties" | "minItems" | "maxItems" | "minLength" | "maxLength" => {
                if !value.as_u64().is_some() && !value.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0 && n.fract() == 0.0) {
                    return Err(malformed(&at, "count must be a nonnegative integer"));
                }
            },
            "minimum" | "maximum" | "multipleOf" => {
                if !value.is_number() { return Err(malformed(&at, "bound must be numeric")); }
                if key == "multipleOf" && value.as_f64().is_none_or(|v| v <= 0.0) { return Err(malformed(&at, "multipleOf must be positive")); }
            },
            "required" => {
                let values = value.as_array().ok_or_else(|| malformed(&at, "required must be an array"))?;
                if dialect == Dialect::Draft4 && values.is_empty() { return Err(malformed(&at, "Draft 4 required must be nonempty")); }
                let mut seen = std::collections::BTreeSet::new();
                for v in values {
                    let name = v.as_str().ok_or_else(|| malformed(&at, "required entries must be strings"))?;
                    if !seen.insert(name) { return Err(malformed(&at, "required entries must be unique")); }
                }
            },
            "uniqueItems" => if !value.is_boolean() { return Err(malformed(&at, "uniqueItems must be boolean")); },
            "pattern" => {
                let pattern = value.as_str().ok_or_else(|| malformed(&at, "pattern must be a string"))?;
                let normalized = super::string::preprocess_ascii_shorthand(pattern);
                regex::Regex::new(&normalized).map_err(|error| {
                    if error.to_string().contains("not supported") || error.to_string().contains("size limit") {
                        unsupported(&at, &format!("regex constraint: {error}"))
                    } else { malformed(&at, &format!("invalid regex: {error}")) }
                })?;
            },
            "dependentRequired" if dialect.modern() => validate_dependencies(value, &at, false, true)?,
            "dependencies" if !dialect.modern() => validate_dependencies(value, &at, dialect == Dialect::Draft4, false)?,
            "minContains" | "maxContains" if dialect.modern() => {
                if !value.as_f64().is_some_and(|v| v >= 0.0 && v.fract() == 0.0) {
                    return Err(malformed(&at, "count must be a nonnegative integer"));
                }
            },
            _ => {},
        }
    }
    Ok(())
}

fn validate_dependencies(value: &Value, path: &str, nonempty: bool, arrays_only: bool) -> ImportResult<()> {
    let entries = value.as_object().ok_or_else(|| malformed(path, "dependencies must be an object"))?;
    for (key, value) in entries {
        let at = pointer(path, key);
        if let Some(names) = value.as_array() {
            if nonempty && names.is_empty() { return Err(malformed(&at, "dependency array must be nonempty")); }
            let mut seen = std::collections::BTreeSet::new();
            for name in names {
                let name = name.as_str().ok_or_else(|| malformed(&at, "dependency entries must be strings"))?;
                if !seen.insert(name) { return Err(malformed(&at, "dependency entries must be unique")); }
            }
        } else if arrays_only { return Err(malformed(&at, "dependentRequired values must be arrays")); }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn malformed_constraints_do_not_disappear_when_inapplicable() {
        for schema in [json!({"type":"null","pattern":"["}), json!({"dependentRequired":{"a":["b","b"]}}), json!({"$schema":"http://json-schema.org/draft-07/schema#","dependencies":{"a":["b","b"]}}), json!({"minContains":"one"})] {
            assert!(prepare_document(&schema).unwrap_err().message().contains("malformed"));
        }
    }
    #[test]
    fn annotations_are_opaque_and_format_is_not_asserted() {
        assert_eq!(prepare_document(&json!({"format":"unknown", "description":{"oneOf":false}, "x-vendor":{"not":4}, "default":{"$ref":4}, "const":{"oneOf":4}})).unwrap(), json!({"const":{"oneOf":4}}));
    }
    #[test]
    fn malformed_nested_types_are_rejected_before_normalization() {
        for schema in [json!({"not":{"type":[]}}), json!({"not":{"type":["string", "string"]}}), json!({"type":"unknown"}), json!({"type":"null", "minimum":"bad"}), json!({"enum":[]})] {
            assert!(prepare_document(&schema).unwrap_err().message().contains("malformed"));
        }
    }
    #[test]
    fn dialect_tuple_and_bounds() {
        assert!(prepare_document(&json!({"items":[{}]})).unwrap_err().message().contains("malformed"));
        assert_eq!(prepare_document(&json!({"$schema":"http://json-schema.org/draft-04/schema#", "minimum":3, "exclusiveMinimum":true})).unwrap(), json!({"minimum":3,"exclusiveMinimum":3}));
        assert!(prepare_document(&json!({"exclusiveMinimum":true})).is_err());
        assert_eq!(prepare_document(&json!({"$schema":"http://json-schema.org/draft-07/schema#", "prefixItems":[false], "items":[{}]})).unwrap(), json!({"items":[{}]}));
    }
    #[test]
    fn references_follow_dialect_sibling_rules() {
        assert_eq!(prepare_document(&json!({"$schema":"http://json-schema.org/draft-07/schema#", "$ref":"#", "not":42})).unwrap(), json!({"$ref":"#"}));
        assert!(prepare_document(&json!({"$ref":"#", "not":42})).is_err());
    }
    #[test]
    fn vocabularies_and_mixed_dialects_fail_closed() {
        assert!(prepare_document(&json!({"$vocabulary":{"https://example.com/vocab":true}})).unwrap_err().message().contains("unsupported"));
        assert!(prepare_document(&json!({"$vocabulary":{"https://json-schema.org/draft/2020-12/vocab/format-assertion":true}})).is_err());
        assert_eq!(prepare_document(&json!({"$vocabulary":{"https://example.com/vocab":false}})).unwrap(), json!({}));
        assert!(prepare_document(&json!({"properties":{"a":{"$schema":"http://json-schema.org/draft-07/schema#"}}})).is_err());
    }
}
