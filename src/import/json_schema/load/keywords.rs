//! Raw JSON Schema keyword readers.
//!
//! This module is the only loader module that should know the exact spelling of
//! validation keywords such as `properties`, `minItems`, or `multipleOf`.  It
//! converts those keys into typed schema structs and rejects unsupported
//! validation keywords with source-location diagnostics.

use serde_json::{Map, Value};

use super::super::diagnostics::{ImportResult, SchemaImportError};
use super::super::schema::{
    AdditionalProperties, ArraySchema, NumberSchema, ObjectSchema, PatternPropertySchema,
    PropertySchema, SchemaType, StringSchema,
};
use super::pointers::escape_pointer_segment;
use super::typed::load_schema_at;

pub(super) fn validate_supported_keys(object: &Map<String, Value>, location: &str) -> ImportResult<()> {
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
    super::super::diagnostics::is_documented_unsupported_keyword(key)
}

pub(super) fn load_types(object: &Map<String, Value>, location: &str) -> ImportResult<Option<Vec<SchemaType>>> {
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

pub(super) fn load_enum_values(object: &Map<String, Value>, location: &str) -> ImportResult<Option<Vec<Value>>> {
    let Some(value) = object.get("enum") else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(SchemaImportError::at(location, "enum must be an array"));
    };
    Ok(Some(values.clone()))
}


pub(super) fn load_object_keywords(
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

pub(super) fn load_array_keywords(
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

pub(super) fn load_string_keywords(object: &Map<String, Value>, location: &str) -> ImportResult<StringSchema> {
    Ok(StringSchema {
        min_length: read_usize_keyword(object, "minLength", location)?.unwrap_or(0),
        max_length: read_usize_keyword(object, "maxLength", location)?,
        pattern: read_string_keyword(object, "pattern", location)?,
        format: read_string_keyword(object, "format", location)?,
    })
}

pub(super) fn load_number_keywords(
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

