//! JSON Schema Parser
//!
//! Converts serde_json::Value to SchemaType.
//!
//! # Assumptions and Limitations
//!
//! This parser implements a subset of JSON Schema focused on what can be
//! represented as a context-free grammar:
//!
//! ## Supported Features
//! - `type`: string, integer, number, boolean, null, object, array
//! - `type` as array: ["string", "null"]
//! - `properties`, `required`, `additionalProperties`
//! - `items`, `prefixItems`
//! - `$ref`, `$defs`, `definitions`
//! - `allOf`, `anyOf`, `oneOf`
//! - `const`, `enum`
//! - `pattern`, `minLength`, `maxLength` for strings
//!
//! ## Intentionally Unsupported (semantic, not syntactic)
//! - `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`
//! - `minItems`, `maxItems` (array length)
//! - `minProperties`, `maxProperties`  
//! - `uniqueItems`
//! - `if/then/else`
//! - `not` (negation)
//! - `dependencies`
//! - `format` (not enforced, only stored)
//!
//! ## Behavior Notes
//! - Unknown keywords are ignored
//! - Recursive references create cyclic grammars (handled at emission time)
//! - `additionalProperties: false` restricts to only declared properties
//! - `additionalProperties: true` or schema allows any additional properties
//! - Properties appear in declaration order (for disambiguation)

use super::json_schema_types::*;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

/// Context for schema parsing, holds definitions and ref resolution
pub struct SchemaParser {
    /// Root schema for resolving local refs
    root: Value,
    /// Cached definitions from $defs and definitions
    definitions: BTreeMap<String, Value>,
    /// Stack for cycle detection during parsing
    parsing_stack: Vec<String>,
    /// Already parsed refs (for caching)
    parsed_refs: BTreeMap<String, SchemaType>,
}

impl SchemaParser {
    /// Create a new parser with the given root schema
    pub fn new(root: Value) -> Self {
        let mut parser = Self {
            root: root.clone(),
            definitions: BTreeMap::new(),
            parsing_stack: Vec::new(),
            parsed_refs: BTreeMap::new(),
        };
        parser.collect_definitions(&root);
        parser
    }
    
    /// Collect all definitions from the root schema
    fn collect_definitions(&mut self, root: &Value) {
        // Check $defs (JSON Schema draft 2019-09+)
        if let Some(defs) = root.get("$defs").and_then(|v| v.as_object()) {
            for (name, schema) in defs {
                let path = format!("#/$defs/{}", name);
                self.definitions.insert(path, schema.clone());
            }
        }
        // Check definitions (JSON Schema draft-07)
        if let Some(defs) = root.get("definitions").and_then(|v| v.as_object()) {
            for (name, schema) in defs {
                let path = format!("#/definitions/{}", name);
                self.definitions.insert(path, schema.clone());
            }
        }
    }
    
    /// Parse the root schema
    pub fn parse_root(&mut self) -> Result<SchemaType, String> {
        self.parse_schema(&self.root.clone())
    }
    
    /// Parse a schema value into SchemaType
    pub fn parse_schema(&mut self, schema: &Value) -> Result<SchemaType, String> {
        // Handle boolean schemas
        if let Some(b) = schema.as_bool() {
            return Ok(if b { SchemaType::Any } else { SchemaType::Never });
        }
        
        let obj = schema.as_object()
            .ok_or_else(|| "Schema must be boolean or object".to_string())?;
        
        // Handle $ref
        if let Some(ref_str) = obj.get("$ref").and_then(|v| v.as_str()) {
            return self.resolve_ref(ref_str);
        }
        
        // Handle const
        if let Some(const_val) = obj.get("const") {
            return Ok(SchemaType::Const(const_val.clone()));
        }
        
        // Handle enum
        if let Some(enum_vals) = obj.get("enum").and_then(|v| v.as_array()) {
            return Ok(SchemaType::Enum(enum_vals.clone()));
        }
        
        // Handle allOf
        if let Some(all_of) = obj.get("allOf").and_then(|v| v.as_array()) {
            let schemas: Result<Vec<SchemaType>, String> = all_of.iter()
                .map(|s| self.parse_schema(s))
                .collect();
            return Ok(SchemaType::AllOf(schemas?));
        }
        
        // Handle anyOf
        if let Some(any_of) = obj.get("anyOf").and_then(|v| v.as_array()) {
            let schemas: Result<Vec<SchemaType>, String> = any_of.iter()
                .map(|s| self.parse_schema(s))
                .collect();
            return Ok(SchemaType::AnyOf(schemas?));
        }
        
        // Handle oneOf  
        if let Some(one_of) = obj.get("oneOf").and_then(|v| v.as_array()) {
            let schemas: Result<Vec<SchemaType>, String> = one_of.iter()
                .map(|s| self.parse_schema(s))
                .collect();
            return Ok(SchemaType::OneOf(schemas?));
        }
        
        // Handle type
        if let Some(type_val) = obj.get("type") {
            return self.parse_typed_schema(type_val, obj);
        }
        
        // Check if it's an implicit object (has properties but no type)
        if obj.contains_key("properties") || obj.contains_key("additionalProperties") {
            return self.parse_object_schema(obj);
        }
        
        // Check if it's an implicit array (has items but no type)
        if obj.contains_key("items") || obj.contains_key("prefixItems") {
            return self.parse_array_schema(obj);
        }
        
        // No constraints - matches any JSON value
        Ok(SchemaType::Any)
    }
    
    /// Resolve a $ref to a SchemaType
    fn resolve_ref(&mut self, ref_str: &str) -> Result<SchemaType, String> {
        // Cycle detection
        if self.parsing_stack.contains(&ref_str.to_string()) {
            // Return a Ref that will be resolved later during grammar generation
            return Ok(SchemaType::Ref(ref_str.to_string()));
        }
        
        // Check cache
        if let Some(cached) = self.parsed_refs.get(ref_str) {
            return Ok(cached.clone());
        }
        
        // Find the referenced schema
        let target_schema = if let Some(schema) = self.definitions.get(ref_str) {
            schema.clone()
        } else if ref_str.starts_with("#/") {
            // Navigate the root schema
            let parts: Vec<&str> = ref_str[2..].split('/').collect();
            let mut target = self.root.clone();
            for part in parts {
                target = target.get(part)
                    .ok_or_else(|| format!("Could not resolve ref: {}", ref_str))?
                    .clone();
            }
            target
        } else {
            return Err(format!("Unsupported ref format: {}", ref_str));
        };
        
        // Parse the referenced schema
        self.parsing_stack.push(ref_str.to_string());
        let result = self.parse_schema(&target_schema);
        self.parsing_stack.pop();
        
        // Cache the result
        if let Ok(ref schema_type) = result {
            self.parsed_refs.insert(ref_str.to_string(), schema_type.clone());
        }
        
        result
    }
    
    /// Parse a schema with a type field
    fn parse_typed_schema(&mut self, type_val: &Value, obj: &serde_json::Map<String, Value>) -> Result<SchemaType, String> {
        // Handle array of types
        if let Some(types) = type_val.as_array() {
            let schemas: Result<Vec<SchemaType>, String> = types.iter()
                .map(|t| {
                    if let Some(type_str) = t.as_str() {
                        self.parse_single_type(type_str, obj)
                    } else {
                        Err("Type array must contain strings".to_string())
                    }
                })
                .collect();
            return Ok(SchemaType::MultiType(schemas?));
        }
        
        // Handle single type
        if let Some(type_str) = type_val.as_str() {
            return self.parse_single_type(type_str, obj);
        }
        
        Err("Invalid type value".to_string())
    }
    
    /// Parse a single type string
    fn parse_single_type(&mut self, type_str: &str, obj: &serde_json::Map<String, Value>) -> Result<SchemaType, String> {
        match type_str {
            "string" => {
                let constraints = StringConstraints {
                    pattern: obj.get("pattern").and_then(|v| v.as_str()).map(String::from),
                    min_length: obj.get("minLength").and_then(|v| v.as_u64()),
                    max_length: obj.get("maxLength").and_then(|v| v.as_u64()),
                    format: obj.get("format").and_then(|v| v.as_str()).map(String::from),
                };
                Ok(SchemaType::String(constraints))
            }
            "integer" => Ok(SchemaType::Primitive(PrimitiveType::Integer)),
            "number" => Ok(SchemaType::Primitive(PrimitiveType::Number)),
            "boolean" => Ok(SchemaType::Primitive(PrimitiveType::Boolean)),
            "null" => Ok(SchemaType::Primitive(PrimitiveType::Null)),
            "object" => self.parse_object_schema(obj),
            "array" => self.parse_array_schema(obj),
            _ => Ok(SchemaType::Any), // Unknown type - be permissive
        }
    }
    
    /// Parse an object schema
    fn parse_object_schema(&mut self, obj: &serde_json::Map<String, Value>) -> Result<SchemaType, String> {
        let mut schema = ObjectSchema::default();
        
        // Get required properties
        let required: HashSet<String> = obj.get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect())
            .unwrap_or_default();
        
        // Parse properties
        if let Some(props) = obj.get("properties").and_then(|v| v.as_object()) {
            for (name, prop_schema) in props {
                let prop_type = self.parse_schema(prop_schema)?;
                let is_required = required.contains(name);
                schema.properties.push((name.clone(), prop_type, is_required));
            }
        }
        
        // Parse additionalProperties
        match obj.get("additionalProperties") {
            Some(Value::Bool(false)) => schema.additional_properties = None,
            Some(Value::Bool(true)) => schema.additional_properties = Some(Box::new(SchemaType::Any)),
            Some(additional_schema) => {
                let additional_type = self.parse_schema(additional_schema)?;
                schema.additional_properties = Some(Box::new(additional_type));
            }
            None => schema.additional_properties = Some(Box::new(SchemaType::Any)), // Default: allow any
        }
        
        // Parse patternProperties
        if let Some(pattern_props) = obj.get("patternProperties").and_then(|v| v.as_object()) {
            for (pattern, pattern_schema) in pattern_props {
                let pattern_type = self.parse_schema(pattern_schema)?;
                schema.pattern_properties.push((pattern.clone(), pattern_type));
            }
        }
        
        Ok(SchemaType::Object(schema))
    }
    
    /// Parse an array schema
    fn parse_array_schema(&mut self, obj: &serde_json::Map<String, Value>) -> Result<SchemaType, String> {
        let mut schema = ArraySchema::default();
        
        // Parse prefixItems (tuple-style, draft 2020-12)
        if let Some(prefix) = obj.get("prefixItems").and_then(|v| v.as_array()) {
            for item_schema in prefix {
                schema.prefix_items.push(self.parse_schema(item_schema)?);
            }
        }
        
        // Parse items
        if let Some(items) = obj.get("items") {
            // In draft 2020-12, if prefixItems exists, items applies to additional items
            // In draft-07, items can be a single schema or array of schemas
            if items.is_array() && schema.prefix_items.is_empty() {
                // Draft-07 tuple style (items as array)
                for item_schema in items.as_array().unwrap() {
                    schema.prefix_items.push(self.parse_schema(item_schema)?);
                }
            } else if items.is_object() || items.is_boolean() {
                // Single schema for all items (or additional items in 2020-12)
                if schema.prefix_items.is_empty() {
                    schema.items = Some(Box::new(self.parse_schema(items)?));
                } else {
                    schema.additional_items = Some(Box::new(self.parse_schema(items)?));
                }
            }
        }
        
        // Parse additionalItems (draft-07)
        if let Some(additional) = obj.get("additionalItems") {
            match additional {
                Value::Bool(false) => schema.additional_items = None,
                Value::Bool(true) => schema.additional_items = Some(Box::new(SchemaType::Any)),
                _ => schema.additional_items = Some(Box::new(self.parse_schema(additional)?)),
            }
        }
        
        // Parse length constraints (stored but not enforced in grammar)
        schema.min_items = obj.get("minItems").and_then(|v| v.as_u64());
        schema.max_items = obj.get("maxItems").and_then(|v| v.as_u64());
        
        Ok(SchemaType::Array(schema))
    }
}

/// Convenience function to parse a JSON Schema string
pub fn parse_json_schema(json_str: &str) -> Result<SchemaType, String> {
    let value: Value = serde_json::from_str(json_str)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;
    let mut parser = SchemaParser::new(value);
    parser.parse_root()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_primitive_types() {
        let schema = parse_json_schema(r#"{"type": "string"}"#).unwrap();
        assert!(matches!(schema, SchemaType::String(_)));
        
        let schema = parse_json_schema(r#"{"type": "integer"}"#).unwrap();
        assert!(matches!(schema, SchemaType::Primitive(PrimitiveType::Integer)));
        
        let schema = parse_json_schema(r#"{"type": "boolean"}"#).unwrap();
        assert!(matches!(schema, SchemaType::Primitive(PrimitiveType::Boolean)));
    }
    
    #[test]
    fn test_parse_object() {
        let schema = parse_json_schema(r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }"#).unwrap();
        
        if let SchemaType::Object(obj) = schema {
            assert_eq!(obj.properties.len(), 2);
            assert!(obj.properties[0].2); // name is required
            assert!(!obj.properties[1].2); // age is not required
        } else {
            panic!("Expected Object schema");
        }
    }
    
    #[test]
    fn test_parse_array() {
        let schema = parse_json_schema(r#"{
            "type": "array",
            "items": {"type": "string"}
        }"#).unwrap();
        
        if let SchemaType::Array(arr) = schema {
            assert!(arr.items.is_some());
        } else {
            panic!("Expected Array schema");
        }
    }
    
    #[test]
    fn test_parse_enum() {
        let schema = parse_json_schema(r#"{"enum": ["red", "green", "blue"]}"#).unwrap();
        if let SchemaType::Enum(values) = schema {
            assert_eq!(values.len(), 3);
        } else {
            panic!("Expected Enum schema");
        }
    }
    
    #[test]
    fn test_parse_ref() {
        let schema_str = r##"{"$defs": {"name": {"type": "string"}}, "type": "object", "properties": {"firstName": {"$ref": "#/$defs/name"}, "lastName": {"$ref": "#/$defs/name"}}}"##;
        let schema = parse_json_schema(schema_str).unwrap();
        
        if let SchemaType::Object(obj) = schema {
            assert_eq!(obj.properties.len(), 2);
            // Both should resolve to String
            assert!(matches!(obj.properties[0].1, SchemaType::String(_)));
            assert!(matches!(obj.properties[1].1, SchemaType::String(_)));
        } else {
            panic!("Expected Object schema");
        }
    }
    
    #[test]
    fn test_parse_any_of() {
        let schema = parse_json_schema(r#"{
            "anyOf": [
                {"type": "string"},
                {"type": "integer"}
            ]
        }"#).unwrap();
        
        if let SchemaType::AnyOf(schemas) = schema {
            assert_eq!(schemas.len(), 2);
        } else {
            panic!("Expected AnyOf schema");
        }
    }
    
    #[test]
    fn test_parse_multi_type() {
        let schema = parse_json_schema(r#"{"type": ["string", "null"]}"#).unwrap();
        if let SchemaType::MultiType(types) = schema {
            assert_eq!(types.len(), 2);
        } else {
            panic!("Expected MultiType schema");
        }
    }
}
