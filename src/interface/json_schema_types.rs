//! JSON Schema Intermediate Representations
//!
//! This module defines two intermediate representations for JSON Schema conversion:
//!
//! 1. `SchemaType` - A Rust enum that faithfully represents a parsed JSON Schema
//!    with all its constraints and features. This is schema-centric.
//!
//! 2. `GrammarType` - An intermediate grammar representation that's closer to
//!    the final grammar but still retains semantic meaning (e.g., "this is an object",
//!    "this property is required"). This is grammar-centric but not yet the final form.
//!
//! # Conversion Pipeline
//!
//! ```text
//! JSON Schema (serde_json::Value)
//!       |
//!       v
//!   SchemaType  (parse_schema)
//!       |
//!       v
//!   GrammarType (schema_to_grammar)
//!       |
//!       v
//!   GrammarExpr (emit_grammar)
//! ```
//!
//! # Design Philosophy
//!
//! - Each intermediate form is self-contained and can be inspected/debugged independently
//! - Transformations between forms are explicit and testable
//! - Complex schema features are normalized early in the pipeline
//! - The final emission phase only deals with simple grammar constructs

use serde_json::Value;
use std::collections::BTreeMap;

// ============================================================================
// SchemaType: Schema-centric representation
// ============================================================================

/// A parsed JSON Schema element.
///
/// This enum represents the logical structure of a JSON Schema after parsing
/// but before any grammar-specific transformations.
#[derive(Debug, Clone, PartialEq)]
pub enum SchemaType {
    /// Matches any valid JSON value
    Any,
    
    /// Never matches (schema: false)
    Never,
    
    /// A primitive JSON type
    Primitive(PrimitiveType),
    
    /// A string with optional constraints
    String(StringConstraints),
    
    /// An object with property definitions
    Object(ObjectSchema),
    
    /// An array with item definitions
    Array(ArraySchema),
    
    /// A constant value
    Const(Value),
    
    /// One of several allowed values
    Enum(Vec<Value>),
    
    /// Reference to another schema definition
    Ref(String),
    
    /// All schemas must match (intersection)
    AllOf(Vec<SchemaType>),
    
    /// Any one schema must match (union)
    AnyOf(Vec<SchemaType>),
    
    /// Exactly one schema must match (exclusive union)
    /// Note: For grammar purposes, treated same as AnyOf
    OneOf(Vec<SchemaType>),
    
    /// Multi-type (e.g., ["string", "null"])
    MultiType(Vec<SchemaType>),
}

/// Primitive JSON types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    Integer,
    Number,
    Boolean,
    Null,
}

/// Constraints on a JSON string
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StringConstraints {
    /// Regex pattern the string must match
    pub pattern: Option<String>,
    /// Minimum string length
    pub min_length: Option<u64>,
    /// Maximum string length
    pub max_length: Option<u64>,
    /// Format hint (e.g., "email", "uri") - not enforced syntactically
    pub format: Option<String>,
}

impl StringConstraints {
    pub fn is_empty(&self) -> bool {
        self.pattern.is_none() && self.min_length.is_none() && 
        self.max_length.is_none() && self.format.is_none()
    }
}

/// JSON object schema
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObjectSchema {
    /// Named properties with their schemas
    pub properties: Vec<(String, SchemaType, bool)>, // (name, schema, required)
    /// Schema for additional properties (None = not allowed, Some(Any) = any allowed)
    pub additional_properties: Option<Box<SchemaType>>,
    /// Pattern-based property schemas
    pub pattern_properties: Vec<(String, SchemaType)>, // (pattern, schema)
}

/// JSON array schema  
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArraySchema {
    /// Schema for all items (homogeneous array)
    pub items: Option<Box<SchemaType>>,
    /// Schemas for prefix items (tuple-style)
    pub prefix_items: Vec<SchemaType>,
    /// Whether additional items beyond prefix are allowed
    pub additional_items: Option<Box<SchemaType>>,
    /// Minimum number of items
    pub min_items: Option<u64>,
    /// Maximum number of items  
    pub max_items: Option<u64>,
}

// ============================================================================
// GrammarType: Grammar-centric representation
// ============================================================================

/// Grammar-centric intermediate representation.
///
/// This is closer to what the final grammar will look like, but still retains
/// semantic information about the structure (e.g., "this is an object property").
/// Complex JSON Schema features like additionalProperties are resolved at this level.
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarType {
    /// Reference to a primitive JSON type rule
    PrimitiveRef(GrammarPrimitive),
    
    /// A literal string/bytes
    Literal(Vec<u8>),
    
    /// A sequence of grammar elements
    Sequence(Vec<GrammarType>),
    
    /// A choice between alternatives  
    Choice(Vec<GrammarType>),
    
    /// An optional element
    Optional(Box<GrammarType>),
    
    /// Zero or more repetitions
    Repeat(Box<GrammarType>),
    
    /// One or more repetitions
    RepeatOnePlus(Box<GrammarType>),
    
    /// A character class pattern
    CharClass(String),
    
    /// Reference to a named rule
    RuleRef(String),
    
    /// Defines a named rule for later reference
    /// (rule_name, body)
    RuleDefinition(String, Box<GrammarType>),
    
    /// A JSON object structure
    /// Contains the sequence of property patterns
    JsonObject {
        /// Opening brace
        open: Box<GrammarType>,
        /// The properties/content pattern
        content: Box<GrammarType>,
        /// Closing brace
        close: Box<GrammarType>,
    },
    
    /// A JSON array structure
    JsonArray {
        /// Opening bracket
        open: Box<GrammarType>,
        /// The items/content pattern
        content: Box<GrammarType>,
        /// Closing bracket
        close: Box<GrammarType>,
    },
    
    /// A JSON key-value pair
    JsonKeyValue {
        key: Box<GrammarType>,
        colon: Box<GrammarType>,
        value: Box<GrammarType>,
    },
    
    /// An empty/epsilon production
    Empty,
}

/// Primitive types that have predefined grammar rules
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarPrimitive {
    JsonString,
    JsonInteger,
    JsonNumber,
    JsonBoolean,
    JsonNull,
    JsonValue,
    JsonObject,
    JsonArray,
    StringChars,
    StringChar,
    EscapeSeq,
}

impl GrammarPrimitive {
    pub fn rule_name(&self) -> &'static str {
        match self {
            GrammarPrimitive::JsonString => "JSON_STRING",
            GrammarPrimitive::JsonInteger => "JSON_INTEGER",
            GrammarPrimitive::JsonNumber => "JSON_NUMBER",
            GrammarPrimitive::JsonBoolean => "JSON_BOOL",
            GrammarPrimitive::JsonNull => "JSON_NULL",
            GrammarPrimitive::JsonValue => "_json_value",
            GrammarPrimitive::JsonObject => "_json_object",
            GrammarPrimitive::JsonArray => "_json_array",
            GrammarPrimitive::StringChars => "STRING_CHARS",
            GrammarPrimitive::StringChar => "STRING_CHAR",
            GrammarPrimitive::EscapeSeq => "ESCAPE_SEQ",
        }
    }
}

// ============================================================================
// Helper constructors
// ============================================================================

impl GrammarType {
    /// Create a literal from a string
    pub fn lit(s: &str) -> Self {
        GrammarType::Literal(s.as_bytes().to_vec())
    }
    
    /// Create a reference to a primitive
    pub fn primitive(p: GrammarPrimitive) -> Self {
        GrammarType::PrimitiveRef(p)
    }
    
    /// Create a sequence, flattening nested sequences
    pub fn seq(items: Vec<GrammarType>) -> Self {
        let flattened: Vec<GrammarType> = items.into_iter()
            .flat_map(|item| match item {
                GrammarType::Sequence(inner) => inner,
                GrammarType::Empty => vec![],
                other => vec![other],
            })
            .collect();
        
        match flattened.len() {
            0 => GrammarType::Empty,
            1 => flattened.into_iter().next().unwrap(),
            _ => GrammarType::Sequence(flattened),
        }
    }
    
    /// Create a choice, flattening nested choices and deduplicating
    pub fn choice(items: Vec<GrammarType>) -> Self {
        let flattened: Vec<GrammarType> = items.into_iter()
            .flat_map(|item| match item {
                GrammarType::Choice(inner) => inner,
                other => vec![other],
            })
            .collect();
        
        match flattened.len() {
            0 => GrammarType::Empty,
            1 => flattened.into_iter().next().unwrap(),
            _ => GrammarType::Choice(flattened),
        }
    }
    
    /// Create an optional element
    pub fn opt(inner: GrammarType) -> Self {
        match inner {
            GrammarType::Empty => GrammarType::Empty,
            GrammarType::Optional(x) => GrammarType::Optional(x), // Already optional
            other => GrammarType::Optional(Box::new(other)),
        }
    }
    
    /// Create a repeat (zero or more)
    pub fn repeat(inner: GrammarType) -> Self {
        match inner {
            GrammarType::Empty => GrammarType::Empty,
            GrammarType::Repeat(x) => GrammarType::Repeat(x), // Already repeating
            other => GrammarType::Repeat(Box::new(other)),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_schema_type_construction() {
        let schema = SchemaType::Object(ObjectSchema {
            properties: vec![
                ("name".to_string(), SchemaType::String(StringConstraints::default()), true),
                ("age".to_string(), SchemaType::Primitive(PrimitiveType::Integer), false),
            ],
            additional_properties: None,
            pattern_properties: vec![],
        });
        
        assert!(matches!(schema, SchemaType::Object(_)));
    }
    
    #[test]
    fn test_grammar_type_flattening() {
        let seq = GrammarType::seq(vec![
            GrammarType::lit("{"),
            GrammarType::seq(vec![
                GrammarType::lit("a"),
                GrammarType::lit("b"),
            ]),
            GrammarType::lit("}"),
        ]);
        
        // Should flatten to single sequence of 4 elements
        if let GrammarType::Sequence(items) = seq {
            assert_eq!(items.len(), 4);
        } else {
            panic!("Expected Sequence");
        }
    }
}
