use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::Value;

/// A loaded JSON Schema document.
///
/// The importer deliberately resolves only local references.  The root schema is
/// stored separately from the map of named local definitions so the lowering
/// phase can turn recursive references into grammar rules without touching
/// serde_json values again.
#[derive(Debug, Clone)]
pub struct SchemaDocument {
    pub root: Schema,
    pub definitions: Vec<SchemaDefinition>,
    pub ref_targets: Vec<SchemaDefinition>,
}

#[derive(Debug, Clone)]
pub struct SchemaDefinition {
    pub pointer: String,
    pub schema: Schema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Schema {
    pub location: String,
    pub kind: SchemaKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum SchemaKind {
    /// JSON Schema boolean `true`.
    Any,
    /// JSON Schema boolean `false` or an explicitly unsatisfiable merge.
    Never,
    /// A local JSON pointer such as `#/$defs/node`.
    Ref(String),
    /// A normal object-form schema.
    Assertions(Box<SchemaAssertions>),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SchemaAssertions {
    pub types: Option<Vec<SchemaType>>,
    pub const_value: Option<Value>,
    pub enum_values: Option<Vec<Value>>,
    pub object: Option<ObjectSchema>,
    pub array: Option<ArraySchema>,
    pub string: Option<StringSchema>,
    pub number: Option<NumberSchema>,
    pub any_of: Vec<Schema>,
    pub one_of: Vec<Schema>,
    pub all_of: Vec<Schema>,
    pub not: Option<Schema>,
}

impl SchemaAssertions {
    pub fn is_empty(&self) -> bool {
        self.types.is_none()
            && self.const_value.is_none()
            && self.enum_values.is_none()
            && self.object.is_none()
            && self.array.is_none()
            && self.string.is_none()
            && self.number.is_none()
            && self.any_of.is_empty()
            && self.one_of.is_empty()
            && self.all_of.is_empty()
            && self.not.is_none()
    }

    pub fn has_value_assertions_without_combinators(&self) -> bool {
        self.types.is_some()
            || self.const_value.is_some()
            || self.enum_values.is_some()
            || self.object.is_some()
            || self.array.is_some()
            || self.string.is_some()
            || self.number.is_some()
    }

    pub fn clone_without_combinators(&self) -> Self {
        Self {
            types: self.types.clone(),
            const_value: self.const_value.clone(),
            enum_values: self.enum_values.clone(),
            object: self.object.clone(),
            array: self.array.clone(),
            string: self.string.clone(),
            number: self.number.clone(),
            any_of: Vec::new(),
            one_of: Vec::new(),
            all_of: Vec::new(),
            not: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum SchemaType {
    Null,
    Boolean,
    Object,
    Array,
    String,
    Number,
    Integer,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ObjectSchema {
    pub properties: Vec<PropertySchema>,
    pub required: BTreeSet<String>,
    pub required_order: Vec<String>,
    pub property_dependencies: BTreeMap<String, BTreeSet<String>>,
    pub min_properties: usize,
    pub max_properties: Option<usize>,
    pub pattern_properties: Vec<PatternPropertySchema>,
    pub property_names: Option<Schema>,
    pub additional_properties: AdditionalProperties,
}

impl Default for ObjectSchema {
    fn default() -> Self {
        Self {
            properties: Vec::new(),
            required: BTreeSet::new(),
            required_order: Vec::new(),
            property_dependencies: BTreeMap::new(),
            min_properties: 0,
            max_properties: None,
            pattern_properties: Vec::new(),
            property_names: None,
            additional_properties: AdditionalProperties::AllowAny,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PropertySchema {
    pub name: String,
    pub schema: Schema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PatternPropertySchema {
    pub pattern: String,
    pub schema: Schema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum AdditionalProperties {
    AllowAny,
    Deny,
    Schema(Box<Schema>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArraySchema {
    pub items: Box<Schema>,
    pub prefix_items: Vec<Schema>,
    pub min_items: usize,
    pub max_items: Option<usize>,
}

impl Default for ArraySchema {
    fn default() -> Self {
        Self {
            items: Box::new(Schema::any("<implicit-array-items>")),
            prefix_items: Vec::new(),
            min_items: 0,
            max_items: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct StringSchema {
    pub min_length: usize,
    pub max_length: Option<usize>,
    pub pattern: Option<String>,
    pub format: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct NumberSchema {
    pub integer: bool,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub exclusive_minimum: bool,
    pub exclusive_maximum: bool,
    pub multiple_of: Option<f64>,
    pub format: Option<String>,
}

impl Schema {
    pub fn normalize_locations_relative(&mut self) {
        let root = self.location.clone();
        self.normalize_locations_relative_to(&root);
    }

    fn normalize_locations_relative_to(&mut self, root: &str) {
        if self.location == root {
            self.location = "#".to_string();
        } else if let Some(suffix) = self.location.strip_prefix(root) {
            if suffix.starts_with('/') {
                self.location = format!("#{suffix}");
            }
        }
        let SchemaKind::Assertions(assertions) = &mut self.kind else {
            return;
        };
        for schema in assertions
            .any_of
            .iter_mut()
            .chain(assertions.one_of.iter_mut())
            .chain(assertions.all_of.iter_mut())
        {
            schema.normalize_locations_relative_to(root);
        }
        if let Some(schema) = assertions.not.as_mut() {
            schema.normalize_locations_relative_to(root);
        }
        if let Some(object) = assertions.object.as_mut() {
            for property in &mut object.properties {
                property.schema.normalize_locations_relative_to(root);
            }
            for property in &mut object.pattern_properties {
                property.schema.normalize_locations_relative_to(root);
            }
            if let Some(schema) = object.property_names.as_mut() {
                schema.normalize_locations_relative_to(root);
            }
            if let AdditionalProperties::Schema(schema) = &mut object.additional_properties {
                schema.normalize_locations_relative_to(root);
            }
        }
        if let Some(array) = assertions.array.as_mut() {
            array.items.normalize_locations_relative_to(root);
            for schema in &mut array.prefix_items {
                schema.normalize_locations_relative_to(root);
            }
        }
    }

    pub fn any(location: impl Into<String>) -> Self {
        Self { location: location.into(), kind: SchemaKind::Any }
    }

    pub fn never(location: impl Into<String>) -> Self {
        Self { location: location.into(), kind: SchemaKind::Never }
    }

    pub fn assertions(location: impl Into<String>, assertions: SchemaAssertions) -> Self {
        Self { location: location.into(), kind: SchemaKind::Assertions(Box::new(assertions)) }
    }
}
