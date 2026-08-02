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
pub(crate) struct SchemaDocument {
    pub(crate) root: Schema,
    pub(crate) definitions: Vec<SchemaDefinition>,
    pub(crate) ref_targets: Vec<SchemaDefinition>,
}

#[derive(Debug, Clone)]
pub(crate) struct SchemaDefinition {
    pub(crate) pointer: String,
    pub(crate) schema: Schema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Schema {
    pub(crate) location: String,
    pub(crate) kind: SchemaKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum SchemaKind {
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
pub(crate) struct SchemaAssertions {
    pub(crate) types: Option<Vec<SchemaType>>,
    pub(crate) const_value: Option<Value>,
    pub(crate) enum_values: Option<Vec<Value>>,
    pub(crate) object: Option<ObjectSchema>,
    pub(crate) array: Option<ArraySchema>,
    pub(crate) string: Option<StringSchema>,
    pub(crate) number: Option<NumberSchema>,
    pub(crate) any_of: Vec<Schema>,
    pub(crate) one_of: Vec<Schema>,
    pub(crate) all_of: Vec<Schema>,
    pub(crate) not: Option<Schema>,
}

impl SchemaAssertions {
    pub(crate) fn is_empty(&self) -> bool {
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

    pub(crate) fn has_value_assertions_without_combinators(&self) -> bool {
        self.types.is_some()
            || self.const_value.is_some()
            || self.enum_values.is_some()
            || self.object.is_some()
            || self.array.is_some()
            || self.string.is_some()
            || self.number.is_some()
    }

    pub(crate) fn clone_without_combinators(&self) -> Self {
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
pub(crate) enum SchemaType {
    Null,
    Boolean,
    Object,
    Array,
    String,
    Number,
    Integer,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ObjectSchema {
    pub(crate) properties: Vec<PropertySchema>,
    pub(crate) required: BTreeSet<String>,
    pub(crate) required_order: Vec<String>,
    pub(crate) property_dependencies: BTreeMap<String, BTreeSet<String>>,
    pub(crate) min_properties: usize,
    pub(crate) max_properties: Option<usize>,
    pub(crate) pattern_properties: Vec<PatternPropertySchema>,
    pub(crate) property_names: Option<Schema>,
    pub(crate) additional_properties: AdditionalProperties,
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
pub(crate) struct PropertySchema {
    pub(crate) name: String,
    pub(crate) schema: Schema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct PatternPropertySchema {
    pub(crate) pattern: String,
    pub(crate) schema: Schema,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum AdditionalProperties {
    AllowAny,
    Deny,
    Schema(Box<Schema>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ArraySchema {
    pub(crate) items: Box<Schema>,
    pub(crate) prefix_items: Vec<Schema>,
    pub(crate) min_items: usize,
    pub(crate) max_items: Option<usize>,
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
pub(crate) struct StringSchema {
    pub(crate) min_length: usize,
    pub(crate) max_length: Option<usize>,
    pub(crate) pattern: Option<String>,
    pub(crate) format: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub(crate) struct NumberSchema {
    pub(crate) integer: bool,
    pub(crate) minimum: Option<f64>,
    pub(crate) maximum: Option<f64>,
    pub(crate) exclusive_minimum: bool,
    pub(crate) exclusive_maximum: bool,
    pub(crate) multiple_of: Option<f64>,
    pub(crate) format: Option<String>,
}

impl Schema {
    pub(crate) fn normalize_locations_relative(&mut self) {
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

    pub(crate) fn any(location: impl Into<String>) -> Self {
        Self { location: location.into(), kind: SchemaKind::Any }
    }

    pub(crate) fn never(location: impl Into<String>) -> Self {
        Self { location: location.into(), kind: SchemaKind::Never }
    }

    pub(crate) fn assertions(location: impl Into<String>, assertions: SchemaAssertions) -> Self {
        Self { location: location.into(), kind: SchemaKind::Assertions(Box::new(assertions)) }
    }
}
