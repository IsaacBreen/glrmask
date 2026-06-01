use std::collections::BTreeSet;

use super::Schema;

/// Object-specific assertions after loading.
#[derive(Debug, Clone)]
pub(crate) struct ObjectSchema {
    pub(crate) properties: Vec<PropertySchema>,
    pub(crate) required: BTreeSet<String>,
    pub(crate) min_properties: usize,
    pub(crate) max_properties: Option<usize>,
    pub(crate) pattern_properties: Vec<PatternPropertySchema>,
    pub(crate) additional_properties: AdditionalProperties,
}

impl Default for ObjectSchema {
    fn default() -> Self {
        Self {
            properties: Vec::new(),
            required: BTreeSet::new(),
            min_properties: 0,
            max_properties: None,
            pattern_properties: Vec::new(),
            additional_properties: AdditionalProperties::AllowAny,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PropertySchema {
    pub(crate) name: String,
    pub(crate) schema: Schema,
}

#[derive(Debug, Clone)]
pub(crate) struct PatternPropertySchema {
    pub(crate) pattern: String,
    pub(crate) schema: Schema,
}

#[derive(Debug, Clone)]
pub(crate) enum AdditionalProperties {
    AllowAny,
    Deny,
    Schema(Box<Schema>),
}
