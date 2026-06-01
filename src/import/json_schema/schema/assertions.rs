use serde_json::Value;

use super::{ArraySchema, NumberSchema, ObjectSchema, Schema, SchemaType, StringSchema};

/// Assertions attached to a JSON Schema object.
///
/// The shape intentionally preserves the syntactic distinction between direct
/// assertions and combinators.  Normalization/lowering decides which
/// combinations can be interpreted exactly, merged, factored, or safely
/// broadened.
#[derive(Debug, Clone, Default)]
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
