//! Typed JSON Schema syntax used by the importer.
//!
//! These structs are **not** grammar constructs.  They represent a loaded JSON
//! Schema document as value-level assertions.  The grammar interpretation lives
//! under `lower`.

mod array;
mod assertions;
mod document;
mod object;
mod scalar;

pub(crate) use self::array::ArraySchema;
pub(crate) use self::assertions::SchemaAssertions;
pub(crate) use self::document::{SchemaDefinition, SchemaDocument};
pub(crate) use self::object::{AdditionalProperties, ObjectSchema, PatternPropertySchema, PropertySchema};
pub(crate) use self::scalar::{NumberSchema, SchemaType, StringSchema};

/// A located schema node.
#[derive(Debug, Clone)]
pub(crate) struct Schema {
    pub(crate) location: String,
    pub(crate) kind: SchemaKind,
}

/// Top-level schema alternatives after loading.
#[derive(Debug, Clone)]
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

impl Schema {
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
