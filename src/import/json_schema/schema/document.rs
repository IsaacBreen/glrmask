use super::Schema;

/// A loaded JSON Schema document.
///
/// `root` is the schema denotation for `#`.  `definitions` are schema objects
/// discovered under `$defs` and legacy `definitions`.  `ref_targets` are other
/// local pointer targets that may be referenced even when they are not in a
/// definitions container, such as property schemas with local ids.
#[derive(Debug, Clone)]
pub(crate) struct SchemaDocument {
    pub(crate) root: Schema,
    pub(crate) definitions: Vec<SchemaDefinition>,
    pub(crate) ref_targets: Vec<SchemaDefinition>,
}

/// A named local reference target.
#[derive(Debug, Clone)]
pub(crate) struct SchemaDefinition {
    pub(crate) pointer: String,
    pub(crate) schema: Schema,
}
