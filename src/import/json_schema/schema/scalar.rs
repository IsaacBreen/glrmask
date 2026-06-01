/// JSON Schema primitive type name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SchemaType {
    Null,
    Boolean,
    Object,
    Array,
    String,
    Number,
    Integer,
}

/// String-specific assertions after loading.
///
/// Lengths are expressed in decoded JSON string scalar values, matching JSON
/// Schema's value semantics.  Lowering is responsible for converting these
/// value-level constraints into regular languages over encoded JSON strings.
#[derive(Debug, Clone, Default)]
pub(crate) struct StringSchema {
    pub(crate) min_length: usize,
    pub(crate) max_length: Option<usize>,
    pub(crate) pattern: Option<String>,
    pub(crate) format: Option<String>,
}

/// Number/integer assertions after loading.
///
/// This importer currently stores numeric bounds as `f64`, matching the prior
/// implementation.  A future exact numeric layer should replace this with a
/// decimal-rational representation before broad publication.
#[derive(Debug, Clone, Default)]
pub(crate) struct NumberSchema {
    pub(crate) integer: bool,
    pub(crate) minimum: Option<f64>,
    pub(crate) maximum: Option<f64>,
    pub(crate) exclusive_minimum: bool,
    pub(crate) exclusive_maximum: bool,
    pub(crate) multiple_of: Option<f64>,
}
