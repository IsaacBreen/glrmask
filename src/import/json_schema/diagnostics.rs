//! Diagnostics for JSON Schema import failures.
//!
//! The importer reports schema locations using local JSON-pointer-like paths.
//! These errors are converted to `GlrMaskError::GrammarParse` at the facade.

use crate::GlrMaskError;

pub(crate) type ImportResult<T> = Result<T, SchemaImportError>;

#[derive(Debug, Clone)]
pub(crate) struct SchemaImportError {
    message: String,
}

impl SchemaImportError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    pub(crate) fn at(location: &str, message: impl AsRef<str>) -> Self {
        Self::new(format!("{location}: {}", message.as_ref()))
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl From<SchemaImportError> for GlrMaskError {
    fn from(value: SchemaImportError) -> Self {
        GlrMaskError::GrammarParse(value.message)
    }
}


/// Validation keywords that are intentionally rejected by the current importer.
///
/// Centralizing this list keeps publication support claims and loader behavior
/// synchronized.
pub(crate) const UNSUPPORTED_VALIDATION_KEYWORDS: &[&str] = &[
    "propertyNames",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "unevaluatedProperties",
    "unevaluatedItems",
];

/// Returns true when a keyword is part of the documented unsupported surface.
pub(crate) fn is_documented_unsupported_keyword(keyword: &str) -> bool {
    UNSUPPORTED_VALIDATION_KEYWORDS.contains(&keyword)
}
