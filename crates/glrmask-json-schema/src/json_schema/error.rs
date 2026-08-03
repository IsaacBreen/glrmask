use crate::GlrMaskError;

pub type ImportResult<T> = Result<T, SchemaImportError>;

#[derive(Debug, Clone)]
pub struct SchemaImportError {
    message: String,
}

impl SchemaImportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    pub fn at(location: &str, message: impl AsRef<str>) -> Self {
        Self::new(format!("{location}: {}", message.as_ref()))
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<SchemaImportError> for GlrMaskError {
    fn from(value: SchemaImportError) -> Self {
        GlrMaskError::GrammarParse(value.message)
    }
}
