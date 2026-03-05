//! JSON Schema → grammar converter.
//!
//! Converts a JSON Schema into a context-free grammar that generates
//! exactly the set of valid JSON strings conforming to the schema.

use crate::compiler::grammar_def::GrammarDef;
use crate::GlrMaskError;

/// Convert a JSON Schema (as a JSON string) into a `GrammarDef`.
pub fn json_schema_to_grammar(_schema_json: &str) -> Result<GrammarDef, GlrMaskError> {
    // TODO: Implement JSON Schema converter
    Err(GlrMaskError::GrammarParse(
        "JSON Schema converter not yet implemented".into(),
    ))
}
