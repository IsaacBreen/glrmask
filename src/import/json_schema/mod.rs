//! JSON Schema importer.
//!
//! This module deliberately treats JSON Schema as a value-level constraint
//! language, not as a grammar language.  The importer is therefore organized as
//! a sequence of mathematical interpretations:
//!
//! ```text
//! serde_json::Value
//!   -> schema::SchemaDocument             // typed, located schema syntax
//!   -> load::reference targets            // local reference graph boundary
//!   -> normalize::semantic combinators     // safe schema algebra rewrites
//!   -> lower::Lowerer                      // grammar_ir/NamedGrammar emission
//! ```
//!
//! The public surface remains intentionally tiny: callers provide a JSON value
//! and receive the project grammar IR.  All JSON-Schema-specific policy stays in
//! this namespace.

pub(crate) mod diagnostics;
pub(crate) mod load;
pub(crate) mod lower;
pub(crate) mod normalize;
pub(crate) mod options;
pub(crate) mod schema;

#[cfg(test)]
mod tests;

use serde_json::Value;

use crate::GlrMaskError;
use crate::import::ast::NamedGrammar;

use self::load::load_document;
use self::lower::lower_document;
use self::options::JsonSchemaConfig;

/// Convert a JSON Schema value into the project's named grammar IR.
///
/// This function is the importer facade.  It intentionally exposes no internal
/// JSON Schema structs because those structs are not part of the crate API: they
/// describe one implementation strategy for the schema-to-grammar compiler.
///
/// The conversion has three deliberately separated responsibilities:
///
/// 1. [`load_document`] parses JSON values into a typed, located schema syntax.
/// 2. `normalize` helpers perform semantics-preserving or documented
///    over-approximation rewrites for combinators.
/// 3. [`lower_document`] emits grammar rules over JSON lexical terminals.
///
/// Unsupported schema keywords are rejected while loading whenever rejecting is
/// safer than silently broadening.  Known annotations and explicitly-broadened
/// constructs are documented in `docs/json_schema_support.md` and
/// `docs/refactor/chunk_08/semantic_coverage_matrix.md`.
pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {
    let config = JsonSchemaConfig::from_env();
    let document = load_document(schema).map_err(GlrMaskError::from)?;
    lower_document(&document, config).map_err(GlrMaskError::from)
}

/// Whether the post-import grammar simplification pass should run for JSON Schema.
pub(crate) fn simplify_grammar_enabled() -> bool {
    options::simplify_grammar_enabled()
}

/// Whether exact subtraction lowering should run after JSON Schema import.
pub(crate) fn lower_exact_subtractions_enabled() -> bool {
    options::lower_exact_subtractions_enabled()
}

/// Whether literal choices should be promoted after JSON Schema import.
pub(crate) fn promote_literal_choices_enabled() -> bool {
    options::promote_literal_choices_enabled()
}
