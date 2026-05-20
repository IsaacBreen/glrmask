mod array;
mod ast;
mod combinators;
mod config;
mod error;
mod load;
mod lower;
mod number;
mod object;
mod string;

#[cfg(test)]
mod tests;

use serde_json::Value;

use crate::GlrMaskError;
use crate::import::ast::NamedGrammar;

use self::config::JsonSchemaConfig;
use self::load::load_document;
use self::lower::lower_document;

/// Convert a JSON Schema value into the project's named grammar AST.
///
/// The implementation intentionally has two phases:
///
/// 1. [`load_document`] parses serde_json data into a typed schema AST.
/// 2. [`lower_document`] lowers that schema AST into `GrammarExpr` rules.
///
/// Unsupported schema keywords are rejected while loading so the lowering phase
/// is not forced to carry partially-understood JSON values.
pub fn schema_to_named_grammar(schema: &Value) -> Result<NamedGrammar, GlrMaskError> {
    let config = JsonSchemaConfig::from_env();
    let document = load_document(schema).map_err(GlrMaskError::from)?;
    lower_document(&document, config).map_err(GlrMaskError::from)
}

/// The new importer deliberately does not depend on the old post-import grammar
/// simplification pass.
pub(crate) fn simplify_grammar_enabled() -> bool {
    false
}

/// Exact terminal subtraction is kept enabled because open-object lowering uses
/// `JSON_STRING - {fixed literal keys}` for additional-property keys.
pub(crate) fn lower_exact_subtractions_enabled() -> bool {
    true
}

/// Literal-choice promotion was an optimization knob in the old importer.  The
/// simple importer leaves choices as written.
pub(crate) fn promote_literal_choices_enabled() -> bool {
    false
}
