mod array;
mod ast;
mod combinators;
mod config;
mod error;
mod load;
mod lower;
mod number;
mod object;
mod preflight;
pub(crate) mod string;

#[cfg(test)]
mod tests;

use std::env;

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
    preflight::check_schema_size(schema).map_err(GlrMaskError::from)?;
    let document = load_document(schema).map_err(GlrMaskError::from)?;
    lower_document(&document, config).map_err(GlrMaskError::from)
}

/// The new importer deliberately does not depend on the old post-import grammar
/// simplification pass.
pub(crate) fn simplify_grammar_enabled() -> bool {
    false
}

/// Exact terminal subtraction lowering is disabled by default.
///
/// Set `GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS=1` (or any non-empty,
/// non-falsey value) to enable exact-subtraction lowering in downstream import
/// and compile paths.
///
/// Note: JSON Schema GLRM dumps preserve exact subtraction syntax and do not
/// apply this lowering pass.
pub(crate) fn lower_exact_subtractions_enabled() -> bool {
    match env::var("GLRMASK_JSON_SCHEMA_LOWER_EXACT_SUBTRACTIONS") {
        Ok(value) => {
            let trimmed = value.trim();
            !trimmed.is_empty()
                && !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
        }
        Err(_) => false,
    }
}

/// Fold additional-property excluded-key add-backs into the shared terminal
/// instead of emitting one parser alternative per excluded key. Default ON.
/// Disable with GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK=0 (or false/no/off/empty).
pub(crate) fn share_additional_addback_choices_enabled() -> bool {
    match env::var("GLRMASK_JSON_SCHEMA_SHARE_AP_ADDBACK") {
        Ok(value) => {
            let trimmed = value.trim();
            !trimmed.is_empty()
                && !matches!(
                    trimmed.to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
        }
        Err(_) => true,
    }
}

/// Literal-choice promotion was an optimization knob in the old importer.  The
/// simple importer leaves choices as written.
pub(crate) fn promote_literal_choices_enabled() -> bool {
    false
}
