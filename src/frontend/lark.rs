//! Lark grammar parser.
//!
//! Parses Lark-format grammars into the internal `GrammarDef` IR.

use crate::compiler::grammar_def::GrammarDef;
use crate::GlrMaskError;

/// Parse a Lark grammar string into a `GrammarDef`.
pub fn parse_lark(_input: &str) -> Result<GrammarDef, GlrMaskError> {
    // TODO: Implement Lark parser
    Err(GlrMaskError::GrammarParse("Lark parser not yet implemented".into()))
}
