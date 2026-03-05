//! EBNF grammar parser.
//!
//! Parses Extended Backus-Naur Form grammars into the internal `GrammarDef` IR.

use crate::compiler::grammar_def::GrammarDef;
use crate::GlrMaskError;

/// Parse an EBNF grammar string into a `GrammarDef`.
pub fn parse_ebnf(_input: &str) -> Result<GrammarDef, GlrMaskError> {
    // TODO: Implement EBNF parser
    Err(GlrMaskError::GrammarParse("EBNF parser not yet implemented".into()))
}
