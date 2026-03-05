//! GLR parser.
//!
//! A Generalized LR parser that operates on the GLR parse table.
//! Used during compilation (not at inference time).

use super::table::GlrTable;

/// GLR parser that uses a Graph-Structured Stack for handling ambiguity.
pub struct GlrParser {
    _table: GlrTable,
}

impl GlrParser {
    /// Create a new parser from a table.
    pub fn new(table: GlrTable) -> Self {
        Self { _table: table }
    }

    // TODO: Implement parse() method
}
