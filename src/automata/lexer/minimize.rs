//! Lexer-side DFA minimization placeholder.
//!
//! This file is intentionally gutted. The future lexer-DFA minimization step
//! should be straightforward once the sep1-style DFA shape is in place.

use super::dfa::DFA;

impl DFA {
    pub fn minimize(&self) -> DFA {
        todo!("lexer DFA minimization is self-explanatory and intentionally deferred")
    }
}