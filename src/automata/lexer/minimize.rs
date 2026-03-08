//! NOTE: this file is intentionally gutted.
//! The lexer DFA minimization step should be straightforward once the
//! sep1-style DFA shape is in place.

use super::dfa::DFA;

impl DFA {
    pub fn minimize(&self) -> DFA {
        self.clone()
    }
}
