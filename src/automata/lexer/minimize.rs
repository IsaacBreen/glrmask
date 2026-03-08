//! NOTE: this file is intentionally gutted.
//! The lexer DFA minimization step should be straightforward once the
//! sep1-style DFA shape is in place.
// SEP1_MAP: sep1 performs the nearest lexer-DFA minimization work inside `dfa_u8/dfa.rs`; glrmask keeps it split out as a separate placeholder.

use super::dfa::DFA;

impl DFA {
    pub fn minimize(&self) -> DFA {
        self.clone()
    }
}
