//! NOTE: lexer NFA → DFA determinization is intentionally deferred until the
//! sep1-style DFA rewrite.
// SEP1_MAP: The nearest sep1 analogue is the lexer determinization work inside `dfa_u8/dfa.rs`; glrmask keeps the stage boundary in its own placeholder file.

use super::dfa::DFA;
use super::nfa::NFA;

impl NFA {
    pub fn to_dfa(&self) -> DFA {
        todo!("lexer NFA determinization is intentionally deferred until the sep1-style DFA rewrite")
    }
}