//! Lexer-side NFA → DFA determinization.

use super::dfa::DFA;
use super::nfa::NFA;

impl NFA {
    pub fn to_dfa(&self) -> DFA {
        todo!("lexer NFA determinization is intentionally deferred until the sep1-style DFA rewrite")
    }
}