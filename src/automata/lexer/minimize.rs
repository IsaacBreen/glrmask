//! Lexer-side DFA minimization.

use super::dfa::Dfa;

impl Dfa {
    /// Minimize this DFA using Hopcroft's algorithm. Returns a new minimized DFA.
    pub fn minimize(&self) -> Dfa {
        unimplemented!()
    }
}

/// Hopcroft's DFA minimization algorithm.
///
/// Groups states into equivalence classes based on their transition behavior
/// and finalizer sets. States with different finalizers or different transition
/// signatures (w.r.t. equivalence classes) are separated.
fn hopcroft_minimize(dfa: &Dfa) -> Dfa {
    unimplemented!()
}