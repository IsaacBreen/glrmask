//! Tokenizer DFA construction.
//!
//! Builds byte-level DFAs from the vocabulary for matching token bytes
//! against terminal patterns.

use crate::automata::dfa::Dfa;
use crate::Vocab;

/// Build a tokenizer DFA from a vocabulary.
///
/// The resulting DFA recognizes all token byte sequences in the vocabulary.
pub fn build_tokenizer_dfa(_vocab: &Vocab) -> Dfa {
    // TODO: Implement
    Dfa::new(1)
}
