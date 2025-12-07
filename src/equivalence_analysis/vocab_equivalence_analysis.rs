//! Vocab Equivalence Analysis
//!
//! Groups LLM vocabulary tokens by their parsing behavior across all tokenizer states.
//! Two tokens are equivalent if they produce identical parsing behavior across all
//! initial tokenizer states.
//!
//! The algorithm builds a parse graph for each token and computes a deterministic hash
//! that captures the full structure of which groups match at which positions.

use std::collections::BTreeSet;

use crate::finite_automata::Regex;

// Re-export from the implementation modules
pub use super::vocab_equivalence_analysis_fast::VocabEquivalenceResult;
use super::vocab_equivalence_analysis_fast;

/// Find vocab equivalence classes of tokens based on DFA behavior.
///
/// Two tokens are equivalent if they produce identical parsing behavior
/// across all initial tokenizer states, including:
/// - Which groups match during parsing
/// - At which positions groups match
/// - The final state and its possible future groups
///
/// # Arguments
/// * `regex` - The tokenizer DFA
/// * `strings` - Vocabulary tokens to analyze
/// * `initial_states` - Tokenizer states to consider for equivalence
///
/// # Returns
/// Sets of token indices that are equivalent (produce identical parsing behavior).
pub fn find_vocab_equivalence_classes(
    regex: &Regex,
    strings: &[Vec<u8>],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    vocab_equivalence_analysis_fast::find_vocab_equivalence_classes(regex, strings, initial_states)
}
