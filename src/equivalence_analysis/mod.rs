//! Equivalence Analysis Module
//!
//! This module provides algorithms for analyzing equivalence of tokens and states
//! in the context of constrained decoding. It contains:
//!
//! - **State Equivalence Analysis**: Determines which tokenizer states behave
//!   identically for all tokens in a vocabulary, allowing them to be merged.
//!
//! - **Vocab Equivalence Analysis**: Groups LLM vocabulary tokens that produce
//!   identical parsing behavior across all tokenizer states.
//!
//! - **Combined Equivalence Analysis**: Orchestrates both analyses efficiently,
//!   applying state reduction before vocab analysis for optimal performance.

mod state_equivalence_analysis;
mod vocab_equivalence_analysis;
pub mod vocab_equivalence_analysis_fast;
mod vocab_equivalence_analysis_reference;
mod combined_equivalence_analysis;
pub mod vocab_equivalence_trie;

pub use state_equivalence_analysis::find_state_equivalence_classes;
pub use vocab_equivalence_analysis::find_vocab_equivalence_classes;
pub use vocab_equivalence_analysis::VocabEquivalenceResult;
pub use combined_equivalence_analysis::compute_combined_equivalence;
pub use combined_equivalence_analysis::CombinedEquivalenceResult;
pub use combined_equivalence_analysis::find_vocab_equivalence_classes_with_state_reduction;
pub use vocab_equivalence_trie::find_vocab_equivalence_classes_trie;
