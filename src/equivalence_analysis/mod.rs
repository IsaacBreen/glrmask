//! Equivalence Analysis Module
//!
//! This module provides algorithms for analyzing equivalence of tokens and states
//! in the context of constrained decoding. It contains:
//!
//! - **State Equivalence Analysis**: Determines which tokenizer states behave
//!   identically for all tokens in a vocabulary, allowing them to be merged.
//!   - `state_equivalence_analysis_fast`: Optimized version with batching
//!   - `state_equivalence_analysis_reference`: Simple reference implementation
//!
//! - **Vocab Equivalence Analysis**: Groups LLM vocabulary tokens that produce
//!   identical parsing behavior across all tokenizer states.
//!   - `vocab_equivalence_analysis_fast`: Highly optimized with precomputation
//!   - `vocab_equivalence_analysis_fast_reference`: Simple fast reference
//!   - `vocab_equivalence_analysis_reference`: Slow graph-based reference
//!
//! - **Combined Equivalence Analysis**: Orchestrates both analyses efficiently,
//!   applying state reduction before vocab analysis for optimal performance.

// State equivalence
mod state_equivalence_analysis_fast;
mod state_equivalence_analysis_reference;

// Vocab equivalence
mod vocab_equivalence_analysis_fast;
mod vocab_equivalence_analysis_fast_reference;
mod vocab_equivalence_analysis_reference;

// Trellis-based ground truth (very slow, for testing only)
mod trellis_equivalence_analysis;

// Combined analysis
mod combined_equivalence_analysis;

// Re-exports: use the fast versions by default
pub use vocab_equivalence_analysis_fast::VocabEquivalenceResult;
pub use combined_equivalence_analysis::compute_combined_equivalence;
