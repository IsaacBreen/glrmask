//! Vocab equivalence analysis implementations: slow, medium, fast.
//!
//! - `fast`: Parallel batched with byte-class compression (production runtime default)
//! - `medium`: Flat DFA with self-loop optimization (validation)
//! - `slow`: Trie-based per-token hashing (validation reference)

pub mod fast;
pub mod medium;
pub mod slow;
