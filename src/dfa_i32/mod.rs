//! Unweighted finite automata (DFA and NFA).
//!
//! This module provides DFA and NFA types that wrap rustfst internally for now.
//! These are used for building template DFAs which don't need weights during
//! construction. The DFA can be converted to a DWA when weights are needed.
//!
//! The API mirrors the weighted_automata module structure for familiarity.

pub mod dfa;
pub mod nfa;

pub use dfa::{DFA, DFAState, DFAStates};
pub use nfa::{NFA, NFAState, NFAStates};
