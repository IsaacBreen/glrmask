//! Tokenizer-related DFA modules with u8 labels.
//!
//! This module contains finite automata types using u8 labels (byte values)
//! for tokenizer/lexer operations. This is distinct from dfa_i32/dwa_i32 which
//! use i32 labels for parser state IDs.
//!
//! This module contains:
//! - `dfa`: Finite automata (DFA/NFA) and regex expression types
//! - `tokenizer_ops`: Higher-level tokenizer operations
//! - `string_utils`: String escaping utilities

pub mod dfa;
pub mod tokenizer_ops;
pub mod string_utils;

// Re-export commonly used types for convenience
pub use dfa::*;
pub use tokenizer_ops::*;
pub use string_utils::*;
