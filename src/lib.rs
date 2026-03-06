//! GLRMask: Efficient Grammar-Constrained Decoding
//!
//! This library compiles context-free grammars and tokenizers into deterministic
//! weighted automata (DWAs), enabling microsecond-scale mask computation during
//! LLM inference.
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use glrmask::{Constraint, Vocab};
//!
//! let vocab = Vocab::new(entries, Some(eos_id));
//! let constraint = Constraint::from_ebnf(grammar, &vocab)?;
//! let mut state = constraint.start();
//!
//! loop {
//!     let mask = state.compute_mask(&constraint);
//!     if state.is_accepting(&constraint) { break; }
//!     let token = sample(logits, &mask);
//!     state.commit(&constraint, token)?;
//! }
//! ```
//!
//! # Module Organization
//!
//! - [`compiler`]: Compilation pipeline (grammar → DWA → constraint)
//! - [`frontend`]: Grammar parsing (EBNF, Lark, JSON Schema)
//! - [`runtime`]: Mask computation and state management
//! - [`automata`]: Finite automata (DFA, NFA, DWA, NWA)
//! - [`ds`]: Data structures (bitset, rangeset)

#![deny(warnings)]

pub mod automata;
pub mod compiler;
pub mod ds;
pub mod frontend;
pub mod runtime;

// Re-export public API types
pub use ds::bitset::BitSet;
pub use runtime::{Constraint, ConstraintState};

use thiserror::Error;

/// Errors that can occur during grammar compilation or constraint operations.
#[derive(Error, Debug)]
pub enum GlrMaskError {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, GlrMaskError>;

/// The vocabulary: token ID → byte sequence mapping.
///
/// Tokens carry their own IDs — the index in `entries` is NOT the token ID.
/// This allows sparse vocabularies (e.g., special tokens with high IDs).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    /// (token_id, byte_sequence) pairs.
    pub entries: Vec<(u32, Vec<u8>)>,
    /// End-of-sequence token ID, if any.
    pub eos_token_id: Option<u32>,
}

impl Vocab {
    /// Well-known EOS token byte sequence (GPT-2 / GPT-NeoX / LLaMA / etc.).
    const EOS_BYTES: &[u8] = b"<|endoftext|>";

    /// Create a new vocabulary from (id, bytes) pairs.
    ///
    /// If `eos_token_id` is `None`, auto-detects by looking for a token whose
    /// bytes are `<|endoftext|>`.
    pub fn new(entries: Vec<(u32, Vec<u8>)>, eos_token_id: Option<u32>) -> Self {
        let eos = eos_token_id.or_else(|| {
            entries.iter().find_map(|(id, bytes)| {
                if bytes == Self::EOS_BYTES { Some(*id) } else { None }
            })
        });
        Self {
            entries,
            eos_token_id: eos,
        }
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum token ID + 1 (determines bitvector size for masks).
    pub fn max_token_id(&self) -> u32 {
        self.entries
            .iter()
            .map(|(id, _)| *id)
            .max()
            .map(|id| id + 1)
            .unwrap_or(0)
    }
}
