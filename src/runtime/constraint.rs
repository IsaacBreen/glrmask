//! Immutable runtime artifact.
//!
//! `Constraint` owns the compiled data needed by inference-time state
//! machines. Construction and serialization live with the artifact; mutable
//! sequence state is kept in `state.rs`.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet};

use range_set_blaze::RangeSetBlaze;

use crate::automata::lexer::tokenizer::TokenizerDfa;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GlrTable;
use crate::compiler::grammar_def::TerminalId;
use crate::ds::leveled_gss::LeveledGSS;

use super::state::{ConstraintState, terminals_disallowed_fresh};

/// A compiled grammar constraint, ready for inference.
///
/// Immutable after creation. Thread-safe (`Send + Sync`).
/// Create [`ConstraintState`] instances from this to track per-sequence state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub struct Constraint {
    /// The compiled parser DWA.
    /// Labels = parser state IDs (i32), weights = token bitvectors.
    pub(crate) parser_dwa: DWA,

    /// The GLR parse table.
    pub(crate) table: GlrTable,

    /// The byte-level tokenizer DFA.
    pub(crate) tokenizer: TokenizerDfa,

    /// Number of token-set IDs.
    pub(crate) num_tsids: u32,

    /// Tokenizer DFA state → TSID mapping.
    /// `state_to_tsid[dfa_state]` = compacted TSID (u32::MAX if unreachable).
    pub(crate) state_to_tsid: Vec<u32>,

    /// TSID → tokenizer DFA state mapping.
    pub(crate) tsid_to_state: Vec<u32>,

    /// Per-TSID: { terminal_id → token range-set }.
    /// `possible_matches[tsid][terminal] = set of allowed token IDs`.
    #[serde(with = "crate::runtime::serde::serde_vec_btmap_rsb")]
    pub(crate) possible_matches: Vec<BTreeMap<TerminalId, RangeSetBlaze<u32>>>,

    /// Maximum token ID in the vocabulary.
    pub(crate) max_token: u32,

    /// EOS token ID, if any.
    pub(crate) eos_token_id: Option<u32>,

    /// Token ID → byte sequence mapping.
    pub(crate) token_bytes: BTreeMap<u32, Vec<u8>>,

    /// Precomputed reachable terminals per tokenizer DFA state.
    /// `reachable_terminals[state]` = set of terminals reachable from `state`.
    /// Immutable after construction; avoids ~0.7ms fixed-point computation per mask.
    #[serde(skip)]
    pub(crate) reachable_terminals: Vec<BTreeSet<TerminalId>>,
}

impl Constraint {
    /// Create a new `ConstraintState` at the start position.
    pub fn start(&self) -> ConstraintState<'_> {
        // The initial parser state is 0.
        // The initial tokenizer state is 0 (initial DFA state).
        let initial_parser_state = 0u32;
        let initial_tok_state = self.tokenizer.initial_state();

        let mut state = BTreeMap::new();
        let gss = LeveledGSS::from_stacks(&[(vec![initial_parser_state], terminals_disallowed_fresh())]);
        state.insert(initial_tok_state, gss);

        ConstraintState { constraint: self, state }
    }

    /// Number of `u32` words required in a mask buffer for this vocabulary.
    ///
    /// Allocate the buffer with `vec![0u32; self.constraint.mask_len()]`.
    /// Token `i` is allowed iff `buf[i / 32] & (1u32 << (i % 32)) != 0`.
    pub fn mask_len(&self) -> usize {
        unimplemented!()
    }

    /// Access the compiled parser DWA (for debugging/analysis).
    pub fn parser_dwa(&self) -> &DWA {
        unimplemented!()
    }
}