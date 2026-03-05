//! Constraint and ConstraintState — the main runtime types.
//!
//! `Constraint` holds all compiled artifacts needed at inference time.
//! `ConstraintState` tracks per-sequence state and computes token masks.

use std::collections::BTreeMap;

use crate::automata::weighted::dwa::CompDwa;
use crate::compiler::glr::table::GlrTable;
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::ds::rangeset::RangeSet;
use crate::GlrMaskError;

/// A compiled grammar constraint, ready for inference.
///
/// Immutable after creation. Thread-safe (`Send + Sync`).
/// Create [`ConstraintState`] instances from this to track per-sequence state.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Constraint {
    /// The compiled parser DWA.
    /// Labels = parser state IDs (i32), weights = token bitvectors.
    pub(crate) parser_dwa: CompDwa,

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

    /// Per-TSID: { terminal_id → token RangeSet }.
    /// `possible_matches[tsid][terminal] = set of allowed token IDs`.
    pub(crate) possible_matches: Vec<BTreeMap<TerminalId, RangeSet>>,

    /// Maximum token ID in the vocabulary.
    pub(crate) max_token: u32,

    /// EOS token ID, if any.
    pub(crate) eos_token_id: Option<u32>,
}

impl Constraint {
    /// Number of DWA states.
    pub fn num_dwa_states(&self) -> u32 {
        self.parser_dwa.num_states()
    }

    /// Number of parser states (GLR table states).
    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states as u32
    }

    /// Number of token-set IDs.
    pub fn num_tsids(&self) -> u32 {
        self.num_tsids
    }

    /// Create a new `ConstraintState` at the start position.
    pub fn start(&self) -> ConstraintState {
        ConstraintState {
            // Phase 4 will replace this with full GSS + tokenizer state tracking.
            dwa_state: self.parser_dwa.start_state,
        }
    }
}

/// Per-sequence constraint state.
///
/// Tracks the current parse + tokenizer state. Computes token masks and
/// advances state when tokens are committed.
///
/// NOTE: Phase 4 will significantly expand this to include the full GSS,
/// multiple tokenizer states (TSIDs), and proper GLR parser stepping.
#[derive(Debug, Clone)]
pub struct ConstraintState {
    /// Current DWA state (placeholder — Phase 4 will add GSS, tokenizer state, etc.)
    #[allow(dead_code)]
    dwa_state: u32,
}

impl ConstraintState {
    /// Get the current DWA state.
    pub fn current_dwa_state(&self) -> u32 {
        self.dwa_state
    }

    /// Commit a token: advance the state.
    ///
    /// Phase 4 will implement the full commit logic:
    /// 1. Run tokenizer on token bytes
    /// 2. Step GLR parser on matched terminals
    /// 3. Update GSS + DWA state
    pub fn commit(&mut self, _constraint: &Constraint, _token_id: u32) -> std::result::Result<(), GlrMaskError> {
        // TODO: Phase 4 implementation
        Err(GlrMaskError::Compilation("commit not yet implemented".into()))
    }

    /// Compute the allowed-token mask for the current state.
    ///
    /// Phase 4 will implement the full mask computation:
    /// 1. For each (TSID, GSS head), walk the DWA reading parser states
    /// 2. Project DWA weights to current TSID
    /// 3. Union all projected token sets
    pub fn compute_mask(&self, _constraint: &Constraint) -> Vec<bool> {
        // TODO: Phase 4 implementation
        Vec::new()
    }

    /// Whether the current state is accepting (valid end-of-sequence).
    pub fn is_accepting(&self, _constraint: &Constraint) -> bool {
        // TODO: Phase 4 implementation
        false
    }
}
