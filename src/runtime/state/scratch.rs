//! Commit and Mask scratch buffers owned by [`ConstraintState`](super::ConstraintState).
//!
//! Scratch buffers are intentionally separated from the live frontier so that
//! cloning a state clones the mathematical configuration but not its temporary
//! allocation history.

use crate::parser::glr::advance::ParserGSS;
use rustc_hash::{FxHashMap, FxHashSet};

/// Reusable scratch buffers for `commit_bytes_impl`, retained between calls
/// to avoid repeated heap allocation.
#[derive(Debug, Default)]
pub(crate) struct CommitBuffers {
    pub advance_result_cache: FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>,
    pub pending_state: FxHashMap<u32, ParserGSS>,
    pub seen_matches: FxHashSet<(usize, u32)>,
    pub terminal_result_cache: FxHashMap<u32, ParserGSS>,
    pub exec_results: FxHashMap<u32, crate::automata::lexer::tokenizer::TokenizerExecResult>,
    pub remapped_tokenizer_states: FxHashMap<u32, u32>,
    pub accepted_terminals: FxHashMap<u32, FxHashSet<u32>>,
    pub processing_queue: Vec<FxHashMap<u32, ParserGSS>>,
}

impl Clone for CommitBuffers {
    fn clone(&self) -> Self {
        // Don't clone scratch buffers — start fresh
        Self::default()
    }
}

impl CommitBuffers {
    pub fn clear_all(&mut self) {
        self.advance_result_cache.clear();
        self.pending_state.clear();
        self.seen_matches.clear();
        self.terminal_result_cache.clear();
        self.exec_results.clear();
        self.remapped_tokenizer_states.clear();
        self.accepted_terminals.clear();
        for bucket in &mut self.processing_queue {
            bucket.clear();
        }
    }
}
