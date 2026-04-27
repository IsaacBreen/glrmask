use std::sync::Mutex;
use std::collections::BTreeMap;

use crate::compiler::glr::parser::{ParserGSS, stacks_finished};
use rustc_hash::{FxHashMap, FxHashSet};

use super::constraint::Constraint;

/// Cached fill_mask result, keyed on generation counter.
pub(crate) struct MaskCacheData {
    pub generation: u64,
    pub mask: Vec<u32>,
    /// The merged internal token dense bitmap used to compute this mask.
    /// Enables incremental updates when the state changes slightly.
    pub merged_dense: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct MaskScratch {
    pub merged_dense: Vec<u64>,
    pub chain_merged_dense: Vec<u64>,
}

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

pub struct ConstraintState<'a> {
    pub(crate) constraint: &'a Constraint,
    pub(crate) state: BTreeMap<u32, ParserGSS>,
    pub(crate) buffers: CommitBuffers,
    /// Monotonically increasing counter, bumped on every commit.
    /// Used for cheap cache invalidation in fill_mask.
    pub(crate) generation: u64,
    /// Cached fill_mask result: returned directly when state matches cached snapshot.
    /// Not cloned — clone starts with empty cache.
    pub(crate) mask_cache: Mutex<Option<MaskCacheData>>,
    /// Reusable scratch buffers for fill_mask to avoid per-call allocation.
    pub(crate) mask_scratch: Mutex<MaskScratch>,
}

impl<'a> Clone for ConstraintState<'a> {
    fn clone(&self) -> Self {
        ConstraintState {
            constraint: self.constraint,
            state: self.state.clone(),
            buffers: self.buffers.clone(),
            generation: self.generation,
            mask_cache: Mutex::new(None),
            mask_scratch: Mutex::new(MaskScratch::default()),
        }
    }
}

impl<'a> std::fmt::Debug for ConstraintState<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintState")
            .field("state_len", &self.state.len())
            .field("mask_cached", &self.mask_cache.lock().unwrap().is_some())
            .finish()
    }
}

impl<'a> ConstraintState<'a> {
    pub fn is_complete(&self) -> bool {
        let initial_tsid = self.constraint.tokenizer.initial_state();
        let Some(stack) = self.state.get(&initial_tsid) else {
            return false;
        };
        !stack.is_empty() && stacks_finished(&self.constraint.table, stack)
    }

    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }

    pub fn parser_root_count(&self) -> usize {
        self.state.values().map(|gss| gss.peek_values().len()).sum()
    }

    pub fn parser_path_count(&self, limit: usize) -> usize {
        self.state.values().map(|gss| gss.path_count_at_most(limit)).sum::<usize>().min(limit)
    }

    /// Return all flattened parser stacks for debugging.
    /// Each entry is (tokenizer_state, Vec<(stack_of_parser_states, disallowed_terminals)>).
    pub fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
        self.state.iter().map(|(&ts, gss)| {
            let stacks = gss.to_stacks();
            let formatted: Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)> = stacks.into_iter().map(|(stack, acc)| {
                let disallowed: Vec<(u32, Vec<u32>)> = acc.0.iter().map(|(&k, v)| {
                    (k, v.iter().copied().collect())
                }).collect();
                (stack, disallowed)
            }).collect();
            (ts, formatted)
        }).collect()
    }
}
