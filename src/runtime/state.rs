use std::collections::BTreeMap;

use crate::compiler::glr::parser::{ParserGSS, stacks_finished};
use rustc_hash::{FxHashMap, FxHashSet};

use super::constraint::Constraint;

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
    }
}

#[derive(Debug, Clone)]
pub struct ConstraintState<'a> {
    pub(crate) constraint: &'a Constraint,
    pub(crate) state: BTreeMap<u32, ParserGSS>,
    pub(crate) buffers: CommitBuffers,
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
