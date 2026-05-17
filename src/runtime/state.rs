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

enum ForcedFirstByte {
    None,
    Unique(u8),
    Ambiguous,
}

enum GreedyTokenizationStep {
    Match { token_id: u32, width: usize },
    BlockedByLongerToken,
    NoMatch,
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

    pub fn force(&self) -> Vec<u32> {
        if self.is_complete() {
            return Vec::new();
        }

        self.force_by_bytes()
            .unwrap_or_else(|| self.single_token_force())
    }

    fn force_by_bytes(&self) -> Option<Vec<u32>> {
        let forced_bytes = self.compute_forced_byte_prefix();
        let tokens = self.tokenize_forced_with_stop(&forced_bytes);
        (!tokens.is_empty()).then_some(tokens)
    }

    fn single_token_force(&self) -> Vec<u32> {
        let mut forced = Vec::new();
        let mut cursor = self.clone();

        loop {
            let mask = cursor.mask();
            let Some(token) = single_allowed_token(&mask) else {
                break;
            };
            forced.push(token);
            cursor.commit_token(token).expect("forced token should be in vocabulary");
            if cursor.state.is_empty() || cursor.is_complete() {
                break;
            }
        }

        forced
    }

    fn compute_forced_byte_prefix(&self) -> Vec<u8> {
        let eos = self.constraint.eos_token_id;
        let mut bytes = Vec::new();
        let mut cursor = self.clone();
        const MAX_FORCED_BYTES: usize = 10_000;

        loop {
            if bytes.len() >= MAX_FORCED_BYTES {
                break;
            }

            let mask = cursor.mask();
            if let Some(eos_id) = eos {
                if is_token_set(&mask, eos_id) {
                    break;
                }
            }

            match cursor.forced_first_byte(&mask) {
                ForcedFirstByte::Unique(byte) => {
                    bytes.push(byte);
                    let _ = cursor.commit_bytes(&[byte]);
                    if cursor.state.is_empty() {
                        bytes.pop();
                        break;
                    }
                }
                ForcedFirstByte::None | ForcedFirstByte::Ambiguous => break,
            }
        }

        bytes
    }

    fn forced_first_byte(&self, mask: &[u32]) -> ForcedFirstByte {
        let mut first_byte = None;
        let mut ambiguous = false;
        let mut saw_token = false;

        for_each_set_bit(mask, |token_id| {
            let Some(token_bytes) = self.constraint.token_bytes.get(&token_id) else {
                return;
            };
            let Some(byte) = token_bytes.first().copied() else {
                return;
            };

            saw_token = true;
            match first_byte {
                None => first_byte = Some(byte),
                Some(existing) if existing == byte => {}
                Some(_) => ambiguous = true,
            }
        });

        if !saw_token {
            ForcedFirstByte::None
        } else if ambiguous {
            ForcedFirstByte::Ambiguous
        } else {
            ForcedFirstByte::Unique(first_byte.expect("saw_token implies a first byte"))
        }
    }

    fn tokenize_forced_with_stop(&self, forced_bytes: &[u8]) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < forced_bytes.len() {
            match self.greedy_tokenization_step(&forced_bytes[pos..]) {
                GreedyTokenizationStep::Match { token_id, width } => {
                    tokens.push(token_id);
                    pos += width;
                }
                GreedyTokenizationStep::BlockedByLongerToken
                | GreedyTokenizationStep::NoMatch => break,
            }
        }

        tokens
    }

    fn greedy_tokenization_step(&self, remaining: &[u8]) -> GreedyTokenizationStep {
        let mut best_match = None;
        let mut blocked_by_longer_token = false;

        for (&token_id, token_bytes) in self.constraint.token_bytes.iter() {
            if token_bytes.is_empty() {
                continue;
            }
            if remaining.starts_with(token_bytes) {
                match best_match {
                    Some((_, best_width)) if token_bytes.len() <= best_width => {}
                    _ => best_match = Some((token_id, token_bytes.len())),
                }
                continue;
            }
            if token_bytes.starts_with(remaining) && token_bytes.len() > remaining.len() {
                blocked_by_longer_token = true;
            }
        }

        if blocked_by_longer_token {
            GreedyTokenizationStep::BlockedByLongerToken
        } else if let Some((token_id, width)) = best_match {
            GreedyTokenizationStep::Match { token_id, width }
        } else {
            GreedyTokenizationStep::NoMatch
        }
    }
}

fn is_token_set(mask: &[u32], token_id: u32) -> bool {
    let word_index = token_id as usize / 32;
    let bit = token_id % 32;
    mask.get(word_index).is_some_and(|word| word & (1 << bit) != 0)
}

fn single_allowed_token(mask: &[u32]) -> Option<u32> {
    let mut found = None;
    for (word_index, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as u32;
            let token = word_index as u32 * 32 + bit;
            if found.replace(token).is_some() {
                return None;
            }
            bits &= bits - 1;
        }
    }
    found
}

fn for_each_set_bit(mask: &[u32], mut f: impl FnMut(u32)) {
    for (word_index, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as u32;
            let token_id = word_index as u32 * 32 + bit;
            f(token_id);
            bits &= bits - 1;
        }
    }
}
