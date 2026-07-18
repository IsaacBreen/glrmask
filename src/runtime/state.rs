use crate::automata::lexer::Lexer;
use std::sync::Mutex;
use std::collections::{BTreeMap, VecDeque};

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
    pub output_buf: Vec<u32>,
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
    pub small_exec_result: crate::automata::lexer::tokenizer::TokenizerExecResult,
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
        self.small_exec_result.end_state.clear();
        self.small_exec_result.matches.clear();
        for bucket in &mut self.processing_queue {
            bucket.clear();
        }
    }
}

#[derive(Clone)]
pub(crate) struct StateSnapshot {
    pub state: BTreeMap<u32, ParserGSS>,
    pub generation: u64,
}

/// Mutable parser state for one generated sequence.
///
/// Obtain a mask, sample a permitted token, and commit it to advance the state.
/// Create separate states for concurrently generated sequences.
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
    /// Maximum number of token commits whose pre-commit states are retained.
    pub(crate) max_rollback_tokens: usize,
    /// Bounded pre-commit snapshots for token-level rollback.
    pub(crate) history: VecDeque<StateSnapshot>,
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
            max_rollback_tokens: self.max_rollback_tokens,
            history: self.history.clone(),
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
    pub(crate) fn clone_without_history(&self) -> Self {
        Self {
            constraint: self.constraint,
            state: self.state.clone(),
            buffers: self.buffers.clone(),
            generation: self.generation,
            mask_cache: Mutex::new(None),
            mask_scratch: Mutex::new(MaskScratch::default()),
            max_rollback_tokens: 0,
            history: VecDeque::new(),
        }
    }

    pub(crate) fn record_pre_commit_snapshot(&mut self) {
        if self.max_rollback_tokens == 0 {
            return;
        }
        if self.history.len() == self.max_rollback_tokens {
            self.history.pop_front();
        }
        self.history.push_back(StateSnapshot {
            state: self.state.clone(),
            generation: self.generation,
        });
    }

    /// Roll back committed tokens retained by `start_with_rollback`.
    pub fn rollback(&mut self, num_tokens: usize) -> Result<(), String> {
        if num_tokens == 0 {
            return Ok(());
        }
        if num_tokens > self.history.len() {
            return Err(format!(
                "rollback requested {num_tokens} tokens but only {} are available",
                self.history.len()
            ));
        }
        let target_index = self.history.len() - num_tokens;
        let snapshot = self.history[target_index].clone();
        self.history.truncate(target_index);
        self.state = snapshot.state;
        self.generation = snapshot.generation;
        self.buffers.clear_all();
        *self.mask_cache.lock().unwrap() = None;
        *self.mask_scratch.lock().unwrap() = MaskScratch::default();
        Ok(())
    }

    /// Return the longest valid prefix of `tokens` without modifying this state.
    pub fn validate_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        let mut cursor = self.clone_without_history();
        let mut accepted = Vec::with_capacity(tokens.len());
        for &token in tokens {
            if cursor.commit_token(token).is_err() || cursor.is_failed() {
                break;
            }
            accepted.push(token);
        }
        accepted
    }

    /// Return whether no valid parser state remains.
    pub fn is_failed(&self) -> bool {
        self.state.is_empty()
    }

    /// Return whether the committed prefix completes the grammar.
    pub fn is_complete(&self) -> bool {
        let initial_tsid = self.constraint.tokenizer.initial_state();
        let Some(stack) = self.state.get(&initial_tsid) else {
            return false;
        };
        !stack.is_empty() && stacks_finished(&self.constraint.table, stack)
    }

    /// Return whether generation has finished.
    ///
    /// This is currently equivalent to [`ConstraintState::is_complete`].
    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }

    pub(crate) fn parser_root_count(&self) -> usize {
        self.state.values().map(|gss| gss.peek_values().len()).sum()
    }

    pub(crate) fn parser_path_count(&self, limit: usize) -> usize {
        self.state.values().map(|gss| gss.path_count_at_most(limit)).sum::<usize>().min(limit)
    }

    pub(crate) fn has_parser_ambiguity(&self) -> bool {
        self.parser_path_count(2) > 1
    }

    /// Return all flattened parser stacks for debugging.
    /// Each entry is (tokenizer_state, Vec<(stack_of_parser_states, disallowed_terminals)>).
    pub(crate) fn debug_parser_stacks(&self) -> Vec<(u32, Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)>)> {
        self.state.iter().map(|(&ts, gss)| {
            let stacks = gss.to_stacks(4_096).expect("stack enumeration exceeded explicit limit");
            let formatted: Vec<(Vec<u32>, Vec<(u32, Vec<u32>)>)> = stacks.into_iter().map(|(stack, acc)| {
                let disallowed: Vec<(u32, Vec<u32>)> = acc.0.iter().map(|(&k, v)| {
                    (k, v.iter().copied().collect())
                }).collect();
                (stack, disallowed)
            }).collect();
            (ts, formatted)
        }).collect()
    }

    /// Return a forced token sequence when one can be determined.
    pub fn forced(&self) -> Vec<u32> {
        self.forced_impl(false)
    }

    pub(crate) fn forced_dynamic(&self) -> Vec<u32> {
        self.forced_impl(true)
    }

    fn forced_impl(&self, dynamic: bool) -> Vec<u32> {
        if self.is_complete() {
            return Vec::new();
        }

        self.forced_by_bytes(dynamic)
            .unwrap_or_else(|| self.single_token_forced(dynamic))
    }

    fn mask_for_forced(&self, dynamic: bool) -> Vec<u32> {
        if dynamic {
            let mut mask = vec![0u32; self.constraint.mask_len()];
            self.fill_mask_dynamic(&mut mask);
            mask
        } else {
            self.mask()
        }
    }

    fn forced_by_bytes(&self, dynamic: bool) -> Option<Vec<u32>> {
        let forced_bytes = self.compute_forced_byte_prefix(dynamic);
        let tokens = self.tokenize_forced_with_stop(&forced_bytes);
        (!tokens.is_empty()).then_some(tokens)
    }

    fn single_token_forced(&self, dynamic: bool) -> Vec<u32> {
        let mut forced = Vec::new();
        let mut cursor = self.clone();

        loop {
            let mask = cursor.mask_for_forced(dynamic);
            let Some(token) = single_allowed_token(&mask) else {
                break;
            };
            forced.push(token);
            if dynamic {
                cursor
                    .commit_token_dynamic(token)
                    .expect("forced token should be in vocabulary");
            } else {
                cursor
                    .commit_token(token)
                    .expect("forced token should be in vocabulary");
            }
            if cursor.state.is_empty() || cursor.is_complete() {
                break;
            }
        }

        forced
    }

    fn compute_forced_byte_prefix(&self, dynamic: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut cursor = self.clone();
        const MAX_FORCED_BYTES: usize = 10_000;

        loop {
            if bytes.len() >= MAX_FORCED_BYTES {
                break;
            }

            let mask = cursor.mask_for_forced(dynamic);
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
