//! Possible-match tables for tokenizer states and vocab-prefix subtrees.

use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};

pub(crate) type PossibleMatchesByState = BTreeMap<u32, BTreeMap<TerminalID, RangeSetBlaze<u32>>>;
type PossibleMatchMap = FxHashMap<TerminalID, RangeSetBlaze<u32>>;

fn debug_profile_enabled() -> bool {
    std::env::var("GLRMASK_DEBUG_PROFILE")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn profile_summary_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_COMPILE_SUMMARY")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "" | "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PossibleMatchesProfile {
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) reachable_cache_hits: u64,
    pub(crate) reachable_cache_misses: u64,
    pub(crate) child_segments_visited: u64,
    pub(crate) byte_steps: u64,
    pub(crate) blocked_segments: u64,
    pub(crate) recursive_descents: u64,
    pub(crate) self_loop_subtrees_skipped: u64,
    pub(crate) terminal_insertions: u64,
    pub(crate) cache_entries: usize,
    pub(crate) reachable_cache_entries: usize,
    pub(crate) cache_lookup_ms: f64,
    pub(crate) reachable_lookup_ms: f64,
    pub(crate) node_terminal_insert_ms: f64,
    pub(crate) segment_walk_ms: f64,
    pub(crate) self_loop_check_ms: f64,
    pub(crate) merge_child_matches_ms: f64,
    pub(crate) root_compute_ms: f64,
    pub(crate) materialize_output_ms: f64,
}

pub(crate) fn emit_possible_matches_profile_summary(
    label: &str,
    token_count: usize,
    state_count: u32,
    trie_build_ms: f64,
    collect_ms: f64,
    profile: &PossibleMatchesProfile,
) {
    if !profile_summary_enabled() {
        return;
    }

    eprintln!(
        "[glrmask/profile][possible_matches] label={} tokens={} states={} trie_build_ms={:.3} collect_ms={:.3} root_compute_ms={:.3} materialize_output_ms={:.3} cache_lookup_ms={:.3} reachable_lookup_ms={:.3} node_terminal_insert_ms={:.3} segment_walk_ms={:.3} self_loop_check_ms={:.3} merge_child_matches_ms={:.3} cache_entries={} reachable_cache_entries={} cache_hits={} cache_misses={} reachable_cache_hits={} reachable_cache_misses={} child_segments={} byte_steps={} blocked_segments={} recursive_descents={} self_loop_subtrees_skipped={} terminal_insertions={}",
        label,
        token_count,
        state_count,
        trie_build_ms,
        collect_ms,
        profile.root_compute_ms,
        profile.materialize_output_ms,
        profile.cache_lookup_ms,
        profile.reachable_lookup_ms,
        profile.node_terminal_insert_ms,
        profile.segment_walk_ms,
        profile.self_loop_check_ms,
        profile.merge_child_matches_ms,
        profile.cache_entries,
        profile.reachable_cache_entries,
        profile.cache_hits,
        profile.cache_misses,
        profile.reachable_cache_hits,
        profile.reachable_cache_misses,
        profile.child_segments_visited,
        profile.byte_steps,
        profile.blocked_segments,
        profile.recursive_descents,
        profile.self_loop_subtrees_skipped,
        profile.terminal_insertions,
    );
}

fn owned_token_entries(token_bytes: &BTreeMap<u32, Vec<u8>>) -> Vec<(u32, Vec<u8>)> {
    token_bytes
        .iter()
        .map(|(token_id, bytes)| (*token_id, bytes.clone()))
        .collect()
}

fn clone_token_entries(token_entries: &[(u32, Vec<u8>)]) -> Vec<(u32, Vec<u8>)> {
    token_entries
        .iter()
        .map(|(token_id, bytes)| (*token_id, bytes.clone()))
        .collect()
}

fn ordered_possible_matches(matches_for_state: Rc<PossibleMatchMap>) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
    match Rc::try_unwrap(matches_for_state) {
        Ok(map) => map.into_iter().collect(),
        Err(shared) => shared
            .iter()
            .map(|(&terminal, token_ids)| (terminal, token_ids.clone()))
            .collect(),
    }
}

fn reachable_u32(node: &VocabPrefixTreeNode) -> RangeSetBlaze<u32> {
    let mut out = RangeSetBlaze::new();
    for range in node.reachable_token_ids().ranges() {
        out.ranges_insert(*range.start() as u32..=*range.end() as u32);
    }
    out
}

fn merge_token_ids(into: &mut RangeSetBlaze<u32>, other: &RangeSetBlaze<u32>) {
    *into |= other;
}

fn merge_possible_match_maps(into: &mut PossibleMatchMap, other: &PossibleMatchMap) {
    for (terminal, token_ids) in other {
        let existing = into.entry(*terminal).or_default();
        merge_token_ids(existing, token_ids);
    }
}

pub(crate) struct PossibleMatchesComputer<'a> {
    tokenizer: &'a Tokenizer,
    cache: FxHashMap<(usize, u32), Rc<PossibleMatchMap>>,
    reachable_cache: FxHashMap<usize, Rc<RangeSetBlaze<u32>>>,
    self_loop_bytes: FxHashMap<u32, U8Set>,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
    summary_profile_enabled: bool,
    profile: PossibleMatchesProfile,
}

impl<'a> PossibleMatchesComputer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            cache: FxHashMap::default(),
            reachable_cache: FxHashMap::default(),
            self_loop_bytes: FxHashMap::default(),
            flat_transitions: vec![None; tokenizer.num_states() as usize],
            summary_profile_enabled: profile_summary_enabled(),
            profile: PossibleMatchesProfile::default(),
        }
    }

    pub(crate) fn profile(&self) -> PossibleMatchesProfile {
        PossibleMatchesProfile {
            cache_entries: self.cache.len(),
            reachable_cache_entries: self.reachable_cache.len(),
            ..self.profile
        }
    }

    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            let dfa_state = &self.tokenizer.dfa.states()[state_idx];
            let mut flat = Box::new([u32::MAX; 256]);
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            self.flat_transitions[state_idx] = Some(flat);
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn reachable_for_node(&mut self, node: &VocabPrefixTreeNode) -> Rc<RangeSetBlaze<u32>> {
        let started_at = self.summary_profile_enabled.then(Instant::now);
        let cache_key = node as *const VocabPrefixTreeNode as usize;
        let reachable = if let Some(cached) = self.reachable_cache.get(&cache_key) {
            self.profile.reachable_cache_hits += 1;
            Rc::clone(cached)
        } else {
            self.profile.reachable_cache_misses += 1;
            let reachable = Rc::new(reachable_u32(node));
            self.reachable_cache.insert(cache_key, Rc::clone(&reachable));
            reachable
        };
        if let Some(started_at) = started_at {
            self.profile.reachable_lookup_ms += elapsed_ms(started_at);
        }
        reachable
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: u32,
    ) -> bool {
        let self_loop_bytes = self.self_loop_bytes.entry(tokenizer_state).or_insert_with(|| {
            let state = &self.tokenizer.dfa.states()[tokenizer_state as usize];
            let mut bytes = U8Set::empty();
            for (byte, &target) in state.transitions.iter() {
                if target == tokenizer_state {
                    bytes.insert(byte);
                }
            }
            bytes
        });
        U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
    }

    pub(crate) fn possible_matches_for_node(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: u32,
    ) -> Rc<PossibleMatchMap> {
        let cache_lookup_started_at = self.summary_profile_enabled.then(Instant::now);
        let cache_key = (node as *const VocabPrefixTreeNode as usize, tokenizer_state);
        if let Some(cached) = self.cache.get(&cache_key) {
            self.profile.cache_hits += 1;
            if let Some(started_at) = cache_lookup_started_at {
                self.profile.cache_lookup_ms += elapsed_ms(started_at);
            }
            return Rc::clone(cached);
        }
        self.profile.cache_misses += 1;
        if let Some(started_at) = cache_lookup_started_at {
            self.profile.cache_lookup_ms += elapsed_ms(started_at);
        }

        let mut result = PossibleMatchMap::default();

        // This intentionally includes the token ending exactly at `node` before
        // recursing into child segments, so the recursive part only adds longer
        // continuations.
        if node.has_token() {
            let insert_started_at = self.summary_profile_enabled.then(Instant::now);
            let token_id = node.token_id() as u32;
            for terminal in self.tokenizer.matched_terminals_iter(tokenizer_state) {
                result.entry(terminal).or_default().insert(token_id);
                self.profile.terminal_insertions += 1;
            }
            if let Some(started_at) = insert_started_at {
                self.profile.node_terminal_insert_ms += elapsed_ms(started_at);
            }
        }

        for (segment_bytes, child) in node.iter_children() {
            self.profile.child_segments_visited += 1;
            let mut current_state = tokenizer_state;
            let mut segment_blocked = false;
            let reachable = self.reachable_for_node(child);

            let segment_walk_started_at = self.summary_profile_enabled.then(Instant::now);
            for &byte in segment_bytes {
                self.profile.byte_steps += 1;
                let Some(next_state) = self.fast_step(current_state, byte) else {
                    segment_blocked = true;
                    break;
                };
                current_state = next_state;
                for terminal in self.tokenizer.matched_terminals_iter(current_state) {
                    let existing = result.entry(terminal).or_default();
                    merge_token_ids(existing, reachable.as_ref());
                    self.profile.terminal_insertions += 1;
                }
            }
            if let Some(started_at) = segment_walk_started_at {
                self.profile.segment_walk_ms += elapsed_ms(started_at);
            }

            if segment_blocked {
                self.profile.blocked_segments += 1;
            }
            if !segment_blocked && !self.tokenizer.is_end(current_state) {
                let self_loop_check_started_at = self.summary_profile_enabled.then(Instant::now);
                if self.can_skip_self_loop_subtree(child, current_state) {
                    if let Some(started_at) = self_loop_check_started_at {
                        self.profile.self_loop_check_ms += elapsed_ms(started_at);
                    }
                    self.profile.self_loop_subtrees_skipped += 1;
                    continue;
                }
                if let Some(started_at) = self_loop_check_started_at {
                    self.profile.self_loop_check_ms += elapsed_ms(started_at);
                }
                self.profile.recursive_descents += 1;
                let child_matches = self.possible_matches_for_node(child, current_state);
                let merge_started_at = self.summary_profile_enabled.then(Instant::now);
                merge_possible_match_maps(&mut result, child_matches.as_ref());
                if let Some(started_at) = merge_started_at {
                    self.profile.merge_child_matches_ms += elapsed_ms(started_at);
                }
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}

// ---------------------------------------------------------------------------
// Dense bitmap variant of PossibleMatchesComputer
// ---------------------------------------------------------------------------

type DensePossibleMatchMap = BTreeMap<TerminalID, Box<[u64]>>;

fn reachable_bitmap(node: &VocabPrefixTreeNode, num_words: usize) -> Vec<u64> {
    let mut words = vec![0u64; num_words];
    for range in node.reachable_token_ids().ranges() {
        let lo = *range.start() as u32;
        let hi = *range.end() as u32;
        for id in lo..=hi {
            words[id as usize / 64] |= 1u64 << (id % 64);
        }
    }
    words
}

#[inline]
fn merge_bitmaps(into: &mut [u64], other: &[u64]) {
    for (a, b) in into.iter_mut().zip(other.iter()) {
        *a |= *b;
    }
}

fn merge_dense_maps(into: &mut DensePossibleMatchMap, other: &DensePossibleMatchMap, num_words: usize) {
    for (&terminal, bitmap) in other {
        let existing = into.entry(terminal).or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
        merge_bitmaps(existing, bitmap);
    }
}

fn bitmap_to_rangeset(words: &[u64]) -> RangeSetBlaze<u32> {
    let mut result = RangeSetBlaze::new();
    for (word_idx, &word) in words.iter().enumerate() {
        if word == 0 { continue; }
        let base = (word_idx as u32) * 64;
        let mut w = word;
        let mut pos = 0u32;
        while w != 0 {
            let zeros = w.trailing_zeros();
            pos += zeros;
            w >>= zeros;
            let ones = if w == u64::MAX { 64 - pos % 64 } else { (!w).trailing_zeros() };
            let run_start = base + pos;
            let run_end = base + pos + ones - 1;
            pos += ones;
            if ones < 64 { w >>= ones; } else { w = 0; }
            result.ranges_insert(run_start..=run_end);
        }
    }
    result
}

pub(crate) struct DensePossibleMatchesComputer<'a> {
    tokenizer: &'a Tokenizer,
    num_words: usize,
    cache: FxHashMap<(usize, u32), Rc<DensePossibleMatchMap>>,
    reachable_cache: FxHashMap<usize, Rc<Vec<u64>>>,
    self_loop_bytes: FxHashMap<u32, U8Set>,
    flat_transitions: Vec<Option<Box<[u32; 256]>>>,
    summary_profile_enabled: bool,
    profile: PossibleMatchesProfile,
}

impl<'a> DensePossibleMatchesComputer<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, num_internal_tokens: u32) -> Self {
        let num_words = (num_internal_tokens as usize + 63) / 64;
        Self {
            tokenizer,
            num_words,
            cache: FxHashMap::default(),
            reachable_cache: FxHashMap::default(),
            self_loop_bytes: FxHashMap::default(),
            flat_transitions: vec![None; tokenizer.num_states() as usize],
            summary_profile_enabled: profile_summary_enabled(),
            profile: PossibleMatchesProfile::default(),
        }
    }

    pub(crate) fn profile(&self) -> PossibleMatchesProfile {
        PossibleMatchesProfile {
            cache_entries: self.cache.len(),
            reachable_cache_entries: self.reachable_cache.len(),
            ..self.profile
        }
    }

    #[inline]
    fn fast_step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let state_idx = state as usize;
        if self.flat_transitions[state_idx].is_none() {
            let dfa_state = &self.tokenizer.dfa.states()[state_idx];
            let mut flat = Box::new([u32::MAX; 256]);
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            self.flat_transitions[state_idx] = Some(flat);
        }
        let next = self.flat_transitions[state_idx].as_ref().unwrap()[byte as usize];
        if next == u32::MAX { None } else { Some(next) }
    }

    fn reachable_for_node(&mut self, node: &VocabPrefixTreeNode) -> Rc<Vec<u64>> {
        let started_at = self.summary_profile_enabled.then(Instant::now);
        let cache_key = node as *const VocabPrefixTreeNode as usize;
        let reachable = if let Some(cached) = self.reachable_cache.get(&cache_key) {
            self.profile.reachable_cache_hits += 1;
            Rc::clone(cached)
        } else {
            self.profile.reachable_cache_misses += 1;
            let reachable = Rc::new(reachable_bitmap(node, self.num_words));
            self.reachable_cache.insert(cache_key, Rc::clone(&reachable));
            reachable
        };
        if let Some(started_at) = started_at {
            self.profile.reachable_lookup_ms += elapsed_ms(started_at);
        }
        reachable
    }

    fn can_skip_self_loop_subtree(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: u32,
    ) -> bool {
        let self_loop_bytes = self.self_loop_bytes.entry(tokenizer_state).or_insert_with(|| {
            let state = &self.tokenizer.dfa.states()[tokenizer_state as usize];
            let mut bytes = U8Set::empty();
            for (byte, &target) in state.transitions.iter() {
                if target == tokenizer_state {
                    bytes.insert(byte);
                }
            }
            bytes
        });
        U8Set::from_words(*node.subtree_bytes()).is_subset(self_loop_bytes)
    }

    pub(crate) fn possible_matches_for_node(
        &mut self,
        node: &VocabPrefixTreeNode,
        tokenizer_state: u32,
    ) -> Rc<DensePossibleMatchMap> {
        let cache_lookup_started_at = self.summary_profile_enabled.then(Instant::now);
        let cache_key = (node as *const VocabPrefixTreeNode as usize, tokenizer_state);
        if let Some(cached) = self.cache.get(&cache_key) {
            self.profile.cache_hits += 1;
            if let Some(started_at) = cache_lookup_started_at {
                self.profile.cache_lookup_ms += elapsed_ms(started_at);
            }
            return Rc::clone(cached);
        }
        self.profile.cache_misses += 1;
        if let Some(started_at) = cache_lookup_started_at {
            self.profile.cache_lookup_ms += elapsed_ms(started_at);
        }

        let num_words = self.num_words;
        let mut result = DensePossibleMatchMap::default();

        if node.has_token() {
            let insert_started_at = self.summary_profile_enabled.then(Instant::now);
            let token_id = node.token_id() as u32;
            for terminal in self.tokenizer.matched_terminals_iter(tokenizer_state) {
                let entry = result.entry(terminal).or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
                self.profile.terminal_insertions += 1;
            }
            if let Some(started_at) = insert_started_at {
                self.profile.node_terminal_insert_ms += elapsed_ms(started_at);
            }
        }

        for (segment_bytes, child) in node.iter_children() {
            self.profile.child_segments_visited += 1;
            let mut current_state = tokenizer_state;
            let mut segment_blocked = false;
            let reachable = self.reachable_for_node(child);

            let segment_walk_started_at = self.summary_profile_enabled.then(Instant::now);
            for &byte in segment_bytes {
                self.profile.byte_steps += 1;
                let Some(next_state) = self.fast_step(current_state, byte) else {
                    segment_blocked = true;
                    break;
                };
                current_state = next_state;
                for terminal in self.tokenizer.matched_terminals_iter(current_state) {
                    let existing = result.entry(terminal).or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                    merge_bitmaps(existing, reachable.as_ref());
                    self.profile.terminal_insertions += 1;
                }
            }
            if let Some(started_at) = segment_walk_started_at {
                self.profile.segment_walk_ms += elapsed_ms(started_at);
            }

            if segment_blocked {
                self.profile.blocked_segments += 1;
            }
            if !segment_blocked && !self.tokenizer.is_end(current_state) {
                let self_loop_check_started_at = self.summary_profile_enabled.then(Instant::now);
                if self.can_skip_self_loop_subtree(child, current_state) {
                    if let Some(started_at) = self_loop_check_started_at {
                        self.profile.self_loop_check_ms += elapsed_ms(started_at);
                    }
                    self.profile.self_loop_subtrees_skipped += 1;
                    continue;
                }
                if let Some(started_at) = self_loop_check_started_at {
                    self.profile.self_loop_check_ms += elapsed_ms(started_at);
                }
                self.profile.recursive_descents += 1;
                let child_matches = self.possible_matches_for_node(child, current_state);
                let merge_started_at = self.summary_profile_enabled.then(Instant::now);
                merge_dense_maps(&mut result, child_matches.as_ref(), num_words);
                if let Some(started_at) = merge_started_at {
                    self.profile.merge_child_matches_ms += elapsed_ms(started_at);
                }
            }
        }

        let result = Rc::new(result);
        self.cache.insert(cache_key, Rc::clone(&result));
        result
    }
}

/// Collect possible_matches using dense bitmap computation internally,
/// returning dense bitmaps directly (no RangeSetBlaze conversion).
pub(crate) fn collect_possible_matches_by_internal_tsid_dense(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    tokenizer_state_ids: &ManyToOneIdMap,
    num_internal_tokens: u32,
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    // For small workloads, the serial path with FxHashMap cache is faster.
    // For larger workloads, the batched trie walk eliminates cache overhead.
    if tokenizer_state_ids.num_internal_ids() < 5000 {
        return collect_possible_matches_dense_serial(
            tokenizer, root, tokenizer_state_ids, num_internal_tokens,
        );
    }
    collect_possible_matches_dense_batched(
        tokenizer, root, tokenizer_state_ids, num_internal_tokens,
    )
}

fn collect_possible_matches_dense_serial(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    tokenizer_state_ids: &ManyToOneIdMap,
    num_internal_tokens: u32,
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let mut computer = DensePossibleMatchesComputer::new(tokenizer, num_internal_tokens);
    let mut possible_matches_by_state: BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> = BTreeMap::new();
    let root_key = root as *const VocabPrefixTreeNode as usize;

    for (internal_tsid, representative_state) in tokenizer_state_ids
        .iter_representative_ids()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
    {
        let _ = computer.possible_matches_for_node(root, representative_state);
        let matches_for_state = computer
            .cache
            .remove(&(root_key, representative_state))
            .expect("root possible-match map should be cached");
        let map = match Rc::try_unwrap(matches_for_state) {
            Ok(map) => map,
            Err(shared) => (*shared).clone(),
        };
        possible_matches_by_state.insert(internal_tsid, map);
    }

    (possible_matches_by_state, computer.profile())
}

fn collect_possible_matches_dense_batched(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    tokenizer_state_ids: &ManyToOneIdMap,
    num_internal_tokens: u32,
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let num_words = (num_internal_tokens as usize + 63) / 64;

    // Pre-compute flat transitions for all DFA states.
    let flat_transitions: Vec<[u32; 256]> = (0..tokenizer.num_states() as usize)
        .map(|state_idx| {
            let dfa_state = &tokenizer.dfa.states()[state_idx];
            let mut flat = [u32::MAX; 256];
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            flat
        })
        .collect();

    // Pre-compute self-loop bytes for all DFA states.
    let self_loop_bytes: Vec<U8Set> = (0..tokenizer.num_states() as usize)
        .map(|state_idx| {
            let dfa_state = &tokenizer.dfa.states()[state_idx];
            let mut bytes = U8Set::empty();
            for (byte, &target) in dfa_state.transitions.iter() {
                if target == state_idx as u32 {
                    bytes.insert(byte);
                }
            }
            bytes
        })
        .collect();

    // Pre-compute reachable bitmaps for all trie nodes.
    let mut reachable_bitmaps: FxHashMap<usize, Vec<u64>> = FxHashMap::default();
    precompute_reachable_bitmaps(root, num_words, &mut reachable_bitmaps);

    // Gather entries: (internal_tsid, representative_state)
    let entries: Vec<(u32, u32)> = tokenizer_state_ids
        .iter_representative_ids()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();

    let n = entries.len();
    let mut results: Vec<DensePossibleMatchMap> = Vec::with_capacity(n);
    for _ in 0..n {
        results.push(DensePossibleMatchMap::default());
    }

    // Build initial live set: (entry_index, dfa_state)
    let live: Vec<(usize, u32)> = entries
        .iter()
        .enumerate()
        .map(|(i, &(_, rep))| (i, rep))
        .collect();

    batched_walk_node(
        root,
        &live,
        &mut results,
        &flat_transitions,
        &self_loop_bytes,
        tokenizer,
        num_words,
        &reachable_bitmaps,
    );

    let possible_matches_by_state: BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> = results
        .into_iter()
        .enumerate()
        .map(|(i, map)| (entries[i].0, map))
        .collect();

    (possible_matches_by_state, PossibleMatchesProfile::default())
}

fn precompute_reachable_bitmaps(
    node: &VocabPrefixTreeNode,
    num_words: usize,
    cache: &mut FxHashMap<usize, Vec<u64>>,
) {
    let key = node as *const VocabPrefixTreeNode as usize;
    if cache.contains_key(&key) {
        return;
    }
    cache.insert(key, reachable_bitmap(node, num_words));
    for (_, child) in node.iter_children() {
        precompute_reachable_bitmaps(child, num_words, cache);
    }
}

fn batched_walk_node(
    node: &VocabPrefixTreeNode,
    live: &[(usize, u32)],
    results: &mut [DensePossibleMatchMap],
    flat_trans: &[[u32; 256]],
    self_loop: &[U8Set],
    tokenizer: &Tokenizer,
    num_words: usize,
    reachable_bitmaps: &FxHashMap<usize, Vec<u64>>,
) {
    if live.is_empty() {
        return;
    }

    // Handle token at this node
    if node.has_token() {
        let token_id = node.token_id() as u32;
        for &(idx, state) in live {
            for terminal in tokenizer.matched_terminals_iter(state) {
                let entry = results[idx]
                    .entry(terminal)
                    .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
            }
        }
    }

    // Process each child edge
    for (segment, child) in node.iter_children() {
        let child_key = child as *const VocabPrefixTreeNode as usize;
        let reachable = &reachable_bitmaps[&child_key];

        let subtree_bytes = U8Set::from_words(*child.subtree_bytes());

        let mut child_live: Vec<(usize, u32)> = Vec::new();

        for &(idx, state) in live {
            let mut s = state;
            let mut dead = false;

            for &byte in segment {
                let next = flat_trans[s as usize][byte as usize];
                if next == u32::MAX {
                    dead = true;
                    break;
                }
                s = next;
                for terminal in tokenizer.matched_terminals_iter(s) {
                    let entry = results[idx]
                        .entry(terminal)
                        .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                    merge_bitmaps(entry, reachable);
                }
            }

            if dead {
                continue;
            }
            if tokenizer.is_end(s) {
                continue;
            }
            if subtree_bytes.is_subset(&self_loop[s as usize]) {
                continue;
            }
            child_live.push((idx, s));
        }

        if !child_live.is_empty() {
            batched_walk_node(
                child,
                &child_live,
                results,
                flat_trans,
                self_loop,
                tokenizer,
                num_words,
                reachable_bitmaps,
            );
        }
    }
}

fn collect_possible_matches_dense_parallel(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    tokenizer_state_ids: &ManyToOneIdMap,
    num_internal_tokens: u32,
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    use dashmap::DashMap;
    use rayon::prelude::*;

    let num_words = (num_internal_tokens as usize + 63) / 64;

    // Pre-compute flat transitions for all DFA states (shared read-only).
    let flat_transitions: Vec<[u32; 256]> = (0..tokenizer.num_states() as usize)
        .map(|state_idx| {
            let dfa_state = &tokenizer.dfa.states()[state_idx];
            let mut flat = [u32::MAX; 256];
            for (b, &target) in dfa_state.transitions.iter() {
                flat[b as usize] = target;
            }
            flat
        })
        .collect();

    // Pre-compute self-loop bytes for all DFA states (shared read-only).
    let self_loop_bytes: Vec<U8Set> = (0..tokenizer.num_states() as usize)
        .map(|state_idx| {
            let dfa_state = &tokenizer.dfa.states()[state_idx];
            let mut bytes = U8Set::empty();
            for (byte, &target) in dfa_state.transitions.iter() {
                if target == state_idx as u32 {
                    bytes.insert(byte);
                }
            }
            bytes
        })
        .collect();

    // Shared concurrent caches.
    let cache: DashMap<(usize, u32), Arc<DensePossibleMatchMap>> = DashMap::new();
    let reachable_cache: DashMap<usize, Arc<Vec<u64>>> = DashMap::new();

    let entries: Vec<(u32, u32)> = tokenizer_state_ids
        .iter_representative_ids()
        .enumerate()
        .map(|(i, s)| (i as u32, s))
        .collect();

    let results: Vec<(u32, DensePossibleMatchMap)> = entries
        .par_iter()
        .map(|&(internal_tsid, representative_state)| {
            let result = possible_matches_for_node_concurrent(
                tokenizer, root, representative_state,
                num_words, &flat_transitions, &self_loop_bytes,
                &cache, &reachable_cache,
            );
            let map = Arc::try_unwrap(result).unwrap_or_else(|arc| (*arc).clone());
            (internal_tsid, map)
        })
        .collect();

    let possible_matches_by_state: BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> =
        results.into_iter().collect();

    (possible_matches_by_state, PossibleMatchesProfile::default())
}

/// Recursive possible-matches computation using a shared DashMap cache.
/// Multiple threads may call this concurrently; duplicate computation on
/// cache misses is benign since results are deterministic.
fn possible_matches_for_node_concurrent(
    tokenizer: &Tokenizer,
    node: &VocabPrefixTreeNode,
    tokenizer_state: u32,
    num_words: usize,
    flat_transitions: &[[u32; 256]],
    self_loop_bytes: &[U8Set],
    cache: &dashmap::DashMap<(usize, u32), Arc<DensePossibleMatchMap>>,
    reachable_cache: &dashmap::DashMap<usize, Arc<Vec<u64>>>,
) -> Arc<DensePossibleMatchMap> {
    let cache_key = (node as *const VocabPrefixTreeNode as usize, tokenizer_state);

    if let Some(cached) = cache.get(&cache_key) {
        return Arc::clone(cached.value());
    }

    let mut result = DensePossibleMatchMap::default();

    if node.has_token() {
        let token_id = node.token_id() as u32;
        for terminal in tokenizer.matched_terminals_iter(tokenizer_state) {
            let entry = result.entry(terminal).or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
            entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
        }
    }

    for (segment_bytes, child) in node.iter_children() {
        let mut current_state = tokenizer_state;
        let mut segment_blocked = false;

        let reachable_key = child as *const VocabPrefixTreeNode as usize;
        let reachable = if let Some(r) = reachable_cache.get(&reachable_key) {
            Arc::clone(r.value())
        } else {
            let r = Arc::new(reachable_bitmap(child, num_words));
            reachable_cache.insert(reachable_key, Arc::clone(&r));
            r
        };

        for &byte in segment_bytes {
            let next = flat_transitions[current_state as usize][byte as usize];
            if next == u32::MAX {
                segment_blocked = true;
                break;
            }
            current_state = next;
            for terminal in tokenizer.matched_terminals_iter(current_state) {
                let existing = result.entry(terminal).or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                merge_bitmaps(existing, reachable.as_ref());
            }
        }

        if segment_blocked {
            continue;
        }
        if !tokenizer.is_end(current_state) {
            if U8Set::from_words(*child.subtree_bytes()).is_subset(&self_loop_bytes[current_state as usize]) {
                continue;
            }
            let child_matches = possible_matches_for_node_concurrent(
                tokenizer, child, current_state,
                num_words, flat_transitions, self_loop_bytes,
                cache, reachable_cache,
            );
            merge_dense_maps(&mut result, child_matches.as_ref(), num_words);
        }
    }

    let result = Arc::new(result);
    cache.insert(cache_key, Arc::clone(&result));
    result
}

pub(crate) fn build_possible_matches_by_state(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> PossibleMatchesByState {
    let token_entries = owned_token_entries(token_bytes);
    build_possible_matches_from_token_entries(tokenizer, &token_entries)
}

pub(crate) fn build_possible_matches_from_token_bytes(
    tokenizer: &Tokenizer,
    token_bytes: &BTreeMap<u32, Vec<u8>>,
) -> PossibleMatchesByState {
    let token_entries = owned_token_entries(token_bytes);
    build_possible_matches_from_token_entries(tokenizer, &token_entries)
}

pub(crate) fn build_possible_matches_from_token_entries(
    tokenizer: &Tokenizer,
    token_entries: &[(u32, Vec<u8>)],
) -> PossibleMatchesByState {
    build_possible_matches_from_owned_token_entries(tokenizer, clone_token_entries(token_entries))
}

pub(crate) fn build_possible_matches_from_owned_token_entries(
    tokenizer: &Tokenizer,
    token_entries: Vec<(u32, Vec<u8>)>,
) -> PossibleMatchesByState {
    let trie = VocabPrefixTree::build_owned(
        token_entries
            .into_iter()
            .map(|(token_id, bytes)| (token_id as usize, bytes))
            .collect(),
    );

    let mut computer = PossibleMatchesComputer::new(tokenizer);
    collect_possible_matches_by_state(tokenizer, &trie.root, &mut computer)
}

pub(crate) fn collect_possible_matches_by_state(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    computer: &mut PossibleMatchesComputer<'_>,
) -> PossibleMatchesByState {
    collect_possible_matches_by_keys(
        tokenizer,
        root,
        computer,
        (0..tokenizer.num_states()).map(|state| (state, state)),
        tokenizer.num_states(),
    )
}

pub(crate) fn collect_possible_matches_by_internal_tsid(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    computer: &mut PossibleMatchesComputer<'_>,
    tokenizer_state_ids: &ManyToOneIdMap,
) -> PossibleMatchesByState {
    collect_possible_matches_by_keys(
        tokenizer,
        root,
        computer,
        tokenizer_state_ids
            .iter_representative_ids()
            .enumerate()
            .map(|(internal_tsid, representative_state)| (internal_tsid as u32, representative_state)),
        tokenizer_state_ids.num_internal_ids(),
    )
}

fn collect_possible_matches_by_keys(
    _tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    computer: &mut PossibleMatchesComputer<'_>,
    keyed_states: impl Iterator<Item = (u32, u32)>,
    total_keys: u32,
) -> PossibleMatchesByState {
    let mut possible_matches_by_state = BTreeMap::new();
    let root_key = root as *const VocabPrefixTreeNode as usize;
    let debug_profile = debug_profile_enabled();
    let summary_profile = computer.summary_profile_enabled;
    let started_at = Instant::now();
    for (index, (result_state_id, representative_state)) in keyed_states.enumerate() {
        let cache_key = (root_key, representative_state);
        let root_compute_started_at = summary_profile.then(Instant::now);
        let _ = computer.possible_matches_for_node(root, representative_state);
        if let Some(started_at) = root_compute_started_at {
            computer.profile.root_compute_ms += elapsed_ms(started_at);
        }
        let materialize_started_at = summary_profile.then(Instant::now);
        let matches_for_state = computer
            .cache
            .remove(&cache_key)
            .expect("root possible-match map should be cached");
        possible_matches_by_state.insert(result_state_id, ordered_possible_matches(matches_for_state));
        if let Some(started_at) = materialize_started_at {
            computer.profile.materialize_output_ms += elapsed_ms(started_at);
        }

        let states_done = index as u32 + 1;
        if debug_profile && ((states_done % 100_000 == 0) || states_done == total_keys) {
            let profile = computer.profile();
            eprintln!(
                "[glrmask/debug][possible_matches] states_done={} total_states={} elapsed_ms={:.3} cache_entries={} reachable_cache_entries={} cache_hits={} cache_misses={} child_segments={} byte_steps={} recursive_descents={} self_loop_subtrees_skipped={} terminal_insertions={}",
                states_done,
                total_keys,
                started_at.elapsed().as_secs_f64() * 1000.0,
                profile.cache_entries,
                profile.reachable_cache_entries,
                profile.cache_hits,
                profile.cache_misses,
                profile.child_segments_visited,
                profile.byte_steps,
                profile.recursive_descents,
                profile.self_loop_subtrees_skipped,
                profile.terminal_insertions,
            );
        }
    }

    possible_matches_by_state
}

pub(crate) fn permute_possible_matches_in_place(
    possible_matches_by_state: &mut PossibleMatchesByState,
    token_perm: &[u32],
) {
    for matches_by_terminal in possible_matches_by_state.values_mut() {
        for token_ids in matches_by_terminal.values_mut() {
            let mut mapped: Vec<u32> = token_ids
                .iter()
                .filter_map(|token_id| token_perm.get(token_id as usize).copied())
                .collect();
            mapped.sort_unstable();
            mapped.dedup();
            *token_ids = RangeSetBlaze::from_iter(mapped.into_iter().map(|token_id| token_id..=token_id));
        }
    }
}

pub(crate) fn permute_possible_match_state_ids_in_place(
    possible_matches_by_state: &mut PossibleMatchesByState,
    state_perm: &[u32],
) {
    let mut remapped = BTreeMap::new();

    for (&state_id, matches_by_terminal) in possible_matches_by_state.iter() {
        let Some(&new_state_id) = state_perm.get(state_id as usize) else {
            continue;
        };

        let target = remapped
            .entry(new_state_id)
            .or_insert_with(BTreeMap::<TerminalID, RangeSetBlaze<u32>>::new);
        for (&terminal_id, token_ids) in matches_by_terminal {
            let existing = target.entry(terminal_id).or_default();
            *existing |= token_ids;
        }
    }

    *possible_matches_by_state = remapped;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{byte, bytes, star};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use range_set_blaze::RangeSetBlaze;

    #[test]
    fn test_possible_matches_supports_distinct_bytes_for_same_internal_token() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b")]);
        let token_entries = vec![(0u32, b"a".to_vec()), (0u32, b"b".to_vec())];

        let possible_matches =
            build_possible_matches_from_token_entries(&tokenizer, &token_entries);
        let start_matches = possible_matches
            .get(&tokenizer.initial_state())
            .expect("start state should have possible matches");

        assert_eq!(
            start_matches.get(&0),
            Some(&RangeSetBlaze::from_iter([0u32..=0u32]))
        );
        assert_eq!(
            start_matches.get(&1),
            Some(&RangeSetBlaze::from_iter([0u32..=0u32]))
        );
    }

    #[test]
    fn test_possible_matches_self_loop_subtree_skip_preserves_descendants() {
        let tokenizer = build_tokenizer_from_exprs(&[star(byte(b'a'))]);
        let token_entries = vec![
            (0u32, b"a".to_vec()),
            (1u32, b"aa".to_vec()),
            (2u32, b"aaa".to_vec()),
        ];

        let trie = VocabPrefixTree::build_owned(
            token_entries
                .into_iter()
                .map(|(token_id, bytes)| (token_id as usize, bytes))
                .collect(),
        );
        let mut computer = PossibleMatchesComputer::new(&tokenizer);
        let possible_matches = collect_possible_matches_by_state(&tokenizer, &trie.root, &mut computer);

        let start_matches = possible_matches
            .get(&tokenizer.initial_state())
            .expect("start state should have possible matches");

        assert_eq!(
            start_matches.get(&0),
            Some(&RangeSetBlaze::from_iter([0u32..=2u32]))
        );
        assert!(computer.profile().self_loop_subtrees_skipped > 0);
    }
}
