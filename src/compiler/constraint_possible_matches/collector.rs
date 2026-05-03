//! Constraint-specific possible-match dense collector.
//!
//! Migrated from compiler::possible_matches so that the Constraint build path
//! does not depend on possible_matches.rs for its stored possible_matches.
//! Future constraint-specific optimizations belong here.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::pm_profile::{PossibleMatchesProfile, elapsed_ms, merge_possible_matches_profile, profile_summary_enabled};
pub(crate) use crate::compiler::pm_profile::emit_possible_matches_profile_summary;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::grammar::flat::TerminalID;

// ===========================================================================

// ---------------------------------------------------------------------------
// Dense bitmap variant of PossibleMatchesComputer
// ---------------------------------------------------------------------------

pub(crate) type DensePossibleMatchMap = BTreeMap<TerminalID, Box<[u64]>>;

pub(crate) struct DenseTrieClassBuildResult {
    pub(crate) state_classes: Vec<u32>,
    pub(crate) class_maps: Vec<Arc<DensePossibleMatchMap>>,
}

impl DenseTrieClassBuildResult {
    pub(crate) fn expand_to_states(&self, entries: &[u32]) -> BTreeMap<u32, DensePossibleMatchMap> {
        entries
            .iter()
            .copied()
            .map(|state| {
                let class_id = self.state_classes[state as usize];
                let map = self.class_maps[class_id as usize].as_ref().clone();
                (state, map)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
struct TrieMapBuildSegmentOutcome {
    terminals_id: u32,
    end_state: Option<u32>,
}

#[derive(Default)]
struct TrieMapBuildTerminalSetInterner {
    ids: FxHashMap<Vec<TerminalID>, u32>,
    sets: Vec<Vec<TerminalID>>,
}

impl TrieMapBuildTerminalSetInterner {
    fn intern_slice(&mut self, terminals: &[TerminalID]) -> u32 {
        if let Some(&id) = self.ids.get(terminals) {
            return id;
        }

        let id = self.sets.len() as u32;
        let owned = terminals.to_vec();
        self.ids.insert(owned.clone(), id);
        self.sets.push(owned);
        id
    }

    fn intern_vec(&mut self, terminals: Vec<TerminalID>) -> u32 {
        if let Some(&id) = self.ids.get(&terminals) {
            return id;
        }

        let id = self.sets.len() as u32;
        self.ids.insert(terminals.clone(), id);
        self.sets.push(terminals);
        id
    }

    fn get(&self, id: u32) -> &[TerminalID] {
        &self.sets[id as usize]
    }
}

struct TrieMapBuildNodeClasses {
    classes: Vec<u32>,
    class_maps: Vec<Arc<DensePossibleMatchMap>>,
}

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

fn reachable_sparse_bitmap(node: &VocabPrefixTreeNode) -> Box<[(u16, u64)]> {
    let mut entries = Vec::new();

    for range in node.reachable_token_ids().ranges() {
        let mut token_id = *range.start() as u32;
        let end = *range.end() as u32;

        while token_id <= end {
            let word = token_id / 64;
            let word_end = ((word + 1) * 64 - 1).min(end);
            let start_bit = token_id % 64;
            let end_bit = word_end % 64;
            let bit_count = end_bit - start_bit + 1;
            let mask = if bit_count == 64 {
                u64::MAX
            } else {
                ((1u64 << bit_count) - 1) << start_bit
            };

            entries.push((word as u16, mask));
            token_id = word_end.saturating_add(1);
        }
    }

    entries.into_boxed_slice()
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

fn merge_dense_map_into_slots(
    slots: &mut [Option<Box<[u64]>>],
    other: &DensePossibleMatchMap,
    num_words: usize,
) {
    for (&terminal, bitmap) in other.iter() {
        let existing = slots[terminal as usize]
            .get_or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
        merge_bitmaps(existing, bitmap);
    }
}

fn reachable_dense_bitmap(node: &VocabPrefixTreeNode, num_words: usize) -> Box<[u64]> {
    let mut words = vec![0u64; num_words];
    for range in node.reachable_token_ids().ranges() {
        let lo = *range.start() as u32;
        let hi = *range.end() as u32;
        for token_id in lo..=hi {
            words[token_id as usize / 64] |= 1u64 << (token_id % 64);
        }
    }
    words.into_boxed_slice()
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

/// STICKY NOTE: DO NOT REMOVE THIS COMMENT.
/// possible_matches MUST be computed for each ORIGINAL tokenizer state.
/// Do NOT collapse this to an internal TSID, representative state, or
/// tokenizer-state equivalence class, even if that looks like an easy
/// optimization. This exact mistake has recurred and it silently changes
/// semantics by merging distinct tokenizer futures.
///
/// If someone believes per-class possible_matches is safe, they must first
/// prove semantic equivalence for all original states in the class and add
/// regression coverage for divergent-state counterexamples before touching
/// this again. Until then: keep this per-original-state and keep this note.
///
/// Collect possible_matches using dense bitmap computation internally,
/// returning dense bitmaps directly (no RangeSetBlaze conversion).
pub(crate) fn collect_possible_matches_by_original_tsid_dense(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let entries: Vec<u32> = (0..tokenizer.num_states()).collect();
    collect_possible_matches_by_selected_original_tsid_dense(
        tokenizer,
        root,
        num_internal_tokens,
        &entries,
    )
}

pub(crate) fn collect_possible_matches_by_selected_original_tsid_dense(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let trie_class_build = std::env::var("GLRMASK_PM_TRIE_CLASS_BUILD")
        .map_or(false, |value| value == "1");
    if trie_class_build {
        return collect_possible_matches_dense_trie_class_build(
            tokenizer,
            root,
            num_internal_tokens,
            entries,
        );
    }

    let state_chunk_parallel = std::env::var("GLRMASK_PM_STATE_CHUNK_PARALLEL")
        .map_or(false, |value| value == "1");
    if state_chunk_parallel && entries.len() >= 2048 {
        return collect_possible_matches_dense_chunk_parallel(
            tokenizer,
            root,
            num_internal_tokens,
            entries,
        );
    }

    let force_serial = std::env::var("GLRMASK_PM_FORCE_SERIAL")
        .map_or(false, |value| value == "1");
    // For small workloads, the serial path with FxHashMap cache is faster.
    // For medium-to-large workloads, the batched trie walk eliminates cache
    // overhead even before we reach the old 5k-state cutoff.
    if force_serial || entries.len() < 2048 {
        return collect_possible_matches_dense_serial(
            tokenizer,
            root,
            num_internal_tokens,
            entries,
        );
    }
    collect_possible_matches_dense_batched(
        tokenizer,
        root,
        num_internal_tokens,
        entries,
    )
}

pub(crate) fn collect_possible_matches_dense_trie_class_build(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let (result, profile) = collect_possible_matches_dense_trie_class_build_with_classes(
        tokenizer,
        root,
        num_internal_tokens,
        entries,
    );
    (result.expand_to_states(entries), profile)
}

pub(crate) fn collect_possible_matches_dense_trie_class_build_with_classes(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (DenseTrieClassBuildResult, PossibleMatchesProfile) {
    if tokenizer.num_terminals <= 64 {
        if profile_summary_enabled() {
            eprintln!(
                "[glrmask/profile][trie_build_terminal_mask] mode=u64 terminals={}",
                tokenizer.num_terminals,
            );
        }
        collect_possible_matches_dense_trie_class_build_with_classes_u64(
            tokenizer, root, num_internal_tokens, entries,
        )
    } else {
        collect_possible_matches_dense_trie_class_build_with_classes_interned(
            tokenizer, root, num_internal_tokens, entries,
        )
    }
}

fn collect_possible_matches_dense_trie_class_build_with_classes_interned(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (DenseTrieClassBuildResult, PossibleMatchesProfile) {
    let num_words = (num_internal_tokens as usize + 63) / 64;
    let matched_terminals: Vec<Box<[TerminalID]>> = (0..tokenizer.num_states())
        .map(|state| tokenizer.matched_terminals_iter(state).collect::<Vec<_>>().into_boxed_slice())
        .collect();
    let is_end: Vec<bool> = (0..tokenizer.num_states())
        .map(|state| tokenizer.is_end(state))
        .collect();
    let mut byte_transitions = vec![vec![u32::MAX; tokenizer.num_states() as usize]; 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        for (byte, &target) in dfa_state.transitions.iter() {
            byte_transitions[byte as usize][state_idx] = target;
        }
    }
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
    let mut terminal_sets = TrieMapBuildTerminalSetInterner::default();
    let empty_terminals_id = terminal_sets.intern_slice(&[]);
    let node_terminal_ids: Vec<u32> = matched_terminals
        .iter()
        .map(|terminals| terminal_sets.intern_slice(terminals))
        .collect();
    // Segment cache keyed by (hash, length) to avoid cloning Vec<u8> on hit.
    // Collisions almost impossible: we only insert once per unique segment.
    fn segment_key(segment: &[u8]) -> (u64, u8) {
        let mut h: u64 = 0;
        for &b in segment {
            h = h.wrapping_mul(0x517cc1b727220a95).wrapping_add(b as u64);
        }
        (h, segment.len() as u8)
    }

    let mut segment_cache: FxHashMap<(u64, u8), usize> = FxHashMap::default();
    let mut segment_outcome_tables = Vec::<FxHashMap<u32, TrieMapBuildSegmentOutcome>>::new();

    let num_terminals = tokenizer.num_terminals as usize;
    let mut stamp_gen: u32 = 0;
    let mut terminal_stamps: Vec<u32> = vec![0; num_terminals];


    let t_root_start = Instant::now();

    struct TrieBuildTimings {
        segment_table_ms: f64,
        signature_hash_ms: f64,
        map_materialize_ms: f64,
        classes_built: usize,
        child_active_ms: f64,
        recursive_ms: f64,
        reachable_bitmap_ms: f64,
        child_precompute_ms: f64,
    }
    let mut timings = TrieBuildTimings {
        segment_table_ms: 0.0,
        signature_hash_ms: 0.0,
        map_materialize_ms: 0.0,
        classes_built: 0,
        child_active_ms: 0.0,
        recursive_ms: 0.0,
        reachable_bitmap_ms: 0.0,
        child_precompute_ms: 0.0,
    };

    /// Fill missing segment-outcome entries for the given `needed_states`
    /// into `table` by walking the DFA `segment` from each missing state.
    /// Uses a stamp-based dedup array for O(1) terminal dedup.
    ///
    /// For single-byte segments, uses a fast path that directly looks up
    /// `byte_transitions` and `node_terminal_ids` without touching the
    /// hash-table segment cache or the stamp-based terminal collector.
    fn segment_outcomes_for_states(
        table: &mut FxHashMap<u32, TrieMapBuildSegmentOutcome>,
        needed_states: &[u32],
        segment: &[u8],
        matched_terminals: &[Box<[TerminalID]>],
        byte_transitions: &[Vec<u32>],
        terminal_sets: &mut TrieMapBuildTerminalSetInterner,
        empty_terminals_id: u32,
        terminal_stamps: &mut [u32],
        stamp_gen: &mut u32,
        timings: &mut TrieBuildTimings,
        node_terminal_ids: &[u32],
    ) -> Vec<TrieMapBuildSegmentOutcome> {
        let t_start = Instant::now();
        let mut outcomes = Vec::with_capacity(needed_states.len());

        if segment.len() == 1 {
            let byte = segment[0] as usize;
            for &start_state in needed_states {
                let next_state = byte_transitions[byte][start_state as usize];
                if next_state == u32::MAX {
                    outcomes.push(TrieMapBuildSegmentOutcome {
                        terminals_id: empty_terminals_id,
                        end_state: None,
                    });
                } else {
                    outcomes.push(TrieMapBuildSegmentOutcome {
                        terminals_id: node_terminal_ids[next_state as usize],
                        end_state: Some(next_state),
                    });
                }
            }
            timings.segment_table_ms += elapsed_ms(t_start);
            return outcomes;
        }

        for &start_state in needed_states {
            if let Some(&outcome) = table.get(&start_state) {
                outcomes.push(outcome);
                continue;
            }

            let mut current_state = start_state;
            let mut blocked = false;
            *stamp_gen = stamp_gen.wrapping_add(1);
            let current_gen = *stamp_gen;
            let mut term_count = 0usize;

            for &byte in segment {
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX {
                    blocked = true;
                    break;
                }
                current_state = next_state;
                for &terminal in matched_terminals[current_state as usize].iter() {
                    let t = terminal as usize;
                    if terminal_stamps[t] != current_gen {
                        terminal_stamps[t] = current_gen;
                        term_count += 1;
                    }
                }
            }

            let terminals_id = if term_count == 0 {
                empty_terminals_id
            } else {
                let mut list = Vec::with_capacity(term_count);
                for (t_idx, &stamp) in terminal_stamps.iter().enumerate() {
                    if stamp == current_gen {
                        list.push(t_idx as TerminalID);
                    }
                }
                terminal_sets.intern_vec(list)
            };

            let outcome = TrieMapBuildSegmentOutcome {
                terminals_id,
                end_state: (!blocked).then_some(current_state),
            };
            table.insert(start_state, outcome);
            outcomes.push(outcome);
        }

        timings.segment_table_ms += elapsed_ms(t_start);
        outcomes
    }

    #[inline]
    fn mix_signature_word(hash: u64, word: u32) -> u64 {
        hash
            .wrapping_mul(0x517cc1b727220a95)
            .wrapping_add((word as u64).wrapping_add(0x9e3779b97f4a7c15))
    }

    struct SignatureEntry {
        state_pos: usize,
        class_id: u32,
    }

    fn build_node(
        node: &VocabPrefixTreeNode,
        tokenizer: &Tokenizer,
        active_states: &[u32],
        matched_terminals: &[Box<[TerminalID]>],
        node_terminal_ids: &[u32],
        empty_terminals_id: u32,
        is_end: &[bool],
        byte_transitions: &[Vec<u32>],
        self_loop_bytes: &[U8Set],
        terminal_sets: &mut TrieMapBuildTerminalSetInterner,
        segment_cache: &mut FxHashMap<(u64, u8), usize>,
        segment_outcome_tables: &mut Vec<FxHashMap<u32, TrieMapBuildSegmentOutcome>>,
        num_words: usize,
        timings: &mut TrieBuildTimings,
        stamp_gen: &mut u32,
        terminal_stamps: &mut [u32],
    ) -> TrieMapBuildNodeClasses {
        struct ChildBuildData {
            outcomes: Vec<TrieMapBuildSegmentOutcome>,
            child_class_ids: Vec<u32>,
            reachable: Box<[u64]>,
            result: TrieMapBuildNodeClasses,
        }

        let mut child_data = Vec::new();
        for (segment, child) in node.iter_children() {
            let segment_table_idx = if let Some(&table_idx) = segment_cache.get(&segment_key(segment)) {
                table_idx
            } else {
                let idx = segment_outcome_tables.len();
                segment_outcome_tables.push(FxHashMap::default());
                segment_cache.insert(segment_key(segment), idx);
                idx
            };

            // Compute segment outcomes for all active states with one
            // hash lookup per state (cache hit returns immediately, miss
            // walks the DFA and inserts both into the cache and the result).
            let outcomes = segment_outcomes_for_states(
                &mut segment_outcome_tables[segment_table_idx],
                active_states,
                segment,
                matched_terminals,
                byte_transitions,
                terminal_sets,
                empty_terminals_id,
                terminal_stamps,
                stamp_gen,
                timings,
                node_terminal_ids,
            );

            let child_active_started_at = Instant::now();
            let subtree_bytes = U8Set::from_words(*child.subtree_bytes());
            let mut descend_end_states = Vec::with_capacity(active_states.len());
            let mut child_active_states = Vec::new();
            for segment_outcome in outcomes.iter() {
                let descend_end_state = if let Some(end_state) = segment_outcome.end_state {
                    if !is_end[end_state as usize]
                        && !subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                    {
                        child_active_states.push(end_state);
                        end_state
                    } else {
                        u32::MAX
                    }
                } else {
                    u32::MAX
                };
                descend_end_states.push(descend_end_state);
            }
            child_active_states.sort_unstable();
            child_active_states.dedup();
            timings.child_active_ms += elapsed_ms(child_active_started_at);

            let (result, child_class_ids) = if child_active_states.is_empty() {
                (
                    TrieMapBuildNodeClasses {
                        classes: Vec::new(),
                        class_maps: Vec::new(),
                    },
                    vec![u32::MAX; descend_end_states.len()],
                )
            } else {
                let recursive_started_at = Instant::now();
                let result = build_node(
                    child,
                    tokenizer,
                    &child_active_states,
                    matched_terminals,
                    node_terminal_ids,
                    empty_terminals_id,
                    is_end,
                    byte_transitions,
                    self_loop_bytes,
                    terminal_sets,
                    segment_cache,
                    segment_outcome_tables,
                    num_words,
                    timings,
                    stamp_gen,
                    terminal_stamps,
                );
                timings.recursive_ms += elapsed_ms(recursive_started_at);

                // Build child_class_ids from the cached descend_end_states.
                let child_precompute_started_at = Instant::now();
                let child_class_ids: Vec<u32> = descend_end_states
                    .iter()
                    .map(|&end_state| {
                        if end_state == u32::MAX {
                            u32::MAX
                        } else {
                            result.classes[end_state as usize]
                        }
                    })
                    .collect();
                timings.child_precompute_ms += elapsed_ms(child_precompute_started_at);

                (result, child_class_ids)
            };

            let reachable_started_at = Instant::now();
            let reachable = reachable_dense_bitmap(child, num_words);
            timings.reachable_bitmap_ms += elapsed_ms(reachable_started_at);

            child_data.push(ChildBuildData {
                outcomes,
                child_class_ids,
                reachable,
                result,
            });
        }

        let t_sig = Instant::now();
        let mut representative_states = Vec::new();
        let mut representative_state_positions = Vec::new();
        let mut classes = vec![u32::MAX; tokenizer.num_states() as usize];

        match child_data.len() {
            0 => {
                if node.has_token() {
                    let mut class_by_terminal_id: FxHashMap<u32, u32> = FxHashMap::default();
                    for (state_pos, &state) in active_states.iter().enumerate() {
                        let node_terminals_id = node_terminal_ids[state as usize];
                        let class_id = *class_by_terminal_id.entry(node_terminals_id).or_insert_with(|| {
                            let class_id = representative_states.len() as u32;
                            representative_states.push(state);
                            representative_state_positions.push(state_pos);
                            class_id
                        });
                        classes[state as usize] = class_id;
                    }
                } else if let Some(&state) = active_states.first() {
                    representative_states.push(state);
                    representative_state_positions.push(0);
                    for &state in active_states {
                        classes[state as usize] = 0;
                    }
                }
            }
            1 => {
                let child = &child_data[0];
                let mut class_by_signature: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
                for (state_pos, &state) in active_states.iter().enumerate() {
                    let node_terminals_id = if node.has_token() {
                        node_terminal_ids[state as usize]
                    } else {
                        empty_terminals_id
                    };
                    let segment_outcome = child.outcomes[state_pos];
                    let child_class_id = child.child_class_ids[state_pos];
                    let signature = (node_terminals_id, segment_outcome.terminals_id, child_class_id);
                    let class_id = *class_by_signature.entry(signature).or_insert_with(|| {
                        let class_id = representative_states.len() as u32;
                        representative_states.push(state);
                        representative_state_positions.push(state_pos);
                        class_id
                    });
                    classes[state as usize] = class_id;
                }
            }
            _ => {
                let mut signature_buckets: FxHashMap<u64, Vec<SignatureEntry>> = FxHashMap::default();
                let mut next_class_id: u32 = 0;

                for (state_pos, &state) in active_states.iter().enumerate() {
                    let node_terminals_id = if node.has_token() {
                        node_terminal_ids[state as usize]
                    } else {
                        empty_terminals_id
                    };

                    let mut hash: u64 = mix_signature_word(0, node_terminals_id);
                    for child in child_data.iter() {
                        let segment_outcome = child.outcomes[state_pos];
                        let child_class_id = child.child_class_ids[state_pos];
                        hash = mix_signature_word(hash, segment_outcome.terminals_id);
                        hash = mix_signature_word(hash, child_class_id);
                    }

                    let bucket = signature_buckets.entry(hash).or_default();
                    let mut found = false;
                    for entry in bucket.iter() {
                        let rep_pos = entry.state_pos;
                        let rep_state = active_states[rep_pos];
                        let rep_node_terminals_id = if node.has_token() {
                            node_terminal_ids[rep_state as usize]
                        } else {
                            empty_terminals_id
                        };
                        if rep_node_terminals_id != node_terminals_id {
                            continue;
                        }
                        let same_children = child_data.iter().all(|child| {
                            child.outcomes[rep_pos].terminals_id == child.outcomes[state_pos].terminals_id
                                && child.child_class_ids[rep_pos] == child.child_class_ids[state_pos]
                        });
                        if same_children {
                            classes[state as usize] = entry.class_id;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let class_id = next_class_id;
                        next_class_id += 1;
                        classes[state as usize] = class_id;
                        representative_states.push(state);
                        representative_state_positions.push(state_pos);
                        bucket.push(SignatureEntry {
                            state_pos,
                            class_id,
                        });
                    }
                }
            }
        }
        timings.signature_hash_ms += elapsed_ms(t_sig);
        timings.classes_built += representative_states.len();

        let t_map = Instant::now();
        let mut class_maps = Vec::with_capacity(representative_states.len());
        for (&state, &state_pos) in representative_states.iter().zip(representative_state_positions.iter()) {
            let mut result = DensePossibleMatchMap::default();

            if node.has_token() {
                let token_id = node.token_id() as u32;
                for &terminal in terminal_sets.get(node_terminal_ids[state as usize]) {
                    let entry = result
                        .entry(terminal)
                        .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                    entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
                }
            }

            for child in child_data.iter() {
                let segment_outcome = child.outcomes[state_pos];
                for &terminal in terminal_sets.get(segment_outcome.terminals_id) {
                    let entry = result
                        .entry(terminal)
                        .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                    merge_bitmaps(entry, &child.reachable);
                }

                let child_class_id = child.child_class_ids[state_pos];
                if child_class_id != u32::MAX {
                    merge_dense_maps(
                        &mut result,
                        child.result.class_maps[child_class_id as usize].as_ref(),
                        num_words,
                    );
                }
            }

            class_maps.push(Arc::new(result));
        }
        timings.map_materialize_ms += elapsed_ms(t_map);

        TrieMapBuildNodeClasses { classes, class_maps }
    }

    let root_result = build_node(
        root,
        tokenizer,
        entries,
        &matched_terminals,
        &node_terminal_ids,
        empty_terminals_id,
        &is_end,
        &byte_transitions,
        &self_loop_bytes,
        &mut terminal_sets,
        &mut segment_cache,
        &mut segment_outcome_tables,
        num_words,
        &mut timings,
        &mut stamp_gen,
        &mut terminal_stamps,
    );
    let root_compute_ms = elapsed_ms(t_root_start);

    if profile_summary_enabled() {
        eprintln!(
            "[glrmask/profile][trie_build_timings] segment_table_ms={:.3} signature_hash_ms={:.3} map_materialize_ms={:.3} child_active_ms={:.3} recursive_ms={:.3} reachable_bitmap_ms={:.3} child_precompute_ms={:.3} classes_built={}",
            timings.segment_table_ms, timings.signature_hash_ms, timings.map_materialize_ms,
            timings.child_active_ms, timings.recursive_ms, timings.reachable_bitmap_ms, timings.child_precompute_ms,
            timings.classes_built,
        );
    }

    let profile = PossibleMatchesProfile {
        cache_entries: root_result.class_maps.len(),
        root_compute_ms,
        ..Default::default()
    };
    (
        DenseTrieClassBuildResult {
            state_classes: root_result.classes,
            class_maps: root_result
                .class_maps
                .iter()
                .cloned()
                .collect(),
        },
        profile,
    )
}

fn collect_possible_matches_dense_trie_class_build_with_classes_u64(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (DenseTrieClassBuildResult, PossibleMatchesProfile) {
    let num_words = (num_internal_tokens as usize + 63) / 64;
    let state_terminal_masks: Vec<u64> = (0..tokenizer.num_states())
        .map(|state| {
            let mut mask = 0u64;
            for terminal in tokenizer.matched_terminals_iter(state) {
                debug_assert!((terminal as usize) < 64);
                mask |= 1u64 << terminal;
            }
            mask
        })
        .collect();
    let is_end: Vec<bool> = (0..tokenizer.num_states())
        .map(|state| tokenizer.is_end(state))
        .collect();
    let mut byte_transitions = vec![vec![u32::MAX; tokenizer.num_states() as usize]; 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        for (byte, &target) in dfa_state.transitions.iter() {
            byte_transitions[byte as usize][state_idx] = target;
        }
    }
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

    let t_root_start = Instant::now();

    struct TrieBuildTimings {
        segment_table_ms: f64,
        signature_hash_ms: f64,
        map_materialize_ms: f64,
        classes_built: usize,
        child_active_ms: f64,
        recursive_ms: f64,
        reachable_bitmap_ms: f64,
        child_precompute_ms: f64,
    }
    let mut timings = TrieBuildTimings {
        segment_table_ms: 0.0,
        signature_hash_ms: 0.0,
        map_materialize_ms: 0.0,
        classes_built: 0,
        child_active_ms: 0.0,
        recursive_ms: 0.0,
        reachable_bitmap_ms: 0.0,
        child_precompute_ms: 0.0,
    };

    impl TrieBuildTimings {
        fn add_assign(&mut self, other: Self) {
            self.segment_table_ms += other.segment_table_ms;
            self.signature_hash_ms += other.signature_hash_ms;
            self.map_materialize_ms += other.map_materialize_ms;
            self.classes_built += other.classes_built;
            self.child_active_ms += other.child_active_ms;
            self.recursive_ms += other.recursive_ms;
            self.reachable_bitmap_ms += other.reachable_bitmap_ms;
            self.child_precompute_ms += other.child_precompute_ms;
        }
    }

    #[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
    struct TrieMapBuildSegmentOutcomeMask {
        terminals_mask: u64,
        end_state: Option<u32>,
    }

    /// Compute segment outcomes using direct u64 terminal masks.
    /// Single-byte segments look up the precomputed mask directly.
    /// Multi-byte segments OR masks from each intermediate state.
    /// No hash-table cache is needed since OR is cheaper than a lookup.
    fn segment_outcomes_for_states(
        needed_states: &[u32],
        segment: &[u8],
        state_terminal_masks: &[u64],
        byte_transitions: &[Vec<u32>],
        timings: &mut TrieBuildTimings,
    ) -> Vec<TrieMapBuildSegmentOutcomeMask> {
        let t_start = Instant::now();
        let mut outcomes = Vec::with_capacity(needed_states.len());

        if segment.len() == 1 {
            let byte = segment[0] as usize;
            for &start_state in needed_states {
                let next_state = byte_transitions[byte][start_state as usize];
                if next_state == u32::MAX {
                    outcomes.push(TrieMapBuildSegmentOutcomeMask {
                        terminals_mask: 0,
                        end_state: None,
                    });
                } else {
                    outcomes.push(TrieMapBuildSegmentOutcomeMask {
                        terminals_mask: state_terminal_masks[next_state as usize],
                        end_state: Some(next_state),
                    });
                }
            }
            timings.segment_table_ms += elapsed_ms(t_start);
            return outcomes;
        }

        for &start_state in needed_states {
            let mut current_state = start_state;
            let mut blocked = false;
            let mut terminals_mask = 0u64;

            for &byte in segment {
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX {
                    blocked = true;
                    break;
                }
                current_state = next_state;
                terminals_mask |= state_terminal_masks[current_state as usize];
            }

            outcomes.push(TrieMapBuildSegmentOutcomeMask {
                terminals_mask,
                end_state: (!blocked).then_some(current_state),
            });
        }

        timings.segment_table_ms += elapsed_ms(t_start);
        outcomes
    }

    #[inline]
    fn for_each_terminal_mask_bit(mut mask: u64, mut f: impl FnMut(TerminalID)) {
        while mask != 0 {
            let bit = mask.trailing_zeros() as TerminalID;
            f(bit);
            mask &= mask - 1;
        }
    }

    #[inline]
    fn mix_signature_word(hash: u64, word: u64) -> u64 {
        hash
            .wrapping_mul(0x517cc1b727220a95)
            .wrapping_add(word.wrapping_add(0x9e3779b97f4a7c15))
    }

    struct SignatureEntryMask {
        state_pos: usize,
        class_id: u32,
    }

    fn build_node(
        node: &VocabPrefixTreeNode,
        tokenizer: &Tokenizer,
        active_states: &[u32],
        state_terminal_masks: &[u64],
        is_end: &[bool],
        byte_transitions: &[Vec<u32>],
        self_loop_bytes: &[U8Set],
        num_words: usize,
        timings: &mut TrieBuildTimings,
        parallel_depth: u8,
        parallel_min_active: usize,
    ) -> TrieMapBuildNodeClasses {
        struct ChildBuildData {
            outcomes: Vec<TrieMapBuildSegmentOutcomeMask>,
            child_class_ids: Vec<u32>,
            reachable: Box<[u64]>,
            result: TrieMapBuildNodeClasses,
        }

        /// Build ChildBuildData for a single child edge, accumulating into
        /// a local timings struct (safe for parallel use).
        fn build_child_data(
            segment: &[u8],
            child: &VocabPrefixTreeNode,
            tokenizer: &Tokenizer,
            active_states: &[u32],
            state_terminal_masks: &[u64],
            is_end: &[bool],
            byte_transitions: &[Vec<u32>],
            self_loop_bytes: &[U8Set],
            num_words: usize,
            timings: &mut TrieBuildTimings,
            parallel_depth: u8,
            parallel_min_active: usize,
        ) -> ChildBuildData {
            let outcomes = segment_outcomes_for_states(
                active_states,
                segment,
                state_terminal_masks,
                byte_transitions,
                timings,
            );

            let child_active_started_at = Instant::now();
            let subtree_bytes = U8Set::from_words(*child.subtree_bytes());
            let mut descend_end_states = Vec::with_capacity(active_states.len());
            let mut child_active_states = Vec::new();
            for segment_outcome in outcomes.iter() {
                let descend_end_state = if let Some(end_state) = segment_outcome.end_state {
                    if !is_end[end_state as usize]
                        && !subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                    {
                        child_active_states.push(end_state);
                        end_state
                    } else {
                        u32::MAX
                    }
                } else {
                    u32::MAX
                };
                descend_end_states.push(descend_end_state);
            }
            child_active_states.sort_unstable();
            child_active_states.dedup();
            timings.child_active_ms += elapsed_ms(child_active_started_at);

            let (result, child_class_ids) = if child_active_states.is_empty() {
                (
                    TrieMapBuildNodeClasses {
                        classes: Vec::new(),
                        class_maps: Vec::new(),
                    },
                    vec![u32::MAX; descend_end_states.len()],
                )
            } else {
                let recursive_started_at = Instant::now();
                let result = build_node(
                    child,
                    tokenizer,
                    &child_active_states,
                    state_terminal_masks,
                    is_end,
                    byte_transitions,
                    self_loop_bytes,
                    num_words,
                    timings,
                    parallel_depth.saturating_sub(1),
                    parallel_min_active,
                );
                timings.recursive_ms += elapsed_ms(recursive_started_at);

                let child_precompute_started_at = Instant::now();
                let child_class_ids: Vec<u32> = descend_end_states
                    .iter()
                    .map(|&end_state| {
                        if end_state == u32::MAX {
                            u32::MAX
                        } else {
                            result.classes[end_state as usize]
                        }
                    })
                    .collect();
                timings.child_precompute_ms += elapsed_ms(child_precompute_started_at);

                (result, child_class_ids)
            };

            let reachable_started_at = Instant::now();
            let reachable = reachable_dense_bitmap(child, num_words);
            timings.reachable_bitmap_ms += elapsed_ms(reachable_started_at);

            ChildBuildData {
                outcomes,
                child_class_ids,
                reachable,
                result,
            }
        }

        let children: Vec<(&[u8], &VocabPrefixTreeNode)> = node.iter_children().collect();
        let num_children = children.len();
        let mut child_data = Vec::with_capacity(num_children);

        let should_parallelize = parallel_depth > 0
            && num_children >= 4
            && active_states.len() >= parallel_min_active;

        if should_parallelize {
            use rayon::prelude::*;
            let built: Vec<(ChildBuildData, TrieBuildTimings)> = children
                .par_iter()
                .map(|(segment, child)| {
                    let mut local_timings = TrieBuildTimings {
                        segment_table_ms: 0.0,
                        signature_hash_ms: 0.0,
                        map_materialize_ms: 0.0,
                        classes_built: 0,
                        child_active_ms: 0.0,
                        recursive_ms: 0.0,
                        reachable_bitmap_ms: 0.0,
                        child_precompute_ms: 0.0,
                    };
                    let data = build_child_data(
                        segment,
                        child,
                        tokenizer,
                        active_states,
                        state_terminal_masks,
                        is_end,
                        byte_transitions,
                        self_loop_bytes,
                        num_words,
                        &mut local_timings,
                        parallel_depth,
                        parallel_min_active,
                    );
                    (data, local_timings)
                })
                .collect();
            for (data, local_timings) in built {
                timings.add_assign(local_timings);
                child_data.push(data);
            }
        } else {
            for (segment, child) in &children {
                let mut local_timings = TrieBuildTimings {
                    segment_table_ms: 0.0,
                    signature_hash_ms: 0.0,
                    map_materialize_ms: 0.0,
                    classes_built: 0,
                    child_active_ms: 0.0,
                    recursive_ms: 0.0,
                    reachable_bitmap_ms: 0.0,
                    child_precompute_ms: 0.0,
                };
                let data = build_child_data(
                    segment,
                    child,
                    tokenizer,
                    active_states,
                    state_terminal_masks,
                    is_end,
                    byte_transitions,
                    self_loop_bytes,
                    num_words,
                    &mut local_timings,
                    parallel_depth,
                    parallel_min_active,
                );
                timings.add_assign(local_timings);
                child_data.push(data);
            }
        }

        let t_sig = Instant::now();
        let mut representative_states = Vec::new();
        let mut representative_state_positions = Vec::new();
        let mut classes = vec![u32::MAX; tokenizer.num_states() as usize];

        match child_data.len() {
            0 => {
                if node.has_token() {
                    let mut class_by_terminal_mask: FxHashMap<u64, u32> = FxHashMap::default();
                    for (state_pos, &state) in active_states.iter().enumerate() {
                        let node_terminal_mask = state_terminal_masks[state as usize];
                        let class_id = *class_by_terminal_mask.entry(node_terminal_mask).or_insert_with(|| {
                            let class_id = representative_states.len() as u32;
                            representative_states.push(state);
                            representative_state_positions.push(state_pos);
                            class_id
                        });
                        classes[state as usize] = class_id;
                    }
                } else if let Some(&state) = active_states.first() {
                    representative_states.push(state);
                    representative_state_positions.push(0);
                    for &state in active_states {
                        classes[state as usize] = 0;
                    }
                }
            }
            1 => {
                let child = &child_data[0];
                let mut class_by_signature: FxHashMap<(u64, u64, u32), u32> = FxHashMap::default();
                for (state_pos, &state) in active_states.iter().enumerate() {
                    let node_terminal_mask = if node.has_token() {
                        state_terminal_masks[state as usize]
                    } else {
                        0u64
                    };
                    let segment_outcome = child.outcomes[state_pos];
                    let child_class_id = child.child_class_ids[state_pos];
                    let signature = (node_terminal_mask, segment_outcome.terminals_mask, child_class_id);
                    let class_id = *class_by_signature.entry(signature).or_insert_with(|| {
                        let class_id = representative_states.len() as u32;
                        representative_states.push(state);
                        representative_state_positions.push(state_pos);
                        class_id
                    });
                    classes[state as usize] = class_id;
                }
            }
            _ => {
                let mut signature_buckets: FxHashMap<u64, Vec<SignatureEntryMask>> = FxHashMap::default();
                let mut next_class_id: u32 = 0;

                for (state_pos, &state) in active_states.iter().enumerate() {
                    let node_terminal_mask = if node.has_token() {
                        state_terminal_masks[state as usize]
                    } else {
                        0u64
                    };

                    let mut hash: u64 = mix_signature_word(0, node_terminal_mask);
                    for child in child_data.iter() {
                        let segment_outcome = child.outcomes[state_pos];
                        let child_class_id = child.child_class_ids[state_pos];
                        hash = mix_signature_word(hash, segment_outcome.terminals_mask);
                        hash = mix_signature_word(hash, child_class_id as u64);
                    }

                    let bucket = signature_buckets.entry(hash).or_default();
                    let mut found = false;
                    for entry in bucket.iter() {
                        let rep_pos = entry.state_pos;
                        let rep_state = active_states[rep_pos];
                        let rep_node_terminal_mask = if node.has_token() {
                            state_terminal_masks[rep_state as usize]
                        } else {
                            0u64
                        };
                        if rep_node_terminal_mask != node_terminal_mask {
                            continue;
                        }
                        let same_children = child_data.iter().all(|child| {
                            child.outcomes[rep_pos].terminals_mask == child.outcomes[state_pos].terminals_mask
                                && child.child_class_ids[rep_pos] == child.child_class_ids[state_pos]
                        });
                        if same_children {
                            classes[state as usize] = entry.class_id;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let class_id = next_class_id;
                        next_class_id += 1;
                        classes[state as usize] = class_id;
                        representative_states.push(state);
                        representative_state_positions.push(state_pos);
                        bucket.push(SignatureEntryMask {
                            state_pos,
                            class_id,
                        });
                    }
                }
            }
        }
        timings.signature_hash_ms += elapsed_ms(t_sig);
        timings.classes_built += representative_states.len();

        let t_map = Instant::now();
        let mut class_maps = Vec::with_capacity(representative_states.len());
        for (&state, &state_pos) in representative_states.iter().zip(representative_state_positions.iter()) {
            let mut result = DensePossibleMatchMap::default();

            if node.has_token() {
                let token_id = node.token_id() as u32;
                let node_mask = state_terminal_masks[state as usize];
                for_each_terminal_mask_bit(node_mask, |terminal| {
                    let entry = result
                        .entry(terminal)
                        .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                    entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
                });
            }

            for child in child_data.iter() {
                let segment_outcome = child.outcomes[state_pos];
                for_each_terminal_mask_bit(segment_outcome.terminals_mask, |terminal| {
                    let entry = result
                        .entry(terminal)
                        .or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
                    merge_bitmaps(entry, &child.reachable);
                });

                let child_class_id = child.child_class_ids[state_pos];
                if child_class_id != u32::MAX {
                    merge_dense_maps(
                        &mut result,
                        child.result.class_maps[child_class_id as usize].as_ref(),
                        num_words,
                    );
                }
            }

            class_maps.push(Arc::new(result));
        }
        timings.map_materialize_ms += elapsed_ms(t_map);

        TrieMapBuildNodeClasses { classes, class_maps }
    }

    let parallel_depth = std::env::var("GLRMASK_PM_ROOT_PARALLEL_DEPTH")
        .ok()
        .and_then(|value| value.parse::<u8>().ok())
        .unwrap_or(4);

    let parallel_min_active = std::env::var("GLRMASK_PM_PARALLEL_MIN_ACTIVE_STATES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1024);

    if profile_summary_enabled() {
        let num_root_children = root.children().len();
        eprintln!(
            "[glrmask/profile][trie_build_terminal_mask] root_parallel_children={} parallel_depth={} parallel_min_active_states={}",
            num_root_children,
            parallel_depth,
            parallel_min_active,
        );
    }

    let root_result = build_node(
        root,
        tokenizer,
        entries,
        &state_terminal_masks,
        &is_end,
        &byte_transitions,
        &self_loop_bytes,
        num_words,
        &mut timings,
        parallel_depth,
        parallel_min_active,
    );
    let root_compute_ms = elapsed_ms(t_root_start);

    if profile_summary_enabled() {
        eprintln!(
            "[glrmask/profile][trie_build_timings] segment_table_ms={:.3} signature_hash_ms={:.3} map_materialize_ms={:.3} child_active_ms={:.3} recursive_ms={:.3} reachable_bitmap_ms={:.3} child_precompute_ms={:.3} classes_built={}",
            timings.segment_table_ms, timings.signature_hash_ms, timings.map_materialize_ms,
            timings.child_active_ms, timings.recursive_ms, timings.reachable_bitmap_ms, timings.child_precompute_ms,
            timings.classes_built,
        );
    }

    let profile = PossibleMatchesProfile {
        cache_entries: root_result.class_maps.len(),
        root_compute_ms,
        ..Default::default()
    };
    (
        DenseTrieClassBuildResult {
            state_classes: root_result.classes,
            class_maps: root_result
                .class_maps
                .iter()
                .cloned()
                .collect(),
        },
        profile,
    )
}

fn collect_possible_matches_dense_chunk_parallel(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    use rayon::prelude::*;

    let chunk_size = std::env::var("GLRMASK_PM_STATE_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&size| size > 0)
        .unwrap_or(2048);

    let chunks: Vec<&[u32]> = entries.chunks(chunk_size).collect();
    let partials: Vec<_> = chunks
        .par_iter()
        .map(|chunk| {
            if chunk.len() < 2048 {
                collect_possible_matches_dense_serial(tokenizer, root, num_internal_tokens, chunk)
            } else {
                collect_possible_matches_dense_batched(tokenizer, root, num_internal_tokens, chunk)
            }
        })
        .collect();

    let mut merged_maps = BTreeMap::new();
    let mut merged_profile = PossibleMatchesProfile::default();
    for (partial_maps, partial_profile) in partials {
        merged_maps.extend(partial_maps);
        merge_possible_matches_profile(&mut merged_profile, partial_profile);
    }
    (merged_maps, merged_profile)
}

pub(crate) fn count_root_child_internal_tsid_signatures(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    state_to_internal_tsid: &[u32],
) -> usize {
    let mut unique = FxHashSet::default();

    for &start_state in entries {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for (segment_bytes, _child) in root.iter_children() {
            let mut current_state = start_state;
            let mut segment_blocked = false;

            segment_bytes.len().hash(&mut hasher);
            for &byte in segment_bytes {
                byte.hash(&mut hasher);
                let Some(next_state) = tokenizer.step(current_state, byte) else {
                    segment_blocked = true;
                    break;
                };
                current_state = next_state;

                for terminal in tokenizer.matched_terminals_iter(current_state) {
                    terminal.hash(&mut hasher);
                }
                u32::MAX.hash(&mut hasher);
            }

            segment_blocked.hash(&mut hasher);
            if !segment_blocked {
                let is_end = tokenizer.is_end(current_state);
                is_end.hash(&mut hasher);
                if !is_end {
                    state_to_internal_tsid
                        .get(current_state as usize)
                        .copied()
                        .unwrap_or(current_state)
                        .hash(&mut hasher);
                }
            }

            0xFFFF_FFFEu32.hash(&mut hasher);
        }

        unique.insert(hasher.finish());
    }

    unique.len()
}

fn collect_possible_matches_dense_serial(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let mut computer = DensePossibleMatchesComputer::new(tokenizer, num_internal_tokens);
    let mut possible_matches_by_state: BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> = BTreeMap::new();
    let root_key = root as *const VocabPrefixTreeNode as usize;
    let num_words = (num_internal_tokens as usize + 63) / 64;

    for &original_state in entries {
        let _ = computer.possible_matches_for_node(root, original_state);
        let matches_for_state = computer
            .cache
            .remove(&(root_key, original_state))
            .expect("root possible-match map should be cached");
        let map = match Rc::try_unwrap(matches_for_state) {
            Ok(map) => map,
            Err(shared) => (*shared).clone(),
        };
        let mut merged = DensePossibleMatchMap::default();
        merge_dense_maps(&mut merged, &map, num_words);
        possible_matches_by_state.insert(original_state, merged);
    }

    (possible_matches_by_state, computer.profile())
}

fn collect_possible_matches_dense_batched(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    let num_words = (num_internal_tokens as usize + 63) / 64;
    let terminal_count = tokenizer.num_terminals as usize;

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

    let matched_terminals: Vec<Box<[TerminalID]>> = (0..tokenizer.num_states())
        .map(|state| tokenizer.matched_terminals_iter(state).collect::<Vec<_>>().into_boxed_slice())
        .collect();
    let is_end: Vec<bool> = (0..tokenizer.num_states())
        .map(|state| tokenizer.is_end(state))
        .collect();

    // Pre-compute reachable bitmaps for all trie nodes.
    let mut reachable_bitmaps: FxHashMap<usize, Box<[(u16, u64)]>> = FxHashMap::default();
    precompute_reachable_bitmaps(root, &mut reachable_bitmaps);

    let mut subtree_computer = DensePossibleMatchesComputer::new(tokenizer, num_internal_tokens);

    let n = entries.len();
    let mut results: Vec<Vec<Option<Box<[u64]>>>> = Vec::with_capacity(n);
    for _ in 0..n {
        results.push(vec![None; terminal_count]);
    }

    // Build initial live set: (entry_index, dfa_state)
    let live: Vec<(usize, u32)> = entries
        .iter()
        .enumerate()
        .map(|(i, &state)| (i, state))
        .collect();

    batched_walk_node(
        root,
        &live,
        &mut results,
        &mut subtree_computer,
        &flat_transitions,
        &self_loop_bytes,
        &matched_terminals,
        &is_end,
        num_words,
        &reachable_bitmaps,
    );

    let possible_matches_by_state: BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> =
        results
            .into_iter()
            .enumerate()
            .map(|(i, slots)| {
                let map = slots
                    .into_iter()
                    .enumerate()
                    .filter_map(|(terminal_id, bitmap)| {
                        bitmap.map(|bitmap| (terminal_id as TerminalID, bitmap))
                    })
                    .collect();
                (entries[i], map)
            })
            .collect();

    (possible_matches_by_state, subtree_computer.profile())
}

fn precompute_reachable_bitmaps(
    node: &VocabPrefixTreeNode,
    cache: &mut FxHashMap<usize, Box<[(u16, u64)]>>,
) {
    let key = node as *const VocabPrefixTreeNode as usize;
    if cache.contains_key(&key) {
        return;
    }
    cache.insert(key, reachable_sparse_bitmap(node));
    for (_, child) in node.iter_children() {
        precompute_reachable_bitmaps(child, cache);
    }
}

fn batched_walk_node(
    node: &VocabPrefixTreeNode,
    live: &[(usize, u32)],
    results: &mut [Vec<Option<Box<[u64]>>>],
    subtree_computer: &mut DensePossibleMatchesComputer<'_>,
    flat_trans: &[[u32; 256]],
    self_loop: &[U8Set],
    matched_terminals: &[Box<[TerminalID]>],
    is_end: &[bool],
    num_words: usize,
    reachable_bitmaps: &FxHashMap<usize, Box<[(u16, u64)]>>,
) {
    if live.is_empty() {
        return;
    }

    apply_batched_node_token_matches(node, live, results, matched_terminals, num_words);

    // Process each child edge
    for (segment, child) in node.iter_children() {
        batched_process_child(
            segment,
            child,
            live,
            results,
            subtree_computer,
            flat_trans,
            self_loop,
            matched_terminals,
            is_end,
            num_words,
            reachable_bitmaps,
        );
    }
}

fn apply_batched_node_token_matches(
    node: &VocabPrefixTreeNode,
    live: &[(usize, u32)],
    results: &mut [Vec<Option<Box<[u64]>>>],
    matched_terminals: &[Box<[TerminalID]>],
    num_words: usize,
) {
    if !node.has_token() {
        return;
    }

    let token_id = node.token_id() as u32;
    for &(idx, state) in live {
        for &terminal in matched_terminals[state as usize].iter() {
            let entry = results[idx][terminal as usize]
                .get_or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
            entry[token_id as usize / 64] |= 1u64 << (token_id % 64);
        }
    }
}

fn batched_process_child(
    segment: &[u8],
    child: &VocabPrefixTreeNode,
    live: &[(usize, u32)],
    results: &mut [Vec<Option<Box<[u64]>>>],
    subtree_computer: &mut DensePossibleMatchesComputer<'_>,
    flat_trans: &[[u32; 256]],
    self_loop: &[U8Set],
    matched_terminals: &[Box<[TerminalID]>],
    is_end: &[bool],
    num_words: usize,
    reachable_bitmaps: &FxHashMap<usize, Box<[(u16, u64)]>>,
) {
    let child_key = child as *const VocabPrefixTreeNode as usize;
    let reachable = &reachable_bitmaps[&child_key];
    let subtree_bytes = U8Set::from_words(*child.subtree_bytes());

    let mut child_live_by_state: FxHashMap<u32, Vec<usize>> = FxHashMap::default();

    for &(idx, state) in live {
        let mut s = state;
        let mut dead = false;
        let mut encountered_terminals = SmallVec::<[TerminalID; 8]>::new();

        for &byte in segment {
            let next = flat_trans[s as usize][byte as usize];
            if next == u32::MAX {
                dead = true;
                break;
            }
            s = next;
            for &terminal in matched_terminals[s as usize].iter() {
                if !encountered_terminals.contains(&terminal) {
                    encountered_terminals.push(terminal);
                }
            }
        }

        for terminal in encountered_terminals {
            let entry = results[idx][terminal as usize]
                .get_or_insert_with(|| vec![0u64; num_words].into_boxed_slice());
            for &(word_idx, mask) in reachable.iter() {
                entry[word_idx as usize] |= mask;
            }
        }

        if dead {
            continue;
        }
        if is_end[s as usize] {
            continue;
        }
        if subtree_bytes.is_subset(&self_loop[s as usize]) {
            continue;
        }
        child_live_by_state.entry(s).or_default().push(idx);
    }

    for (state, indices) in child_live_by_state {
        let child_matches = subtree_computer.possible_matches_for_node(child, state);
        for idx in indices {
            merge_dense_map_into_slots(&mut results[idx], child_matches.as_ref(), num_words);
        }
    }
}

fn collect_possible_matches_dense_parallel(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
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

    let entries: Vec<u32> = (0..tokenizer.num_states()).collect();

    let results: Vec<DensePossibleMatchMap> = entries
        .par_iter()
        .map(|&original_state| {
            let result = possible_matches_for_node_concurrent(
                tokenizer, root, original_state,
                num_words, &flat_transitions, &self_loop_bytes,
                &cache, &reachable_cache,
            );
            Arc::try_unwrap(result).unwrap_or_else(|arc| (*arc).clone())
        })
        .collect();

    let possible_matches_by_state: BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>> =
        results.into_iter().enumerate().map(|(i, map)| (entries[i], map)).collect();

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

