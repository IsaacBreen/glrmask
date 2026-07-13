//! Constraint-specific possible-match collector.
//!
//! This collector keeps possible-match token sets as trie-order intervals rather
//! than dense token-id bitmaps. `mod.rs` builds the trie with token ids equal to
//! byte-sorted leaf ordinals, so every subtree is normally one contiguous token
//! interval.

use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::pm_profile::{elapsed_ms, profile_summary_enabled, PossibleMatchesProfile};
use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::TokenizerView;
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::grammar::flat::TerminalID;

pub(crate) type TokenRange = (u32, u32);

/// A possible-match row that applies the same token ranges to a whole terminal
/// set.  The previous representation expanded this to one `TerminalID ->
/// ranges` entry per terminal.  Keeping the terminal set intact is exact (the
/// row denotes the Cartesian product `terminals × ranges`) and avoids creating
/// millions of duplicate interval/event records for grammars where many
/// terminals are recognized at the same tokenizer prefix.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TerminalRangeGroup {
    pub(crate) terminals: Box<[TerminalID]>,
    pub(crate) ranges: Vec<TokenRange>,
}

pub(crate) type IntervalPossibleMatchMap = Vec<TerminalRangeGroup>;

pub(crate) struct TrieClassBuildResult {
    pub(crate) state_classes: Vec<u32>,
    pub(crate) class_maps: Vec<Arc<IntervalPossibleMatchMap>>,
}

impl TrieClassBuildResult {
    #[allow(dead_code)]
    pub(crate) fn expand_to_states(&self, entries: &[u32]) -> BTreeMap<u32, IntervalPossibleMatchMap> {
        entries.iter().copied().filter_map(|state| {
            let class_id = *self.state_classes.get(state as usize)?;
            if class_id == u32::MAX { return None; }
            Some((state, self.class_maps.get(class_id as usize)?.as_ref().clone()))
        }).collect()
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, Hash, PartialEq)]
struct SegmentOutcome { terminals_id: u32, end_state: Option<u32> }

enum SegmentOutcomeCache {
    Sparse { map: FxHashMap<u32, SegmentOutcome> },
    Dense { filled: Vec<u32>, outcomes: Vec<SegmentOutcome>, generation: u32 },
}

impl Default for SegmentOutcomeCache {
    fn default() -> Self {
        Self::Sparse { map: FxHashMap::default() }
    }
}

#[derive(Default)]
struct ActiveStateSetInterner {
    buckets: FxHashMap<u64, Vec<u32>>,
    sets: Vec<Box<[u32]>>,
}

impl ActiveStateSetInterner {
    fn intern(&mut self, states: &[u32]) -> u32 {
        let mut hash = 0u64;
        for &state in states {
            hash = mix_signature_word(hash, state);
        }
        self.intern_with_state_hash(states, hash)
    }

    fn intern_with_state_hash(&mut self, states: &[u32], state_hash: u64) -> u32 {
        let hash = mix_signature_word(state_hash, states.len() as u32);
        if let Some(id) = self.buckets.get(&hash).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|&id| self.sets[id as usize].as_ref() == states)
        }) {
            return id;
        }

        let id = self.sets.len() as u32;
        self.sets.push(states.to_vec().into_boxed_slice());
        self.buckets.entry(hash).or_default().push(id);
        id
    }
}

#[derive(Default)]
struct SerialSegmentVectorCache {
    active_sets: ActiveStateSetInterner,
    outcomes: FxHashMap<(usize, u32), Arc<[SegmentOutcome]>>,
}

#[derive(Clone, Default)]
struct TerminalSetInterner {
    ids: FxHashMap<Vec<TerminalID>, u32>,
    mask_ids: FxHashMap<u128, u32>,
    sets: Vec<Vec<TerminalID>>,
}

impl TerminalSetInterner {
    fn intern_slice(&mut self, terminals: &[TerminalID]) -> u32 {
        self.intern_vec(terminals.to_vec())
    }
    fn intern_vec(&mut self, mut terminals: Vec<TerminalID>) -> u32 {
        terminals.sort_unstable();
        terminals.dedup();
        if let Some(&id) = self.ids.get(&terminals) { return id; }
        let id = self.sets.len() as u32;
        self.ids.insert(terminals.clone(), id);
        self.sets.push(terminals);
        id
    }
    fn intern_mask(&mut self, mask: u128) -> u32 {
        if let Some(&id) = self.mask_ids.get(&mask) {
            return id;
        }
        let id = if mask == 0 {
            self.intern_slice(&[])
        } else {
            let mut terminals = Vec::with_capacity(mask.count_ones() as usize);
            let mut remaining = mask;
            while remaining != 0 {
                let bit = remaining.trailing_zeros();
                terminals.push(bit);
                remaining &= remaining - 1;
            }
            self.intern_vec(terminals)
        };
        self.mask_ids.insert(mask, id);
        id
    }
    #[inline]
    fn get(&self, id: u32) -> &[TerminalID] { &self.sets[id as usize] }
}

// `classes` is intentionally sparse: it is parallel to the `active_states`
// slice passed to `build_node`, not indexed by global tokenizer state.  The old
// collector allocated a full `num_states` vector at every trie node, which made
// large DFAs pay O(num_states * trie_nodes) just to write u32::MAX sentinels.
struct NodeClasses { classes: Vec<u32>, class_maps: Vec<Arc<IntervalPossibleMatchMap>> }

#[derive(Clone, Copy, Default)]
struct BuildTimings {
    segment_table_ms: f64,
    signature_hash_ms: f64,
    map_materialize_ms: f64,
    classes_built: usize,
    nodes_built: usize,
    active_state_rows: usize,
    max_active_states: usize,
    child_edges: usize,
    max_children_per_node: usize,
    child_active_state_rows: usize,
    segment_calls: usize,
    segment_single_byte_calls: usize,
    segment_multi_byte_calls: usize,
    segment_states_requested: usize,
    segment_cache_hits: usize,
    segment_cache_misses: usize,
    segment_dense_promotions: usize,
    segment_vector_cache_hits: usize,
    segment_vector_cache_misses: usize,
    segment_vector_cached_states: usize,
    segment_dense_hits: usize,
    segment_dense_misses: usize,
    segment_sparse_hits: usize,
    segment_sparse_misses: usize,
    segment_bytes_scanned: usize,
    segment_terminal_iters: usize,
    segment_terminal_pushes: usize,
    segment_mask_accumulations: usize,
    signature_nodes: usize,
    signature_rows: usize,
    signature_child_pairs: usize,
    signature_bucket_probes: usize,
    child_active_ms: f64,
    recursive_ms: f64,
    reachable_interval_ms: f64,
    child_precompute_ms: f64,
    parallel_terminal_sets_clone_ms: f64,
    parallel_segment_cache_init_ms: f64,
    parallel_stamp_alloc_ms: f64,
    parallel_child_class_project_ms: f64,
    serial_child_class_project_ms: f64,
    parallel_children_built: usize,
    serial_children_built: usize,
    parallel_empty_children: usize,
    serial_empty_children: usize,
}

impl BuildTimings {
    fn add_assign(&mut self, other: Self) {
        self.segment_table_ms += other.segment_table_ms;
        self.signature_hash_ms += other.signature_hash_ms;
        self.map_materialize_ms += other.map_materialize_ms;
        self.classes_built += other.classes_built;
        self.nodes_built += other.nodes_built;
        self.active_state_rows += other.active_state_rows;
        self.max_active_states = self.max_active_states.max(other.max_active_states);
        self.child_edges += other.child_edges;
        self.max_children_per_node = self.max_children_per_node.max(other.max_children_per_node);
        self.child_active_state_rows += other.child_active_state_rows;
        self.segment_calls += other.segment_calls;
        self.segment_single_byte_calls += other.segment_single_byte_calls;
        self.segment_multi_byte_calls += other.segment_multi_byte_calls;
        self.segment_states_requested += other.segment_states_requested;
        self.segment_cache_hits += other.segment_cache_hits;
        self.segment_cache_misses += other.segment_cache_misses;
        self.segment_dense_promotions += other.segment_dense_promotions;
        self.segment_vector_cache_hits += other.segment_vector_cache_hits;
        self.segment_vector_cache_misses += other.segment_vector_cache_misses;
        self.segment_vector_cached_states += other.segment_vector_cached_states;
        self.segment_dense_hits += other.segment_dense_hits;
        self.segment_dense_misses += other.segment_dense_misses;
        self.segment_sparse_hits += other.segment_sparse_hits;
        self.segment_sparse_misses += other.segment_sparse_misses;
        self.segment_bytes_scanned += other.segment_bytes_scanned;
        self.segment_terminal_iters += other.segment_terminal_iters;
        self.segment_terminal_pushes += other.segment_terminal_pushes;
        self.segment_mask_accumulations += other.segment_mask_accumulations;
        self.signature_nodes += other.signature_nodes;
        self.signature_rows += other.signature_rows;
        self.signature_child_pairs += other.signature_child_pairs;
        self.signature_bucket_probes += other.signature_bucket_probes;
        self.child_active_ms += other.child_active_ms;
        self.recursive_ms += other.recursive_ms;
        self.reachable_interval_ms += other.reachable_interval_ms;
        self.child_precompute_ms += other.child_precompute_ms;
        self.parallel_terminal_sets_clone_ms += other.parallel_terminal_sets_clone_ms;
        self.parallel_segment_cache_init_ms += other.parallel_segment_cache_init_ms;
        self.parallel_stamp_alloc_ms += other.parallel_stamp_alloc_ms;
        self.parallel_child_class_project_ms += other.parallel_child_class_project_ms;
        self.serial_child_class_project_ms += other.serial_child_class_project_ms;
        self.parallel_children_built += other.parallel_children_built;
        self.serial_children_built += other.serial_children_built;
        self.parallel_empty_children += other.parallel_empty_children;
        self.serial_empty_children += other.serial_empty_children;
    }
}

#[inline]
fn canonical_terminal_box(terminals: &[TerminalID]) -> Option<Box<[TerminalID]>> {
    if terminals.is_empty() { return None; }
    Some(terminals.to_vec().into_boxed_slice())
}

#[inline]
fn append_range(map: &mut IntervalPossibleMatchMap, terminals: &[TerminalID], range: TokenRange) {
    if range.0 <= range.1 {
        if let Some(terminals) = canonical_terminal_box(terminals) {
            map.push(TerminalRangeGroup { terminals, ranges: vec![range] });
        }
    }
}

#[inline]
fn append_ranges(map: &mut IntervalPossibleMatchMap, terminals: &[TerminalID], ranges: &[TokenRange]) {
    if !ranges.is_empty() {
        if let Some(terminals) = canonical_terminal_box(terminals) {
            map.push(TerminalRangeGroup { terminals, ranges: ranges.to_vec() });
        }
    }
}

fn merge_interval_maps(into: &mut IntervalPossibleMatchMap, other: &IntervalPossibleMatchMap) {
    into.extend(other.iter().cloned());
}

fn normalize_ranges(ranges: &mut Vec<TokenRange>) {
    if ranges.len() <= 1 { return; }
    ranges.sort_unstable();
    let mut write = 0usize;
    for read in 1..ranges.len() {
        let (start, end) = ranges[read];
        let current = &mut ranges[write];
        if start <= current.1.saturating_add(1) { current.1 = current.1.max(end); }
        else { write += 1; ranges[write] = (start, end); }
    }
    ranges.truncate(write + 1);
}

fn normalize_interval_map(map: &mut IntervalPossibleMatchMap) {
    if map.is_empty() { return; }

    map.retain(|entry| !entry.terminals.is_empty() && !entry.ranges.is_empty());
    for entry in map.iter_mut() { normalize_ranges(&mut entry.ranges); }
    map.retain(|entry| !entry.ranges.is_empty());
    if map.len() <= 1 { return; }

    map.sort_unstable_by(|left, right| left.terminals.as_ref().cmp(right.terminals.as_ref()));
    let mut merged: IntervalPossibleMatchMap = Vec::with_capacity(map.len());
    for entry in map.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.terminals.as_ref() == entry.terminals.as_ref() {
                last.ranges.extend_from_slice(&entry.ranges);
                continue;
            }
        }
        merged.push(entry);
    }
    for entry in merged.iter_mut() { normalize_ranges(&mut entry.ranges); }
    merged.retain(|entry| !entry.ranges.is_empty());
    *map = merged;
}

fn reachable_ranges(node: &VocabPrefixTreeNode) -> Box<[TokenRange]> {
    let mut ranges = Vec::new();
    for range in node.reachable_token_ids().ranges() {
        ranges.push((*range.start() as u32, *range.end() as u32));
    }
    normalize_ranges(&mut ranges);
    ranges.into_boxed_slice()
}

fn next_nonzero_generation(generation: &mut u32, stamps: &mut [u32]) -> u32 {
    *generation = generation.wrapping_add(1);
    if *generation == 0 {
        stamps.fill(0);
        *generation = 1;
    }
    *generation
}

fn dense_segment_cache_min_entries() -> usize {
    std::env::var("GLRMASK_PM_DENSE_SEGMENT_CACHE_MIN_ENTRIES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

fn promote_segment_outcome_cache(
    cache: &mut SegmentOutcomeCache,
    num_states: usize,
    timings: &mut BuildTimings,
) {
    let sparse = std::mem::take(cache);
    let SegmentOutcomeCache::Sparse { map } = sparse else {
        *cache = sparse;
        return;
    };
    let generation = 1u32;
    let mut filled = vec![0u32; num_states];
    let mut outcomes = vec![SegmentOutcome::default(); num_states];
    for (state, outcome) in map {
        let idx = state as usize;
        filled[idx] = generation;
        outcomes[idx] = outcome;
    }
    *cache = SegmentOutcomeCache::Dense { filled, outcomes, generation };
    timings.segment_dense_promotions += 1;
}

fn segment_outcomes_for_states(
    table: &mut SegmentOutcomeCache,
    needed_states: &[u32],
    segment: &[u8],
    matched_terminals: &[Box<[TerminalID]>],
    matched_terminal_masks: Option<&[u128]>,
    byte_transitions: &[Vec<u32>],
    terminal_sets: &mut TerminalSetInterner,
    empty_terminals_id: u32,
    terminal_stamps: &mut [u32],
    stamp_gen: &mut u32,
    timings: &mut BuildTimings,
    node_terminal_ids: &[u32],
    num_states: usize,
    dense_segment_cache_min_entries: usize,
    use_state_cache: bool,
) -> Vec<SegmentOutcome> {
    let started_at = Instant::now();
    timings.segment_calls += 1;
    timings.segment_states_requested += needed_states.len();
    let mut outcomes = Vec::with_capacity(needed_states.len());
    if segment.len() == 1 {
        timings.segment_single_byte_calls += 1;
        let byte = segment[0] as usize;
        for &start_state in needed_states {
            let next_state = byte_transitions[byte][start_state as usize];
            outcomes.push(if next_state == u32::MAX {
                SegmentOutcome { terminals_id: empty_terminals_id, end_state: None }
            } else {
                SegmentOutcome { terminals_id: node_terminal_ids[next_state as usize], end_state: Some(next_state) }
            });
        }
        timings.segment_table_ms += elapsed_ms(started_at);
        return outcomes;
    }

    timings.segment_multi_byte_calls += 1;

    for &start_state in needed_states {
        let outcome_idx = start_state as usize;
        let cached = if use_state_cache {
            match table {
                SegmentOutcomeCache::Sparse { map } => map.get(&start_state).copied().inspect(|_| {
                    timings.segment_cache_hits += 1;
                    timings.segment_sparse_hits += 1;
                }),
                SegmentOutcomeCache::Dense { filled, outcomes, generation } => {
                    if filled[outcome_idx] == *generation {
                        timings.segment_cache_hits += 1;
                        timings.segment_dense_hits += 1;
                        Some(outcomes[outcome_idx])
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        };
        if let Some(outcome) = cached {
            outcomes.push(outcome);
            continue;
        }
        if use_state_cache {
            timings.segment_cache_misses += 1;
            match table {
                SegmentOutcomeCache::Sparse { .. } => timings.segment_sparse_misses += 1,
                SegmentOutcomeCache::Dense { .. } => timings.segment_dense_misses += 1,
            }
        }
        let mut current_state = start_state;
        let mut blocked = false;
        let terminals_id = if let Some(matched_terminal_masks) = matched_terminal_masks {
            let mut terminal_mask = 0u128;
            for &byte in segment {
                timings.segment_bytes_scanned += 1;
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX { blocked = true; break; }
                current_state = next_state;
                let state_mask = matched_terminal_masks[current_state as usize];
                timings.segment_mask_accumulations += 1;
                timings.segment_terminal_iters += state_mask.count_ones() as usize;
                let new_bits = state_mask & !terminal_mask;
                timings.segment_terminal_pushes += new_bits.count_ones() as usize;
                terminal_mask |= state_mask;
            }
            terminal_sets.intern_mask(terminal_mask)
        } else {
            let current_gen = next_nonzero_generation(stamp_gen, terminal_stamps);
            let mut terminal_list = Vec::new();
            for &byte in segment {
                timings.segment_bytes_scanned += 1;
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX { blocked = true; break; }
                current_state = next_state;
                timings.segment_terminal_iters += matched_terminals[current_state as usize].len();
                for &terminal in matched_terminals[current_state as usize].iter() {
                    let terminal_idx = terminal as usize;
                    if terminal_stamps[terminal_idx] != current_gen {
                        terminal_stamps[terminal_idx] = current_gen;
                        terminal_list.push(terminal);
                        timings.segment_terminal_pushes += 1;
                    }
                }
            }
            if terminal_list.is_empty() {
                empty_terminals_id
            } else {
                terminal_sets.intern_vec(terminal_list)
            }
        };
        let outcome = SegmentOutcome { terminals_id, end_state: (!blocked).then_some(current_state) };
        if use_state_cache {
            let mut should_promote = false;
            match table {
                SegmentOutcomeCache::Sparse { map } => {
                    map.insert(start_state, outcome);
                    should_promote = dense_segment_cache_min_entries > 0
                        && map.len() >= dense_segment_cache_min_entries;
                }
                SegmentOutcomeCache::Dense { filled, outcomes, generation } => {
                    filled[outcome_idx] = *generation;
                    outcomes[outcome_idx] = outcome;
                }
            }
            if should_promote {
                promote_segment_outcome_cache(table, num_states, timings);
            }
        }
        outcomes.push(outcome);
    }
    timings.segment_table_ms += elapsed_ms(started_at);
    outcomes
}

#[inline]
fn mix_signature_word(hash: u64, word: u32) -> u64 {
    hash.wrapping_mul(0x517cc1b727220a95).wrapping_add((word as u64).wrapping_add(0x9e3779b97f4a7c15))
}

struct SignatureEntry { state_pos: usize, class_id: u32 }
struct ChildBuildData {
    outcomes: Arc<[SegmentOutcome]>,
    child_class_ids: Vec<u32>,
    nondefault_rows: Vec<(u32, u32, u32)>,
    reachable: Box<[TokenRange]>,
    result: NodeClasses,
}
struct ChildPendingData<'a> {
    child: &'a VocabPrefixTreeNode,
    outcomes: Arc<[SegmentOutcome]>,
    descend_positions: Vec<u32>,
    child_active_states: Vec<u32>,
    child_active_set_id: Option<u32>,
    reachable: Box<[TokenRange]>,
}

fn build_node(
    node: &VocabPrefixTreeNode,
    num_states: usize,
    num_terminals: usize,
    active_states: &[u32],
    matched_terminals: &[Box<[TerminalID]>],
    matched_terminal_masks: Option<&[u128]>,
    node_terminal_ids: &[u32],
    empty_terminals_id: u32,
    is_end: &[bool],
    byte_transitions: &[Vec<u32>],
    self_loop_bytes: &[U8Set],
    canonical_state: Option<&[u32]>,
    terminal_sets: &mut TerminalSetInterner,
    segment_cache: &mut FxHashMap<Vec<u8>, usize>,
    segment_outcome_tables: &mut Vec<SegmentOutcomeCache>,
    timings: &mut BuildTimings,
    stamp_gen: &mut u32,
    terminal_stamps: &mut [u32],
    active_seen_gen: &mut u32,
    active_seen_stamps: &mut [u32],
    active_seen_positions: &mut [u32],
    parallel_depth: u8,
    parallel_min_active: usize,
    dense_segment_cache_min_entries: usize,
    mut serial_segment_cache: Option<&mut SerialSegmentVectorCache>,
    serial_active_set_id: Option<u32>,
) -> NodeClasses {
    timings.nodes_built += 1;
    timings.active_state_rows += active_states.len();
    timings.max_active_states = timings.max_active_states.max(active_states.len());
    let mut child_pending = Vec::new();
    for (segment, child) in node.iter_children() {
        let segment_table_idx = if let Some(&idx) = segment_cache.get(segment) {
            idx
        } else {
            let idx = segment_outcome_tables.len();
            segment_outcome_tables.push(SegmentOutcomeCache::default());
            segment_cache.insert(segment.to_vec(), idx);
            idx
        };
        let outcomes = if let (Some(cache), Some(active_set_id)) =
            (serial_segment_cache.as_deref_mut(), serial_active_set_id)
        {
            let key = (segment_table_idx, active_set_id);
            if let Some(outcomes) = cache.outcomes.get(&key) {
                timings.segment_vector_cache_hits += 1;
                timings.segment_vector_cached_states += active_states.len();
                Arc::clone(outcomes)
            } else {
                let outcomes: Arc<[SegmentOutcome]> = Arc::from(segment_outcomes_for_states(
                    &mut segment_outcome_tables[segment_table_idx],
                    active_states,
                    segment,
                    matched_terminals,
                    matched_terminal_masks,
                    byte_transitions,
                    terminal_sets,
                    empty_terminals_id,
                    terminal_stamps,
                    stamp_gen,
                    timings,
                    node_terminal_ids,
                    num_states,
                    dense_segment_cache_min_entries,
                    false,
                ));
                timings.segment_vector_cache_misses += 1;
                cache.outcomes.insert(key, Arc::clone(&outcomes));
                outcomes
            }
        } else {
            Arc::from(segment_outcomes_for_states(
                &mut segment_outcome_tables[segment_table_idx],
                active_states,
                segment,
                matched_terminals,
                matched_terminal_masks,
                byte_transitions,
                terminal_sets,
                empty_terminals_id,
                terminal_stamps,
                stamp_gen,
                timings,
                node_terminal_ids,
                num_states,
                dense_segment_cache_min_entries,
                true,
            ))
        };
        let child_active_started_at = Instant::now();
        let subtree_bytes = U8Set::from_words(*child.subtree_bytes());
        let mut descend_positions = Vec::with_capacity(active_states.len());
        let mut child_active_states = Vec::new();
        let mut child_active_state_hash = 0u64;
        let seen_gen = next_nonzero_generation(active_seen_gen, active_seen_stamps);
        for outcome in outcomes.iter() {
            let descend = if let Some(end_state) = outcome.end_state {
                let end_idx = end_state as usize;
                if !is_end[end_idx] && !subtree_bytes.is_subset(&self_loop_bytes[end_idx]) {
                    let descend_state = canonical_state
                        .and_then(|map| map.get(end_idx).copied())
                        .unwrap_or(end_state);
                    let descend_idx = descend_state as usize;
                    if active_seen_stamps[descend_idx] != seen_gen {
                        active_seen_stamps[descend_idx] = seen_gen;
                        active_seen_positions[descend_idx] = child_active_states.len() as u32;
                        child_active_states.push(descend_state);
                        child_active_state_hash =
                            mix_signature_word(child_active_state_hash, descend_state);
                    }
                    active_seen_positions[descend_idx]
                } else { u32::MAX }
            } else { u32::MAX };
            descend_positions.push(descend);
        }
        let child_active_set_id = if child_active_states.is_empty() {
            None
        } else {
            serial_segment_cache.as_deref_mut().map(|cache| {
                cache
                    .active_sets
                    .intern_with_state_hash(&child_active_states, child_active_state_hash)
            })
        };
        timings.child_active_ms += elapsed_ms(child_active_started_at);
        let reachable_started_at = Instant::now();
        let reachable = reachable_ranges(child);
        timings.reachable_interval_ms += elapsed_ms(reachable_started_at);
        child_pending.push(ChildPendingData {
            child,
            outcomes,
            descend_positions,
            child_active_states,
            child_active_set_id,
            reachable,
        });
    }
    timings.child_edges += child_pending.len();
    timings.max_children_per_node = timings.max_children_per_node.max(child_pending.len());
    for pending in &child_pending {
        timings.child_active_state_rows += pending.child_active_states.len();
    }

    let should_parallelize = rayon::current_num_threads() > 1 && parallel_depth > 0 && child_pending.len() >= 4 && active_states.len() >= parallel_min_active;
    let mut child_data = Vec::with_capacity(child_pending.len());
    if should_parallelize {
        let built: Vec<(ChildBuildData, BuildTimings)> = child_pending.into_par_iter().map(|pending| {
            let mut local_timings = BuildTimings::default();
            local_timings.parallel_children_built += 1;
            let (result, child_class_ids) = if pending.child_active_states.is_empty() {
                local_timings.parallel_empty_children += 1;
                (NodeClasses { classes: Vec::new(), class_maps: Vec::new() }, vec![u32::MAX; pending.descend_positions.len()])
            } else {
                let terminal_sets_clone_started_at = Instant::now();
                let mut local_terminal_sets = terminal_sets.clone();
                local_timings.parallel_terminal_sets_clone_ms += elapsed_ms(terminal_sets_clone_started_at);

                let segment_cache_init_started_at = Instant::now();
                let mut local_segment_cache = FxHashMap::default();
                let mut local_segment_outcome_tables = Vec::<SegmentOutcomeCache>::new();
                local_timings.parallel_segment_cache_init_ms += elapsed_ms(segment_cache_init_started_at);

                let mut local_stamp_gen = 0u32;
                let mut local_active_seen_gen = 0u32;
                let stamp_alloc_started_at = Instant::now();
                let mut local_terminal_stamps = vec![0u32; num_terminals];
                let mut local_active_seen_stamps = vec![0u32; num_states];
                let mut local_active_seen_positions = vec![0u32; num_states];
                local_timings.parallel_stamp_alloc_ms += elapsed_ms(stamp_alloc_started_at);

                let recursive_started_at = Instant::now();
                let result = build_node(pending.child, num_states, num_terminals, &pending.child_active_states, matched_terminals, matched_terminal_masks, node_terminal_ids, empty_terminals_id, is_end, byte_transitions, self_loop_bytes, canonical_state, &mut local_terminal_sets, &mut local_segment_cache, &mut local_segment_outcome_tables, &mut local_timings, &mut local_stamp_gen, &mut local_terminal_stamps, &mut local_active_seen_gen, &mut local_active_seen_stamps, &mut local_active_seen_positions, 0, parallel_min_active, dense_segment_cache_min_entries, None, None);
                local_timings.recursive_ms += elapsed_ms(recursive_started_at);
                let child_class_project_started_at = Instant::now();
                let child_class_ids = pending.descend_positions.iter().map(|&pos| if pos == u32::MAX { u32::MAX } else { result.classes[pos as usize] }).collect();
                local_timings.parallel_child_class_project_ms += elapsed_ms(child_class_project_started_at);
                (result, child_class_ids)
            };
            let nondefault_rows = pending
                .outcomes
                .iter()
                .zip(child_class_ids.iter())
                .enumerate()
                .filter_map(|(state_pos, (outcome, &child_class_id))| {
                    (outcome.terminals_id != empty_terminals_id || child_class_id != u32::MAX)
                        .then_some((state_pos as u32, outcome.terminals_id, child_class_id))
                })
                .collect();
            (ChildBuildData { outcomes: pending.outcomes, child_class_ids, nondefault_rows, reachable: pending.reachable, result }, local_timings)
        }).collect();
        for (data, local_timings) in built { timings.add_assign(local_timings); child_data.push(data); }
    } else {
        for pending in child_pending {
            timings.serial_children_built += 1;
            let (result, child_class_ids) = if pending.child_active_states.is_empty() {
                timings.serial_empty_children += 1;
                (NodeClasses { classes: Vec::new(), class_maps: Vec::new() }, vec![u32::MAX; pending.descend_positions.len()])
            } else {
                let recursive_started_at = Instant::now();
                let result = build_node(pending.child, num_states, num_terminals, &pending.child_active_states, matched_terminals, matched_terminal_masks, node_terminal_ids, empty_terminals_id, is_end, byte_transitions, self_loop_bytes, canonical_state, terminal_sets, segment_cache, segment_outcome_tables, timings, stamp_gen, terminal_stamps, active_seen_gen, active_seen_stamps, active_seen_positions, parallel_depth.saturating_sub(1), parallel_min_active, dense_segment_cache_min_entries, serial_segment_cache.as_deref_mut(), pending.child_active_set_id);
                timings.recursive_ms += elapsed_ms(recursive_started_at);
                let child_class_project_started_at = Instant::now();
                let child_class_ids = pending.descend_positions.iter().map(|&pos| if pos == u32::MAX { u32::MAX } else { result.classes[pos as usize] }).collect();
                timings.serial_child_class_project_ms += elapsed_ms(child_class_project_started_at);
                (result, child_class_ids)
            };
            let nondefault_rows = pending
                .outcomes
                .iter()
                .zip(child_class_ids.iter())
                .enumerate()
                .filter_map(|(state_pos, (outcome, &child_class_id))| {
                    (outcome.terminals_id != empty_terminals_id || child_class_id != u32::MAX)
                        .then_some((state_pos as u32, outcome.terminals_id, child_class_id))
                })
                .collect();
            child_data.push(ChildBuildData { outcomes: pending.outcomes, child_class_ids, nondefault_rows, reachable: pending.reachable, result });
        }
    }

    let signature_started_at = Instant::now();
    timings.signature_nodes += 1;
    timings.signature_rows += active_states.len();
    timings.signature_child_pairs += active_states.len() * child_data.len();
    let mut representative_states = Vec::new();
    let mut representative_state_positions = Vec::new();
    let mut classes = vec![u32::MAX; active_states.len()];
    match child_data.len() {
        0 => {
            if node.has_token() {
                let mut by_term: FxHashMap<u32, u32> = FxHashMap::default();
                for (state_pos, &state) in active_states.iter().enumerate() {
                    let term_id = node_terminal_ids[state as usize];
                    let class_id = *by_term.entry(term_id).or_insert_with(|| { let id = representative_states.len() as u32; representative_states.push(state); representative_state_positions.push(state_pos); id });
                    classes[state_pos] = class_id;
                }
            } else if let Some(&state) = active_states.first() {
                representative_states.push(state); representative_state_positions.push(0);
                for class_id in &mut classes { *class_id = 0; }
            }
        }
        1 => {
            let child = &child_data[0];
            let mut by_sig: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
            for (state_pos, &state) in active_states.iter().enumerate() {
                let node_terms = if node.has_token() { node_terminal_ids[state as usize] } else { empty_terminals_id };
                let sig = (node_terms, child.outcomes[state_pos].terminals_id, child.child_class_ids[state_pos]);
                let class_id = *by_sig.entry(sig).or_insert_with(|| { let id = representative_states.len() as u32; representative_states.push(state); representative_state_positions.push(state_pos); id });
                classes[state_pos] = class_id;
            }
        }
        _ => {
            let dense_pairs = active_states.len() * child_data.len();
            let sparse_pairs = child_data
                .iter()
                .map(|child| child.nondefault_rows.len())
                .sum::<usize>();
            if dense_pairs >= 4096 && sparse_pairs.saturating_mul(4) < dense_pairs {
                let mut offsets = vec![0usize; active_states.len() + 1];
                for child in &child_data {
                    for &(state_pos, _, _) in &child.nondefault_rows {
                        offsets[state_pos as usize + 1] += 1;
                    }
                }
                for state_pos in 0..active_states.len() {
                    offsets[state_pos + 1] += offsets[state_pos];
                }
                let mut cursors = offsets[..active_states.len()].to_vec();
                let mut sparse_entries = vec![(0u32, 0u32, 0u32); sparse_pairs];
                for (child_index, child) in child_data.iter().enumerate() {
                    for &(state_pos, terminals_id, child_class_id) in &child.nondefault_rows {
                        let state_pos = state_pos as usize;
                        let slot = cursors[state_pos];
                        sparse_entries[slot] =
                            (child_index as u32, terminals_id, child_class_id);
                        cursors[state_pos] += 1;
                    }
                }

                let mut buckets: FxHashMap<u64, Vec<SignatureEntry>> = FxHashMap::default();
                let mut next_class_id = 0u32;
                for (state_pos, &state) in active_states.iter().enumerate() {
                    let node_terms = if node.has_token() {
                        node_terminal_ids[state as usize]
                    } else {
                        empty_terminals_id
                    };
                    let row = &sparse_entries[offsets[state_pos]..offsets[state_pos + 1]];
                    let mut hash = mix_signature_word(0, node_terms);
                    for &(child_index, terminals_id, child_class_id) in row {
                        hash = mix_signature_word(hash, child_index);
                        hash = mix_signature_word(hash, terminals_id);
                        hash = mix_signature_word(hash, child_class_id);
                    }
                    let bucket = buckets.entry(hash).or_default();
                    let mut found = false;
                    for entry in bucket.iter() {
                        timings.signature_bucket_probes += 1;
                        let rep_pos = entry.state_pos;
                        let rep_state = active_states[rep_pos];
                        let rep_node_terms = if node.has_token() {
                            node_terminal_ids[rep_state as usize]
                        } else {
                            empty_terminals_id
                        };
                        if rep_node_terms == node_terms
                            && &sparse_entries[offsets[rep_pos]..offsets[rep_pos + 1]] == row
                        {
                            classes[state_pos] = entry.class_id;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let class_id = next_class_id;
                        next_class_id += 1;
                        classes[state_pos] = class_id;
                        representative_states.push(state);
                        representative_state_positions.push(state_pos);
                        bucket.push(SignatureEntry { state_pos, class_id });
                    }
                }
            } else {
                match child_data.len() {
                    2 => {
                        let mut by_sig: FxHashMap<[u32; 5], u32> = FxHashMap::default();
                        let c0 = &child_data[0];
                        let c1 = &child_data[1];
                        for (state_pos, &state) in active_states.iter().enumerate() {
                            let node_terms = if node.has_token() {
                                node_terminal_ids[state as usize]
                            } else {
                                empty_terminals_id
                            };
                            let key = [
                                node_terms,
                                c0.outcomes[state_pos].terminals_id,
                                c0.child_class_ids[state_pos],
                                c1.outcomes[state_pos].terminals_id,
                                c1.child_class_ids[state_pos],
                            ];
                            let class_id = *by_sig.entry(key).or_insert_with(|| {
                                let id = representative_states.len() as u32;
                                representative_states.push(state);
                                representative_state_positions.push(state_pos);
                                id
                            });
                            classes[state_pos] = class_id;
                        }
                    }
                    3 => {
                        let mut by_sig: FxHashMap<[u32; 7], u32> = FxHashMap::default();
                        let c0 = &child_data[0];
                        let c1 = &child_data[1];
                        let c2 = &child_data[2];
                        for (state_pos, &state) in active_states.iter().enumerate() {
                            let node_terms = if node.has_token() {
                                node_terminal_ids[state as usize]
                            } else {
                                empty_terminals_id
                            };
                            let key = [
                                node_terms,
                                c0.outcomes[state_pos].terminals_id,
                                c0.child_class_ids[state_pos],
                                c1.outcomes[state_pos].terminals_id,
                                c1.child_class_ids[state_pos],
                                c2.outcomes[state_pos].terminals_id,
                                c2.child_class_ids[state_pos],
                            ];
                            let class_id = *by_sig.entry(key).or_insert_with(|| {
                                let id = representative_states.len() as u32;
                                representative_states.push(state);
                                representative_state_positions.push(state_pos);
                                id
                            });
                            classes[state_pos] = class_id;
                        }
                    }
                    4 => {
                        let mut by_sig: FxHashMap<[u32; 9], u32> = FxHashMap::default();
                        let c0 = &child_data[0];
                        let c1 = &child_data[1];
                        let c2 = &child_data[2];
                        let c3 = &child_data[3];
                        for (state_pos, &state) in active_states.iter().enumerate() {
                            let node_terms = if node.has_token() {
                                node_terminal_ids[state as usize]
                            } else {
                                empty_terminals_id
                            };
                            let key = [
                                node_terms,
                                c0.outcomes[state_pos].terminals_id,
                                c0.child_class_ids[state_pos],
                                c1.outcomes[state_pos].terminals_id,
                                c1.child_class_ids[state_pos],
                                c2.outcomes[state_pos].terminals_id,
                                c2.child_class_ids[state_pos],
                                c3.outcomes[state_pos].terminals_id,
                                c3.child_class_ids[state_pos],
                            ];
                            let class_id = *by_sig.entry(key).or_insert_with(|| {
                                let id = representative_states.len() as u32;
                                representative_states.push(state);
                                representative_state_positions.push(state_pos);
                                id
                            });
                            classes[state_pos] = class_id;
                        }
                    }
                    _ => {
                        let mut buckets: FxHashMap<u64, Vec<SignatureEntry>> = FxHashMap::default();
                        let mut next_class_id = 0u32;
                        for (state_pos, &state) in active_states.iter().enumerate() {
                            let node_terms = if node.has_token() { node_terminal_ids[state as usize] } else { empty_terminals_id };
                            let mut hash = mix_signature_word(0, node_terms);
                            for child in &child_data { hash = mix_signature_word(hash, child.outcomes[state_pos].terminals_id); hash = mix_signature_word(hash, child.child_class_ids[state_pos]); }
                            let bucket = buckets.entry(hash).or_default();
                            let mut found = false;
                            for entry in bucket.iter() {
                                timings.signature_bucket_probes += 1;
                                let rep_pos = entry.state_pos;
                                let rep_state = active_states[rep_pos];
                                let rep_node_terms = if node.has_token() { node_terminal_ids[rep_state as usize] } else { empty_terminals_id };
                                if rep_node_terms != node_terms { continue; }
                                if child_data.iter().all(|child| child.outcomes[rep_pos].terminals_id == child.outcomes[state_pos].terminals_id && child.child_class_ids[rep_pos] == child.child_class_ids[state_pos]) {
                                    classes[state_pos] = entry.class_id;
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                let class_id = next_class_id; next_class_id += 1;
                                classes[state_pos] = class_id;
                                representative_states.push(state); representative_state_positions.push(state_pos);
                                bucket.push(SignatureEntry { state_pos, class_id });
                            }
                        }
                    }
                }
            }
        }
    }
    timings.signature_hash_ms += elapsed_ms(signature_started_at);
    timings.classes_built += representative_states.len();

    let map_started_at = Instant::now();
    let mut class_maps = Vec::with_capacity(representative_states.len());
    for (&state, &state_pos) in representative_states.iter().zip(representative_state_positions.iter()) {
        let mut result = IntervalPossibleMatchMap::default();
        if node.has_token() {
            let token_id = node.token_id() as u32;
            append_range(&mut result, terminal_sets.get(node_terminal_ids[state as usize]), (token_id, token_id));
        }
        for child in &child_data {
            append_ranges(&mut result, terminal_sets.get(child.outcomes[state_pos].terminals_id), &child.reachable);
            let child_class_id = child.child_class_ids[state_pos];
            if child_class_id != u32::MAX { merge_interval_maps(&mut result, child.result.class_maps[child_class_id as usize].as_ref()); }
        }
        normalize_interval_map(&mut result);
        class_maps.push(Arc::new(result));
    }
    timings.map_materialize_ms += elapsed_ms(map_started_at);
    NodeClasses { classes, class_maps }
}

fn collect_possible_matches_interval_trie_class_build_precomputed(
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
    num_states: usize,
    num_terminals: usize,
    matched_terminals: &[Box<[TerminalID]>],
    is_end: &[bool],
    byte_transitions: &[Vec<u32>],
    self_loop_bytes: &[U8Set],
) -> (TrieClassBuildResult, PossibleMatchesProfile) {
    debug_assert_eq!(matched_terminals.len(), num_states);
    debug_assert_eq!(is_end.len(), num_states);
    debug_assert_eq!(self_loop_bytes.len(), num_states);
    debug_assert_eq!(byte_transitions.len(), 256);
    debug_assert!(byte_transitions.iter().all(|column| column.len() == num_states));

    let matched_terminal_masks: Option<Vec<u128>> = if num_terminals <= 128 {
        Some(
            matched_terminals
                .iter()
                .map(|terminals| {
                    let mut mask = 0u128;
                    for &terminal in terminals.iter() {
                        mask |= 1u128 << terminal;
                    }
                    mask
                })
                .collect(),
        )
    } else {
        None
    };
    let mut terminal_sets = TerminalSetInterner::default();
    let empty_terminals_id = terminal_sets.intern_slice(&[]);
    let node_terminal_ids: Vec<u32> = if let Some(matched_terminal_masks) = matched_terminal_masks.as_ref() {
        matched_terminal_masks
            .iter()
            .map(|&mask| terminal_sets.intern_mask(mask))
            .collect()
    } else {
        matched_terminals
            .iter()
            .map(|terminals| terminal_sets.intern_slice(terminals))
            .collect()
    };
    let parallel_depth = std::env::var("GLRMASK_PM_ROOT_PARALLEL_DEPTH")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(5);
    let parallel_min_active = std::env::var("GLRMASK_PM_PARALLEL_MIN_ACTIVE_STATES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    let dense_segment_cache_min_entries = dense_segment_cache_min_entries();
    if profile_summary_enabled() {
        eprintln!(
            "[glrmask/profile][trie_build_interval] root_parallel_children={} parallel_depth={} parallel_min_active_states={} dense_segment_cache_min_entries={}",
            root.children().len(),
            parallel_depth,
            parallel_min_active,
            dense_segment_cache_min_entries,
        );
    }
    let root_started_at = Instant::now();
    let mut timings = BuildTimings::default();
    let mut segment_cache: FxHashMap<Vec<u8>, usize> = FxHashMap::default();
    let mut segment_outcome_tables = Vec::<SegmentOutcomeCache>::new();
    let mut stamp_gen = 0u32;
    let mut terminal_stamps = vec![0u32; num_terminals];
    let mut active_seen_gen = 0u32;
    let mut active_seen_stamps = vec![0u32; num_states];
    let mut active_seen_positions = vec![0u32; num_states];
    let serial_segment_cache_enabled = rayon::current_num_threads() == 1
        && std::env::var_os("GLRMASK_DISABLE_PM_SEGMENT_VECTOR_CACHE").is_none();
    let mut serial_segment_cache =
        serial_segment_cache_enabled.then(SerialSegmentVectorCache::default);
    let root_active_set_id = serial_segment_cache
        .as_mut()
        .map(|cache| cache.active_sets.intern(entries));
    let root_result = build_node(
        root,
        num_states,
        num_terminals,
        entries,
        matched_terminals,
        matched_terminal_masks.as_deref(),
        &node_terminal_ids,
        empty_terminals_id,
        is_end,
        byte_transitions,
        self_loop_bytes,
        canonical_state,
        &mut terminal_sets,
        &mut segment_cache,
        &mut segment_outcome_tables,
        &mut timings,
        &mut stamp_gen,
        &mut terminal_stamps,
        &mut active_seen_gen,
        &mut active_seen_stamps,
        &mut active_seen_positions,
        parallel_depth,
        parallel_min_active,
        dense_segment_cache_min_entries,
        serial_segment_cache.as_mut(),
        root_active_set_id,
    );
    let root_compute_ms = elapsed_ms(root_started_at);
    if profile_summary_enabled() {
        eprintln!("[glrmask/profile][trie_build_interval_timings] segment_table_ms={:.3} signature_hash_ms={:.3} map_materialize_ms={:.3} child_active_ms={:.3} recursive_ms={:.3} reachable_interval_ms={:.3} child_precompute_ms={:.3} parallel_terminal_sets_clone_ms={:.3} parallel_segment_cache_init_ms={:.3} parallel_stamp_alloc_ms={:.3} parallel_child_class_project_ms={:.3} serial_child_class_project_ms={:.3} parallel_children_built={} serial_children_built={} parallel_empty_children={} serial_empty_children={} classes_built={} nodes_built={} active_state_rows={} max_active_states={} child_edges={} max_children_per_node={} child_active_state_rows={} segment_calls={} segment_single_byte_calls={} segment_multi_byte_calls={} segment_states_requested={} segment_cache_hits={} segment_cache_misses={} segment_dense_promotions={} segment_vector_cache_hits={} segment_vector_cache_misses={} segment_vector_cached_states={} segment_dense_hits={} segment_dense_misses={} segment_sparse_hits={} segment_sparse_misses={} segment_bytes_scanned={} segment_terminal_iters={} segment_terminal_pushes={} segment_mask_accumulations={} signature_nodes={} signature_rows={} signature_child_pairs={} signature_bucket_probes={}", timings.segment_table_ms, timings.signature_hash_ms, timings.map_materialize_ms, timings.child_active_ms, timings.recursive_ms, timings.reachable_interval_ms, timings.child_precompute_ms, timings.parallel_terminal_sets_clone_ms, timings.parallel_segment_cache_init_ms, timings.parallel_stamp_alloc_ms, timings.parallel_child_class_project_ms, timings.serial_child_class_project_ms, timings.parallel_children_built, timings.serial_children_built, timings.parallel_empty_children, timings.serial_empty_children, timings.classes_built, timings.nodes_built, timings.active_state_rows, timings.max_active_states, timings.child_edges, timings.max_children_per_node, timings.child_active_state_rows, timings.segment_calls, timings.segment_single_byte_calls, timings.segment_multi_byte_calls, timings.segment_states_requested, timings.segment_cache_hits, timings.segment_cache_misses, timings.segment_dense_promotions, timings.segment_vector_cache_hits, timings.segment_vector_cache_misses, timings.segment_vector_cached_states, timings.segment_dense_hits, timings.segment_dense_misses, timings.segment_sparse_hits, timings.segment_sparse_misses, timings.segment_bytes_scanned, timings.segment_terminal_iters, timings.segment_terminal_pushes, timings.segment_mask_accumulations, timings.signature_nodes, timings.signature_rows, timings.signature_child_pairs, timings.signature_bucket_probes);
    }
    let profile = PossibleMatchesProfile {
        cache_entries: root_result.class_maps.len(),
        root_compute_ms,
        ..Default::default()
    };
    let mut state_classes = vec![u32::MAX; num_states];
    for (state_pos, &state) in entries.iter().enumerate() {
        if let Some(&class_id) = root_result.classes.get(state_pos) {
            if let Some(slot) = state_classes.get_mut(state as usize) {
                *slot = class_id;
            }
        }
    }
    (
        TrieClassBuildResult {
            state_classes,
            class_maps: root_result.class_maps.iter().cloned().collect(),
        },
        profile,
    )
}

pub(crate) fn collect_possible_matches_interval_trie_class_build_with_classes(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
) -> (TrieClassBuildResult, PossibleMatchesProfile) {
    let num_states = tokenizer.num_states() as usize;
    let num_terminals = tokenizer.num_terminals() as usize;
    let matched_terminals: Vec<Box<[TerminalID]>> = (0..tokenizer.num_states())
        .map(|state| {
            tokenizer
                .matched_terminals_iter(state)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect();
    let is_end: Vec<bool> = (0..tokenizer.num_states())
        .map(|state| tokenizer.is_end(state))
        .collect();
    let mut byte_transitions = vec![vec![u32::MAX; num_states]; 256];
    for state_idx in 0..num_states {
        for (byte, target) in tokenizer.transitions_from(state_idx as u32) {
            byte_transitions[byte as usize][state_idx] = target;
        }
    }
    let self_loop_bytes: Vec<U8Set> = (0..num_states)
        .map(|state_idx| tokenizer.self_loop_bytes(state_idx as u32))
        .collect();
    collect_possible_matches_interval_trie_class_build_precomputed(
        root,
        entries,
        canonical_state,
        num_states,
        num_terminals,
        &matched_terminals,
        &is_end,
        &byte_transitions,
        &self_loop_bytes,
    )
}

pub(crate) fn collect_possible_matches_interval_trie_class_build_for_flat_view(
    tokenizer_view: &TokenizerView,
    num_terminals: usize,
    is_end: &[bool],
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
) -> (TrieClassBuildResult, PossibleMatchesProfile) {
    let dfa = tokenizer_view.dfa();
    let num_states = dfa.states.len();
    let matched_terminals = dfa
        .states
        .iter()
        .map(|state| {
            state
                .finalizers
                .iter()
                .map(|&terminal| terminal as TerminalID)
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect::<Vec<_>>();
    let mut byte_transitions = vec![vec![u32::MAX; num_states]; 256];
    for state in 0..num_states {
        for byte in 0..256usize {
            byte_transitions[byte][state] = dfa.trans(state, byte);
        }
    }
    let self_loop_bytes = (0..num_states)
        .map(|state| {
            U8Set::from_predicate(|byte| dfa.trans(state, byte as usize) == state as u32)
        })
        .collect::<Vec<_>>();
    collect_possible_matches_interval_trie_class_build_precomputed(
        root,
        entries,
        canonical_state,
        num_states,
        num_terminals,
        &matched_terminals,
        is_end,
        &byte_transitions,
        &self_loop_bytes,
    )
}
