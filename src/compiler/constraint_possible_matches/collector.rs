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
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;
use crate::grammar::flat::TerminalID;

use crate::automata::lexer::observation::{ByteRows, ConfigInterner};

#[derive(Clone, Copy)]
pub(super) enum ByteTransitions<'a> {
    Projected {
        raw: &'a ByteRows,
        local_to_raw: &'a [u32],
        raw_to_local: &'a [u32],
    },
    Classed(&'a ByteRows),
}

impl ByteTransitions<'_> {
    #[inline]
    pub(super) fn target(self, state: u32, byte: u8) -> u32 {
        match self {
            Self::Projected { raw, local_to_raw, raw_to_local } => {
                let source = local_to_raw[state as usize];
                let target = raw.trans(source as usize, byte as usize);
                if target == u32::MAX { target } else { raw_to_local[target as usize] }
            }
            Self::Classed(rows) => rows.trans(state as usize, byte as usize),
        }
    }
}
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

type SegmentOutcomeCache = FxHashMap<u32, SegmentOutcome>;

#[derive(Default)]
struct SerialSegmentVectorCache {
    active_sets: ConfigInterner,
    outcomes: FxHashMap<(usize, u32), Arc<[SegmentOutcome]>>,
}

/// IDs from the frozen parent prefix remain valid in each worker. New sets
/// belong only to that worker; completed child maps carry terminal values,
/// never worker-local terminal-set IDs across the fork boundary.
#[derive(Default)]
struct TerminalSetInterner<'a> {
    parent: Option<&'a TerminalSetInterner<'a>>,
    parent_len: usize,
    mask_ids: FxHashMap<u128, u32>,
    sets: ConfigInterner,
}

impl TerminalSetInterner<'_> {
    fn fork(&self) -> TerminalSetInterner<'_> {
        TerminalSetInterner {
            parent: Some(self),
            parent_len: self.parent_len + self.sets.len(),
            ..Default::default()
        }
    }

    fn find(&self, terminals: &[TerminalID]) -> Option<u32> {
        self.parent.and_then(|parent| parent.find(terminals)).or_else(|| {
            self.sets.find(terminals).map(|id| self.parent_len as u32 + id)
        })
    }

    fn find_mask(&self, mask: u128) -> Option<u32> {
        self.parent.and_then(|parent| parent.find_mask(mask))
            .or_else(|| self.mask_ids.get(&mask).copied())
    }

    fn intern_slice(&mut self, terminals: &[TerminalID]) -> u32 {
        self.intern_vec(terminals.to_vec())
    }

    fn intern_vec(&mut self, mut terminals: Vec<TerminalID>) -> u32 {
        terminals.sort_unstable();
        terminals.dedup();
        if let Some(id) = self.parent.and_then(|parent| parent.find(&terminals)) {
            return id;
        }
        self.parent_len as u32 + self.sets.intern(terminals)
    }

    fn intern_mask(&mut self, mask: u128) -> u32 {
        if let Some(id) = self.find_mask(mask) {
            return id;
        }
        let mut terminals = Vec::with_capacity(mask.count_ones() as usize);
        let mut remaining = mask;
        while remaining != 0 {
            terminals.push(remaining.trailing_zeros());
            remaining &= remaining - 1;
        }
        let id = self.intern_vec(terminals);
        self.mask_ids.insert(mask, id);
        id
    }

    #[inline]
    fn get(&self, id: u32) -> &[TerminalID] {
        if (id as usize) < self.parent_len {
            self.parent.expect("inherited terminal-set ID").get(id)
        } else {
            &self.sets[id as usize - self.parent_len]
        }
    }
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
    segment_vector_cache_hits: usize,
    segment_vector_cache_misses: usize,
    segment_vector_cached_states: usize,
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
    parallel_worker_init_ms: f64,
    parallel_child_class_project_ms: f64,
    serial_child_class_project_ms: f64,
    parallel_children_built: usize,
    serial_children_built: usize,
    parallel_empty_children: usize,
    serial_empty_children: usize,
    nondefault_build_ms: f64,
    nondefault_rows_built: usize,
    sparse_signature_nodes: usize,
    sparse_signature_pairs: usize,
    empty_subtree_edges: usize,
    empty_subtree_rows_skipped: usize,
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
        self.segment_vector_cache_hits += other.segment_vector_cache_hits;
        self.segment_vector_cache_misses += other.segment_vector_cache_misses;
        self.segment_vector_cached_states += other.segment_vector_cached_states;
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
        self.parallel_worker_init_ms += other.parallel_worker_init_ms;
        self.parallel_child_class_project_ms += other.parallel_child_class_project_ms;
        self.serial_child_class_project_ms += other.serial_child_class_project_ms;
        self.parallel_children_built += other.parallel_children_built;
        self.serial_children_built += other.serial_children_built;
        self.parallel_empty_children += other.parallel_empty_children;
        self.serial_empty_children += other.serial_empty_children;
        self.nondefault_build_ms += other.nondefault_build_ms;
        self.nondefault_rows_built += other.nondefault_rows_built;
        self.sparse_signature_nodes += other.sparse_signature_nodes;
        self.sparse_signature_pairs += other.sparse_signature_pairs;
        self.empty_subtree_edges += other.empty_subtree_edges;
        self.empty_subtree_rows_skipped += other.empty_subtree_rows_skipped;
    }
}

#[inline]
fn canonical_terminal_box(terminals: &[TerminalID]) -> Option<Box<[TerminalID]>> {
    if terminals.is_empty() { return None; }
    Some(terminals.to_vec().into_boxed_slice())
}

#[inline]
fn append_range(map: &mut IntervalPossibleMatchMap, terminals: &[TerminalID], range: TokenRange) {
    if range.0 <= range.1
        && let Some(terminals) = canonical_terminal_box(terminals) {
            map.push(TerminalRangeGroup { terminals, ranges: vec![range] });
        }
}

#[inline]
fn append_ranges(map: &mut IntervalPossibleMatchMap, terminals: &[TerminalID], ranges: &[TokenRange]) {
    if !ranges.is_empty()
        && let Some(terminals) = canonical_terminal_box(terminals) {
            map.push(TerminalRangeGroup { terminals, ranges: ranges.to_vec() });
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
        if let Some(last) = merged.last_mut()
            && last.terminals.as_ref() == entry.terminals.as_ref() {
                last.ranges.extend_from_slice(&entry.ranges);
                continue;
            }
        merged.push(entry);
    }
    for entry in merged.iter_mut() { normalize_ranges(&mut entry.ranges); }
    merged.retain(|entry| !entry.ranges.is_empty());
    *map = merged;
}

#[derive(Default)]
struct IntervalMapBuilder {
    group_index: FxHashMap<Vec<TerminalID>, usize>,
    groups: IntervalPossibleMatchMap,
}

impl IntervalMapBuilder {
    #[inline]
    fn append_ranges(&mut self, terminals: &[TerminalID], ranges: &[TokenRange]) {
        if terminals.is_empty() || ranges.is_empty() {
            return;
        }
        let group_index = if let Some(&index) = self.group_index.get(terminals) {
            index
        } else {
            let key = terminals.to_vec();
            let index = self.groups.len();
            self.group_index.insert(key.clone(), index);
            self.groups.push(TerminalRangeGroup {
                terminals: key.into_boxed_slice(),
                ranges: Vec::new(),
            });
            index
        };
        let destination = &mut self.groups[group_index].ranges;
        for &(start, end) in ranges {
            if start > end {
                continue;
            }
            if let Some(current) = destination.last_mut()
                && start <= current.1.saturating_add(1)
            {
                current.1 = current.1.max(end);
            } else {
                destination.push((start, end));
            }
        }
    }

    #[inline]
    fn append_range(&mut self, terminals: &[TerminalID], range: TokenRange) {
        self.append_ranges(terminals, std::slice::from_ref(&range));
    }

    fn finish(mut self) -> IntervalPossibleMatchMap {
        self.groups.retain(|entry| !entry.ranges.is_empty());
        self.groups
            .sort_unstable_by(|left, right| left.terminals.as_ref().cmp(right.terminals.as_ref()));
        self.groups
    }
}

fn reachable_ranges(node: VocabPrefixTreeNode<'_>) -> Box<[TokenRange]> {
    if let Some((first, end)) = node.ordered_interval() {
        return if first < end { Box::new([(first, end - 1)]) } else { Box::new([]) };
    }
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

fn segment_outcomes_for_states(
    table: &mut SegmentOutcomeCache,
    needed_states: &[u32],
    segment: &[u8],
    matched_terminals: &[Box<[TerminalID]>],
    matched_terminal_masks: Option<&[u128]>,
    byte_transitions: ByteTransitions<'_>,
    terminal_sets: &mut TerminalSetInterner<'_>,
    empty_terminals_id: u32,
    terminal_stamps: &mut [u32],
    stamp_gen: &mut u32,
    timings: &mut BuildTimings,
    node_terminal_ids: &[u32],
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
            let next_state = byte_transitions.target(start_state, byte as u8);
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
        if use_state_cache {
            if let Some(&outcome) = table.get(&start_state) {
                timings.segment_cache_hits += 1;
                outcomes.push(outcome);
                continue;
            }
            timings.segment_cache_misses += 1;
        }
        let mut current_state = start_state;
        let mut blocked = false;
        let terminals_id = if let Some(matched_terminal_masks) = matched_terminal_masks {
            let mut terminal_mask = 0u128;
            for &byte in segment {
                timings.segment_bytes_scanned += 1;
                let next_state = byte_transitions.target(current_state, byte);
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
                let next_state = byte_transitions.target(current_state, byte);
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
            table.insert(start_state, outcome);
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

fn classify_fixed_signatures<const N: usize>(
    active_states: &[u32],
    mut signature_for: impl FnMut(usize, u32) -> [u32; N],
    representative_states: &mut Vec<u32>,
    representative_state_positions: &mut Vec<usize>,
    classes: &mut [u32],
) {
    const SMALL_CLASS_LIMIT: usize = 16;
    let mut small_signatures = [[u32::MAX; N]; SMALL_CLASS_LIMIT];
    let mut small_len = 0usize;
    let mut by_signature: Option<FxHashMap<[u32; N], u32>> = None;

    for (state_pos, &state) in active_states.iter().enumerate() {
        let signature = signature_for(state_pos, state);
        let class_id = if let Some(by_signature) = by_signature.as_mut() {
            if let Some(&class_id) = by_signature.get(&signature) {
                class_id
            } else {
                let class_id = representative_states.len() as u32;
                by_signature.insert(signature, class_id);
                representative_states.push(state);
                representative_state_positions.push(state_pos);
                class_id
            }
        } else if let Some(class_id) = small_signatures[..small_len]
            .iter()
            .position(|&existing| existing == signature)
        {
            class_id as u32
        } else {
            let class_id = representative_states.len() as u32;
            representative_states.push(state);
            representative_state_positions.push(state_pos);
            if small_len < SMALL_CLASS_LIMIT {
                small_signatures[small_len] = signature;
                small_len += 1;
            } else {
                let mut map = FxHashMap::default();
                for (existing_class, &existing_signature) in
                    small_signatures[..small_len].iter().enumerate()
                {
                    map.insert(existing_signature, existing_class as u32);
                }
                map.insert(signature, class_id);
                by_signature = Some(map);
            }
            class_id
        };
        classes[state_pos] = class_id;
    }
}

struct SignatureEntry { state_pos: usize, class_id: u32 }
struct ChildBuildData {
    outcomes: Arc<[SegmentOutcome]>,
    child_class_ids: Vec<u32>,
    reachable: Box<[TokenRange]>,
    result: NodeClasses,
}
struct ChildPendingData<'a> {
    child: VocabPrefixTreeNode<'a>,
    outcomes: Arc<[SegmentOutcome]>,
    descend_positions: Vec<u32>,
    child_active_states: Vec<u32>,
    reachable: Box<[TokenRange]>,
}

#[derive(Clone, Copy)]
struct BuildSource<'a> {
    num_states: usize,
    num_terminals: usize,
    matched_terminals: &'a [Box<[TerminalID]>],
    matched_terminal_masks: Option<&'a [u128]>,
    node_terminal_ids: &'a [u32],
    empty_terminals_id: u32,
    is_end: &'a [bool],
    byte_transitions: ByteTransitions<'a>,
    self_loop_bytes: &'a [U8Set],
    canonical_state: Option<&'a [u32]>,
    parallel_min_active: usize,
}

struct BuildScratch<'a> {
    terminal_sets: TerminalSetInterner<'a>,
    segment_cache: FxHashMap<Vec<u8>, usize>,
    segment_outcome_tables: Vec<SegmentOutcomeCache>,
    stamp_gen: u32,
    terminal_stamps: Vec<u32>,
    active_seen_gen: u32,
    active_seen_stamps: Vec<u32>,
    active_seen_positions: Vec<u32>,
    vector_cache: Option<SerialSegmentVectorCache>,
}

impl<'a> BuildScratch<'a> {
    fn new(source: &BuildSource<'_>, terminal_sets: TerminalSetInterner<'a>, vectors: bool) -> Self {
        Self {
            terminal_sets,
            segment_cache: FxHashMap::default(),
            segment_outcome_tables: Vec::new(),
            stamp_gen: 0,
            terminal_stamps: vec![0; source.num_terminals],
            active_seen_gen: 0,
            active_seen_stamps: vec![0; source.num_states],
            active_seen_positions: vec![0; source.num_states],
            vector_cache: vectors.then(SerialSegmentVectorCache::default),
        }
    }

    fn outcomes(
        &mut self, source: &BuildSource<'_>, states: &[u32],
        active_id: Option<u32>, segment: &[u8], timings: &mut BuildTimings,
    ) -> Arc<[SegmentOutcome]> {
        let segment_id = if let Some(&id) = self.segment_cache.get(segment) {
            id
        } else {
            let id = self.segment_outcome_tables.len();
            self.segment_cache.insert(segment.to_vec(), id);
            self.segment_outcome_tables.push(SegmentOutcomeCache::default());
            id
        };
        let key = active_id.map(|id| (segment_id, id));
        if let Some(outcomes) = key.and_then(|key| self.vector_cache.as_ref()?.outcomes.get(&key)) {
            timings.segment_vector_cache_hits += 1;
            timings.segment_vector_cached_states += states.len();
            return Arc::clone(outcomes);
        }
        let outcomes: Arc<[SegmentOutcome]> = Arc::from(segment_outcomes_for_states(
            &mut self.segment_outcome_tables[segment_id],
            states, segment, source.matched_terminals, source.matched_terminal_masks,
            source.byte_transitions, &mut self.terminal_sets, source.empty_terminals_id,
            &mut self.terminal_stamps, &mut self.stamp_gen, timings, source.node_terminal_ids,
            key.is_none(),
        ));
        if let Some(key) = key {
            timings.segment_vector_cache_misses += 1;
            self.vector_cache.as_mut().expect("active vector ID").outcomes.insert(key, Arc::clone(&outcomes));
        }
        outcomes
    }
}

fn build_child(
    source: &BuildSource<'_>, scratch: &mut BuildScratch<'_>, pending: ChildPendingData<'_>,
    timings: &mut BuildTimings, parallel_depth: u8, parallel: bool,
) -> ChildBuildData {
    if parallel { timings.parallel_children_built += 1; }
    else { timings.serial_children_built += 1; }
    let result = if pending.child_active_states.is_empty() {
        if parallel { timings.parallel_empty_children += 1; }
        else { timings.serial_empty_children += 1; }
        NodeClasses { classes: Vec::new(), class_maps: Vec::new() }
    } else {
        let started = Instant::now();
        let result = build_node(pending.child, &pending.child_active_states, source, scratch, timings, parallel_depth);
        timings.recursive_ms += elapsed_ms(started);
        result
    };
    let started = Instant::now();
    let child_class_ids = pending.descend_positions.iter()
        .map(|&pos| if pos == u32::MAX { u32::MAX } else { result.classes[pos as usize] })
        .collect();
    if parallel { timings.parallel_child_class_project_ms += elapsed_ms(started); }
    else { timings.serial_child_class_project_ms += elapsed_ms(started); }
    ChildBuildData { outcomes: pending.outcomes, child_class_ids, reachable: pending.reachable, result }
}

fn build_node(
    node: VocabPrefixTreeNode<'_>, active_states: &[u32], source: &BuildSource<'_>,
    scratch: &mut BuildScratch<'_>, timings: &mut BuildTimings, parallel_depth: u8,
) -> NodeClasses {
    let BuildSource { node_terminal_ids, empty_terminals_id, is_end, self_loop_bytes,
        canonical_state, parallel_min_active, .. } = *source;
    let active_id = scratch.vector_cache.as_mut().map(|cache| cache.active_sets.intern_slice(active_states));
    timings.nodes_built += 1;
    timings.active_state_rows += active_states.len();
    timings.max_active_states = timings.max_active_states.max(active_states.len());
    let mut child_pending = Vec::new();
    for (segment, child) in node.iter_children() {
        let outcomes = scratch.outcomes(source, active_states, active_id, segment, timings);
        let child_active_started_at = Instant::now();
        let subtree_bytes = U8Set::from_words(*child.subtree_bytes());
        let (descend_positions, child_active_states) =
            if subtree_bytes.is_empty() {
                timings.empty_subtree_edges += 1;
                timings.empty_subtree_rows_skipped += outcomes.len();
                (vec![u32::MAX; outcomes.len()], Vec::new())
            } else {
                let mut descend_positions = Vec::with_capacity(active_states.len());
                let mut child_active_states = Vec::new();
                let seen_gen = next_nonzero_generation(&mut scratch.active_seen_gen, &mut scratch.active_seen_stamps);
                for outcome in outcomes.iter() {
                    let descend = if let Some(end_state) = outcome.end_state {
                        let end_idx = end_state as usize;
                        if !is_end[end_idx]
                            && !subtree_bytes.is_subset(&self_loop_bytes[end_idx])
                        {
                            let descend_state = canonical_state
                                .and_then(|map| map.get(end_idx).copied())
                                .unwrap_or(end_state);
                            let descend_idx = descend_state as usize;
                            if scratch.active_seen_stamps[descend_idx] != seen_gen {
                                scratch.active_seen_stamps[descend_idx] = seen_gen;
                                scratch.active_seen_positions[descend_idx] = child_active_states.len() as u32;
                                child_active_states.push(descend_state);
                            }
                            scratch.active_seen_positions[descend_idx]
                        } else {
                            u32::MAX
                        }
                    } else {
                        u32::MAX
                    };
                    descend_positions.push(descend);
                }
                (
                    descend_positions,
                    child_active_states,
                )
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
            reachable,
        });
    }
    timings.child_edges += child_pending.len();
    timings.max_children_per_node = timings.max_children_per_node.max(child_pending.len());
    for pending in &child_pending {
        timings.child_active_state_rows += pending.child_active_states.len();
    }

    let should_parallelize = rayon::current_num_threads() > 1 && parallel_depth > 0 && child_pending.len() >= 4 && active_states.len() >= parallel_min_active;
    let child_data = if should_parallelize {
        let frozen_sets = &scratch.terminal_sets;
        let built: Vec<(ChildBuildData, BuildTimings)> = child_pending.into_par_iter()
            .map_init(
                || {
                    let started = Instant::now();
                    let worker = BuildScratch::new(source, frozen_sets.fork(), true);
                    (worker, Some(elapsed_ms(started)))
                },
                |(worker, setup_ms), pending| {
                    let mut local = BuildTimings::default();
                    local.parallel_worker_init_ms += setup_ms.take().unwrap_or_default();
                    let data = build_child(source, worker, pending, &mut local, 0, true);
                    (data, local)
                },
            )
            .collect();
        built.into_iter().map(|(data, local)| {
            timings.add_assign(local);
            data
        }).collect::<Vec<_>>()
    } else {
        child_pending.into_iter().map(|pending| {
            build_child(source, scratch, pending, timings, parallel_depth.saturating_sub(1), false)
        }).collect::<Vec<_>>()
    };

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
                classify_fixed_signatures(
                    active_states, |_, state| [node_terminal_ids[state as usize]],
                    &mut representative_states, &mut representative_state_positions, &mut classes,
                );
            } else if let Some(&state) = active_states.first() {
                representative_states.push(state);
                representative_state_positions.push(0);
                classes.fill(0);
            }
        }
        1 => {
            let child = &child_data[0];
            classify_fixed_signatures(
                active_states,
                |pos, state| [
                    if node.has_token() { node_terminal_ids[state as usize] } else { empty_terminals_id },
                    child.outcomes[pos].terminals_id,
                    child.child_class_ids[pos],
                ],
                &mut representative_states, &mut representative_state_positions, &mut classes,
            );
        }
        _ => {
            let dense_pairs = active_states.len() * child_data.len();
            let sparse_pairs = if child_data.len() > 4 && dense_pairs >= 4096 {
                let sparse_probe_started_at = Instant::now();
                let sparse_pairs = child_data
                    .iter()
                    .map(|child| {
                        child
                            .outcomes
                            .iter()
                            .zip(child.child_class_ids.iter())
                            .filter(|(outcome, child_class_id)| {
                                outcome.terminals_id != empty_terminals_id
                                    || **child_class_id != u32::MAX
                            })
                            .count()
                    })
                    .sum::<usize>();
                timings.nondefault_build_ms += elapsed_ms(sparse_probe_started_at);
                (sparse_pairs.saturating_mul(4) < dense_pairs).then_some(sparse_pairs)
            } else {
                None
            };
            if let Some(sparse_pairs) = sparse_pairs {
                timings.sparse_signature_nodes += 1;
                timings.sparse_signature_pairs += sparse_pairs;
                timings.nondefault_rows_built += sparse_pairs;
                let mut offsets = vec![0usize; active_states.len() + 1];
                for child in &child_data {
                    for (state_pos, (outcome, &child_class_id)) in child
                        .outcomes
                        .iter()
                        .zip(child.child_class_ids.iter())
                        .enumerate()
                    {
                        if outcome.terminals_id != empty_terminals_id
                            || child_class_id != u32::MAX
                        {
                            offsets[state_pos + 1] += 1;
                        }
                    }
                }
                for state_pos in 0..active_states.len() {
                    offsets[state_pos + 1] += offsets[state_pos];
                }
                let mut cursors = offsets[..active_states.len()].to_vec();
                let mut sparse_entries = vec![(0u32, 0u32, 0u32); sparse_pairs];
                for (child_index, child) in child_data.iter().enumerate() {
                    for (state_pos, (outcome, &child_class_id)) in child
                        .outcomes
                        .iter()
                        .zip(child.child_class_ids.iter())
                        .enumerate()
                    {
                        if outcome.terminals_id != empty_terminals_id
                            || child_class_id != u32::MAX
                        {
                            let slot = cursors[state_pos];
                            sparse_entries[slot] = (
                                child_index as u32,
                                outcome.terminals_id,
                                child_class_id,
                            );
                            cursors[state_pos] += 1;
                        }
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
                        let c0 = &child_data[0];
                        let c1 = &child_data[1];
                        classify_fixed_signatures(
                            active_states,
                            |state_pos, state| {
                            let node_terms = if node.has_token() {
                                node_terminal_ids[state as usize]
                            } else {
                                empty_terminals_id
                            };
                                [
                                node_terms,
                                c0.outcomes[state_pos].terminals_id,
                                c0.child_class_ids[state_pos],
                                c1.outcomes[state_pos].terminals_id,
                                c1.child_class_ids[state_pos],
                                ]
                            },
                            &mut representative_states,
                            &mut representative_state_positions,
                            &mut classes,
                        );
                    }
                    3 => {
                        let c0 = &child_data[0];
                        let c1 = &child_data[1];
                        let c2 = &child_data[2];
                        classify_fixed_signatures(
                            active_states,
                            |state_pos, state| {
                            let node_terms = if node.has_token() {
                                node_terminal_ids[state as usize]
                            } else {
                                empty_terminals_id
                            };
                                [
                                node_terms,
                                c0.outcomes[state_pos].terminals_id,
                                c0.child_class_ids[state_pos],
                                c1.outcomes[state_pos].terminals_id,
                                c1.child_class_ids[state_pos],
                                c2.outcomes[state_pos].terminals_id,
                                c2.child_class_ids[state_pos],
                                ]
                            },
                            &mut representative_states,
                            &mut representative_state_positions,
                            &mut classes,
                        );
                    }
                    4 => {
                        let c0 = &child_data[0];
                        let c1 = &child_data[1];
                        let c2 = &child_data[2];
                        let c3 = &child_data[3];
                        classify_fixed_signatures(
                            active_states,
                            |state_pos, state| {
                            let node_terms = if node.has_token() {
                                node_terminal_ids[state as usize]
                            } else {
                                empty_terminals_id
                            };
                                [
                                node_terms,
                                c0.outcomes[state_pos].terminals_id,
                                c0.child_class_ids[state_pos],
                                c1.outcomes[state_pos].terminals_id,
                                c1.child_class_ids[state_pos],
                                c2.outcomes[state_pos].terminals_id,
                                c2.child_class_ids[state_pos],
                                c3.outcomes[state_pos].terminals_id,
                                c3.child_class_ids[state_pos],
                                ]
                            },
                            &mut representative_states,
                            &mut representative_state_positions,
                            &mut classes,
                        );
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
        let mut result = IntervalMapBuilder::default();
        if node.has_token() {
            let token_id = node.token_id() as u32;
            result.append_range(
                scratch.terminal_sets.get(node_terminal_ids[state as usize]),
                (token_id, token_id),
            );
        }
        for child in &child_data {
            result.append_ranges(
                scratch.terminal_sets.get(child.outcomes[state_pos].terminals_id),
                &child.reachable,
            );
            let child_class_id = child.child_class_ids[state_pos];
            if child_class_id != u32::MAX {
                for entry in child.result.class_maps[child_class_id as usize].iter() {
                    result.append_ranges(entry.terminals.as_ref(), &entry.ranges);
                }
            }
        }
        class_maps.push(Arc::new(result.finish()));
    }
    timings.map_materialize_ms += elapsed_ms(map_started_at);
    NodeClasses { classes, class_maps }
}

pub(crate) fn collect_possible_matches_interval_trie_class_build_precomputed(
    root: VocabPrefixTreeNode<'_>,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
    num_states: usize,
    num_terminals: usize,
    matched_terminals: &[Box<[TerminalID]>],
    is_end: &[bool],
    byte_transitions: ByteTransitions<'_>,
    self_loop_bytes: &[U8Set],
) -> (TrieClassBuildResult, PossibleMatchesProfile) {
    debug_assert_eq!(matched_terminals.len(), num_states);
    debug_assert_eq!(is_end.len(), num_states);
    debug_assert_eq!(self_loop_bytes.len(), num_states);

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
    let parallel_depth = crate::env::var("GLRMASK_PM_ROOT_PARALLEL_DEPTH")
        .ok()
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(2);
    let parallel_min_active = crate::env::var("GLRMASK_PM_PARALLEL_MIN_ACTIVE_STATES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(512);
    if profile_summary_enabled() {
        eprintln!(
            "[glrmask/profile][trie_build_interval] root_parallel_children={} parallel_depth={} parallel_min_active_states={}",
            root.children().len(),
            parallel_depth,
            parallel_min_active,
        );
    }
    let root_started_at = Instant::now();
    let mut timings = BuildTimings::default();
    let serial_segment_cache_enabled = (rayon::current_num_threads() == 1 || parallel_depth == 0)
        && crate::env::var_os("GLRMASK_DISABLE_PM_SEGMENT_VECTOR_CACHE").is_none();
    let source = BuildSource {
        num_states, num_terminals, matched_terminals, matched_terminal_masks: matched_terminal_masks.as_deref(),
        node_terminal_ids: &node_terminal_ids, empty_terminals_id, is_end, byte_transitions,
        self_loop_bytes, canonical_state, parallel_min_active,
    };
    let mut scratch = BuildScratch::new(&source, terminal_sets, serial_segment_cache_enabled);
    let root_result = build_node(root, entries, &source, &mut scratch, &mut timings, parallel_depth);
    let root_compute_ms = elapsed_ms(root_started_at);
    if profile_summary_enabled() {
        eprintln!("[glrmask/profile][trie_build_interval_timings] segment_table_ms={:.3} signature_hash_ms={:.3} map_materialize_ms={:.3} child_active_ms={:.3} recursive_ms={:.3} reachable_interval_ms={:.3} child_precompute_ms={:.3} nondefault_build_ms={:.3} parallel_worker_init_ms={:.3} parallel_child_class_project_ms={:.3} serial_child_class_project_ms={:.3} parallel_children_built={} serial_children_built={} parallel_empty_children={} serial_empty_children={} classes_built={} nodes_built={} active_state_rows={} max_active_states={} child_edges={} max_children_per_node={} child_active_state_rows={} segment_calls={} segment_single_byte_calls={} segment_multi_byte_calls={} segment_states_requested={} segment_cache_hits={} segment_cache_misses={} segment_vector_cache_hits={} segment_vector_cache_misses={} segment_vector_cached_states={} segment_bytes_scanned={} segment_terminal_iters={} segment_terminal_pushes={} segment_mask_accumulations={} signature_nodes={} signature_rows={} signature_child_pairs={} signature_bucket_probes={} nondefault_rows_built={} sparse_signature_nodes={} sparse_signature_pairs={} empty_subtree_edges={} empty_subtree_rows_skipped={}", timings.segment_table_ms, timings.signature_hash_ms, timings.map_materialize_ms, timings.child_active_ms, timings.recursive_ms, timings.reachable_interval_ms, timings.child_precompute_ms, timings.nondefault_build_ms, timings.parallel_worker_init_ms, timings.parallel_child_class_project_ms, timings.serial_child_class_project_ms, timings.parallel_children_built, timings.serial_children_built, timings.parallel_empty_children, timings.serial_empty_children, timings.classes_built, timings.nodes_built, timings.active_state_rows, timings.max_active_states, timings.child_edges, timings.max_children_per_node, timings.child_active_state_rows, timings.segment_calls, timings.segment_single_byte_calls, timings.segment_multi_byte_calls, timings.segment_states_requested, timings.segment_cache_hits, timings.segment_cache_misses, timings.segment_vector_cache_hits, timings.segment_vector_cache_misses, timings.segment_vector_cached_states, timings.segment_bytes_scanned, timings.segment_terminal_iters, timings.segment_terminal_pushes, timings.segment_mask_accumulations, timings.signature_nodes, timings.signature_rows, timings.signature_child_pairs, timings.signature_bucket_probes, timings.nondefault_rows_built, timings.sparse_signature_nodes, timings.sparse_signature_pairs, timings.empty_subtree_edges, timings.empty_subtree_rows_skipped);
    }
    let profile = PossibleMatchesProfile {
        cache_entries: root_result.class_maps.len(),
        root_compute_ms,
        ..Default::default()
    };
    let mut state_classes = vec![u32::MAX; num_states];
    for (state_pos, &state) in entries.iter().enumerate() {
        if let Some(&class_id) = root_result.classes.get(state_pos)
            && let Some(slot) = state_classes.get_mut(state as usize) {
                *slot = class_id;
            }
    }
    (
        TrieClassBuildResult {
            state_classes,
            class_maps: root_result.class_maps,
        },
        profile,
    )
}

pub(crate) fn collect_possible_matches_interval_trie_class_build_with_classes(
    tokenizer: &Tokenizer,
    raw_transitions: Option<&ByteRows>,
    root: VocabPrefixTreeNode<'_>,
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
    let owned;
    let raw = match raw_transitions {
        Some(raw) => raw,
        None => {
            owned = ByteRows::from_tokenizer(tokenizer);
            &owned
        }
    };
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
        ByteTransitions::Classed(raw),
        &self_loop_bytes,
    )
}


#[cfg(test)]
mod tests {
    use super::TerminalSetInterner;

    #[test]
    fn forked_terminal_sets_preserve_inherited_ids_and_isolate_extensions() {
        let mut root = TerminalSetInterner::default();
        assert_eq!(root.intern_slice(&[]), 0);
        assert_eq!(root.intern_slice(&[7, 2, 7]), 1);
        let mut left = root.fork();
        let mut right = root.fork();
        assert_eq!(left.intern_mask((1 << 2) | (1 << 7)), 1);
        assert_eq!(right.intern_slice(&[7, 2]), 1);
        assert_eq!(left.intern_slice(&[3]), 2);
        assert_eq!(right.intern_slice(&[9]), 2);
        assert_eq!(left.get(2), &[3]);
        assert_eq!(right.get(2), &[9]);
        assert_eq!(left.get(1), &[2, 7]);
        assert_eq!(right.get(0), &[]);
        assert_eq!(root.sets.len(), 2);
    }
}
