//! Constraint-specific possible-match collector.
//!
//! This collector keeps possible-match token sets as trie-order intervals rather
//! than dense token-id bitmaps. `mod.rs` builds the trie with token ids equal to
//! byte-sorted leaf ordinals, so every subtree is normally one contiguous token
//! interval.

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

pub(crate) type TokenRange = (u32, u32);
pub(crate) type IntervalPossibleMatchMap = BTreeMap<TerminalID, Vec<TokenRange>>;

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

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
struct SegmentOutcome { terminals_id: u32, end_state: Option<u32> }

#[derive(Clone, Default)]
struct TerminalSetInterner {
    ids: FxHashMap<Vec<TerminalID>, u32>,
    sets: Vec<Vec<TerminalID>>,
}

impl TerminalSetInterner {
    fn intern_slice(&mut self, terminals: &[TerminalID]) -> u32 {
        if let Some(&id) = self.ids.get(terminals) { return id; }
        let id = self.sets.len() as u32;
        let owned = terminals.to_vec();
        self.ids.insert(owned.clone(), id);
        self.sets.push(owned);
        id
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
    child_active_ms: f64,
    recursive_ms: f64,
    reachable_interval_ms: f64,
    child_precompute_ms: f64,
}

impl BuildTimings {
    fn add_assign(&mut self, other: Self) {
        self.segment_table_ms += other.segment_table_ms;
        self.signature_hash_ms += other.signature_hash_ms;
        self.map_materialize_ms += other.map_materialize_ms;
        self.classes_built += other.classes_built;
        self.child_active_ms += other.child_active_ms;
        self.recursive_ms += other.recursive_ms;
        self.reachable_interval_ms += other.reachable_interval_ms;
        self.child_precompute_ms += other.child_precompute_ms;
    }
}

#[inline]
fn append_range(map: &mut IntervalPossibleMatchMap, terminal: TerminalID, range: TokenRange) {
    if range.0 <= range.1 { map.entry(terminal).or_default().push(range); }
}

#[inline]
fn append_ranges(map: &mut IntervalPossibleMatchMap, terminal: TerminalID, ranges: &[TokenRange]) {
    if !ranges.is_empty() { map.entry(terminal).or_default().extend_from_slice(ranges); }
}

fn merge_interval_maps(into: &mut IntervalPossibleMatchMap, other: &IntervalPossibleMatchMap) {
    for (&terminal, ranges) in other { append_ranges(into, terminal, ranges); }
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
    map.retain(|_, ranges| { normalize_ranges(ranges); !ranges.is_empty() });
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

fn segment_outcomes_for_states(
    table: &mut FxHashMap<u32, SegmentOutcome>,
    needed_states: &[u32],
    segment: &[u8],
    matched_terminals: &[Box<[TerminalID]>],
    byte_transitions: &[Vec<u32>],
    terminal_sets: &mut TerminalSetInterner,
    empty_terminals_id: u32,
    terminal_stamps: &mut [u32],
    stamp_gen: &mut u32,
    timings: &mut BuildTimings,
    node_terminal_ids: &[u32],
) -> Vec<SegmentOutcome> {
    let started_at = Instant::now();
    let mut outcomes = Vec::with_capacity(needed_states.len());
    if segment.len() == 1 {
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

    for &start_state in needed_states {
        if let Some(&outcome) = table.get(&start_state) { outcomes.push(outcome); continue; }
        let current_gen = next_nonzero_generation(stamp_gen, terminal_stamps);
        let mut terminal_list = Vec::new();
        let mut current_state = start_state;
        let mut blocked = false;
        for &byte in segment {
            let next_state = byte_transitions[byte as usize][current_state as usize];
            if next_state == u32::MAX { blocked = true; break; }
            current_state = next_state;
            for &terminal in matched_terminals[current_state as usize].iter() {
                let terminal_idx = terminal as usize;
                if terminal_stamps[terminal_idx] != current_gen {
                    terminal_stamps[terminal_idx] = current_gen;
                    terminal_list.push(terminal);
                }
            }
        }
        let terminals_id = if terminal_list.is_empty() {
            empty_terminals_id
        } else {
            terminal_sets.intern_vec(terminal_list)
        };
        let outcome = SegmentOutcome { terminals_id, end_state: (!blocked).then_some(current_state) };
        table.insert(start_state, outcome);
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
struct ChildBuildData { outcomes: Vec<SegmentOutcome>, child_class_ids: Vec<u32>, reachable: Box<[TokenRange]>, result: NodeClasses }
struct ChildPendingData<'a> { child: &'a VocabPrefixTreeNode, outcomes: Vec<SegmentOutcome>, descend_positions: Vec<u32>, child_active_states: Vec<u32>, reachable: Box<[TokenRange]> }

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
    canonical_state: Option<&[u32]>,
    terminal_sets: &mut TerminalSetInterner,
    segment_cache: &mut FxHashMap<Vec<u8>, usize>,
    segment_outcome_tables: &mut Vec<FxHashMap<u32, SegmentOutcome>>,
    timings: &mut BuildTimings,
    stamp_gen: &mut u32,
    terminal_stamps: &mut [u32],
    active_seen_gen: &mut u32,
    active_seen_stamps: &mut [u32],
    active_seen_positions: &mut [u32],
    parallel_depth: u8,
    parallel_min_active: usize,
) -> NodeClasses {
    let mut child_pending = Vec::new();
    for (segment, child) in node.iter_children() {
        let segment_key = segment.to_vec();
        let segment_table_idx = if let Some(&idx) = segment_cache.get(&segment_key) { idx } else {
            let idx = segment_outcome_tables.len();
            segment_outcome_tables.push(FxHashMap::default());
            segment_cache.insert(segment_key, idx);
            idx
        };
        let outcomes = segment_outcomes_for_states(&mut segment_outcome_tables[segment_table_idx], active_states, segment, matched_terminals, byte_transitions, terminal_sets, empty_terminals_id, terminal_stamps, stamp_gen, timings, node_terminal_ids);
        let child_active_started_at = Instant::now();
        let subtree_bytes = U8Set::from_words(*child.subtree_bytes());
        let mut descend_positions = Vec::with_capacity(active_states.len());
        let mut child_active_states = Vec::new();
        let seen_gen = next_nonzero_generation(active_seen_gen, active_seen_stamps);
        for outcome in &outcomes {
            let descend = if let Some(end_state) = outcome.end_state {
                let end_idx = end_state as usize;
                if !is_end[end_idx] && !subtree_bytes.is_subset(&self_loop_bytes[end_idx]) {
                    // The optional initial map is a k-bounded language equivalence over
                    // all vocab bytes.  After a child edge, the remaining suffix is no
                    // longer than the original token bound, so the representative state
                    // has the same possible-match map but keeps recursion on the compact
                    // state set instead of drifting back to raw DFA states.
                    let descend_state = canonical_state
                        .and_then(|map| map.get(end_idx).copied())
                        .unwrap_or(end_state);
                    let descend_idx = descend_state as usize;
                    if active_seen_stamps[descend_idx] != seen_gen {
                        active_seen_stamps[descend_idx] = seen_gen;
                        active_seen_positions[descend_idx] = child_active_states.len() as u32;
                        child_active_states.push(descend_state);
                    }
                    active_seen_positions[descend_idx]
                } else { u32::MAX }
            } else { u32::MAX };
            descend_positions.push(descend);
        }
        timings.child_active_ms += elapsed_ms(child_active_started_at);
        let reachable_started_at = Instant::now();
        let reachable = reachable_ranges(child);
        timings.reachable_interval_ms += elapsed_ms(reachable_started_at);
        child_pending.push(ChildPendingData { child, outcomes, descend_positions, child_active_states, reachable });
    }

    let should_parallelize = rayon::current_num_threads() > 1 && parallel_depth > 0 && child_pending.len() >= 4 && active_states.len() >= parallel_min_active;
    let mut child_data = Vec::with_capacity(child_pending.len());
    if should_parallelize {
        let built: Vec<(ChildBuildData, BuildTimings)> = child_pending.into_par_iter().map(|pending| {
            let mut local_timings = BuildTimings::default();
            let (result, child_class_ids) = if pending.child_active_states.is_empty() {
                (NodeClasses { classes: Vec::new(), class_maps: Vec::new() }, vec![u32::MAX; pending.descend_positions.len()])
            } else {
                let mut local_terminal_sets = terminal_sets.clone();
                let mut local_segment_cache = FxHashMap::default();
                let mut local_segment_outcome_tables = Vec::<FxHashMap<u32, SegmentOutcome>>::new();
                let mut local_stamp_gen = 0u32;
                let mut local_terminal_stamps = vec![0u32; tokenizer.num_terminals as usize];
                let mut local_active_seen_gen = 0u32;
                let mut local_active_seen_stamps = vec![0u32; tokenizer.num_states() as usize];
                let mut local_active_seen_positions = vec![0u32; tokenizer.num_states() as usize];
                let recursive_started_at = Instant::now();
                let result = build_node(pending.child, tokenizer, &pending.child_active_states, matched_terminals, node_terminal_ids, empty_terminals_id, is_end, byte_transitions, self_loop_bytes, canonical_state, &mut local_terminal_sets, &mut local_segment_cache, &mut local_segment_outcome_tables, &mut local_timings, &mut local_stamp_gen, &mut local_terminal_stamps, &mut local_active_seen_gen, &mut local_active_seen_stamps, &mut local_active_seen_positions, 0, parallel_min_active);
                local_timings.recursive_ms += elapsed_ms(recursive_started_at);
                let child_class_ids = pending.descend_positions.iter().map(|&pos| if pos == u32::MAX { u32::MAX } else { result.classes[pos as usize] }).collect();
                (result, child_class_ids)
            };
            (ChildBuildData { outcomes: pending.outcomes, child_class_ids, reachable: pending.reachable, result }, local_timings)
        }).collect();
        for (data, local_timings) in built { timings.add_assign(local_timings); child_data.push(data); }
    } else {
        for pending in child_pending {
            let (result, child_class_ids) = if pending.child_active_states.is_empty() {
                (NodeClasses { classes: Vec::new(), class_maps: Vec::new() }, vec![u32::MAX; pending.descend_positions.len()])
            } else {
                let recursive_started_at = Instant::now();
                let result = build_node(pending.child, tokenizer, &pending.child_active_states, matched_terminals, node_terminal_ids, empty_terminals_id, is_end, byte_transitions, self_loop_bytes, canonical_state, terminal_sets, segment_cache, segment_outcome_tables, timings, stamp_gen, terminal_stamps, active_seen_gen, active_seen_stamps, active_seen_positions, parallel_depth.saturating_sub(1), parallel_min_active);
                timings.recursive_ms += elapsed_ms(recursive_started_at);
                let child_class_ids = pending.descend_positions.iter().map(|&pos| if pos == u32::MAX { u32::MAX } else { result.classes[pos as usize] }).collect();
                (result, child_class_ids)
            };
            child_data.push(ChildBuildData { outcomes: pending.outcomes, child_class_ids, reachable: pending.reachable, result });
        }
    }

    let signature_started_at = Instant::now();
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
            let mut buckets: FxHashMap<u64, Vec<SignatureEntry>> = FxHashMap::default();
            let mut next_class_id = 0u32;
            for (state_pos, &state) in active_states.iter().enumerate() {
                let node_terms = if node.has_token() { node_terminal_ids[state as usize] } else { empty_terminals_id };
                let mut hash = mix_signature_word(0, node_terms);
                for child in &child_data { hash = mix_signature_word(hash, child.outcomes[state_pos].terminals_id); hash = mix_signature_word(hash, child.child_class_ids[state_pos]); }
                let bucket = buckets.entry(hash).or_default();
                let mut found = false;
                for entry in bucket.iter() {
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
    timings.signature_hash_ms += elapsed_ms(signature_started_at);
    timings.classes_built += representative_states.len();

    let map_started_at = Instant::now();
    let mut class_maps = Vec::with_capacity(representative_states.len());
    for (&state, &state_pos) in representative_states.iter().zip(representative_state_positions.iter()) {
        let mut result = IntervalPossibleMatchMap::default();
        if node.has_token() {
            let token_id = node.token_id() as u32;
            for &terminal in terminal_sets.get(node_terminal_ids[state as usize]) { append_range(&mut result, terminal, (token_id, token_id)); }
        }
        for child in &child_data {
            for &terminal in terminal_sets.get(child.outcomes[state_pos].terminals_id) { append_ranges(&mut result, terminal, &child.reachable); }
            let child_class_id = child.child_class_ids[state_pos];
            if child_class_id != u32::MAX { merge_interval_maps(&mut result, child.result.class_maps[child_class_id as usize].as_ref()); }
        }
        normalize_interval_map(&mut result);
        class_maps.push(Arc::new(result));
    }
    timings.map_materialize_ms += elapsed_ms(map_started_at);
    NodeClasses { classes, class_maps }
}

pub(crate) fn collect_possible_matches_interval_trie_class_build_with_classes(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    entries: &[u32],
    canonical_state: Option<&[u32]>,
) -> (TrieClassBuildResult, PossibleMatchesProfile) {
    let matched_terminals: Vec<Box<[TerminalID]>> = (0..tokenizer.num_states()).map(|state| tokenizer.matched_terminals_iter(state).collect::<Vec<_>>().into_boxed_slice()).collect();
    let is_end: Vec<bool> = (0..tokenizer.num_states()).map(|state| tokenizer.is_end(state)).collect();
    let mut byte_transitions = vec![vec![u32::MAX; tokenizer.num_states() as usize]; 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        for (byte, &target) in dfa_state.transitions.iter() { byte_transitions[byte as usize][state_idx] = target; }
    }
    let self_loop_bytes: Vec<U8Set> = (0..tokenizer.num_states() as usize).map(|state_idx| {
        let dfa_state = &tokenizer.dfa.states()[state_idx];
        let mut bytes = U8Set::empty();
        for (byte, &target) in dfa_state.transitions.iter() { if target == state_idx as u32 { bytes.insert(byte); } }
        bytes
    }).collect();
    let mut terminal_sets = TerminalSetInterner::default();
    let empty_terminals_id = terminal_sets.intern_slice(&[]);
    let node_terminal_ids: Vec<u32> = matched_terminals.iter().map(|terminals| terminal_sets.intern_slice(terminals)).collect();
    let parallel_depth = std::env::var("GLRMASK_PM_ROOT_PARALLEL_DEPTH").ok().and_then(|v| v.parse::<u8>().ok()).unwrap_or(5);
    let parallel_min_active = std::env::var("GLRMASK_PM_PARALLEL_MIN_ACTIVE_STATES").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
    if profile_summary_enabled() {
        eprintln!("[glrmask/profile][trie_build_interval] root_parallel_children={} parallel_depth={} parallel_min_active_states={}", root.children().len(), parallel_depth, parallel_min_active);
    }
    let root_started_at = Instant::now();
    let mut timings = BuildTimings::default();
    let mut segment_cache: FxHashMap<Vec<u8>, usize> = FxHashMap::default();
    let mut segment_outcome_tables = Vec::<FxHashMap<u32, SegmentOutcome>>::new();
    let mut stamp_gen = 0u32;
    let mut terminal_stamps = vec![0u32; tokenizer.num_terminals as usize];
    let mut active_seen_gen = 0u32;
    let mut active_seen_stamps = vec![0u32; tokenizer.num_states() as usize];
    let mut active_seen_positions = vec![0u32; tokenizer.num_states() as usize];
    let root_result = build_node(root, tokenizer, entries, &matched_terminals, &node_terminal_ids, empty_terminals_id, &is_end, &byte_transitions, &self_loop_bytes, canonical_state, &mut terminal_sets, &mut segment_cache, &mut segment_outcome_tables, &mut timings, &mut stamp_gen, &mut terminal_stamps, &mut active_seen_gen, &mut active_seen_stamps, &mut active_seen_positions, parallel_depth, parallel_min_active);
    let root_compute_ms = elapsed_ms(root_started_at);
    if profile_summary_enabled() {
        eprintln!("[glrmask/profile][trie_build_interval_timings] segment_table_ms={:.3} signature_hash_ms={:.3} map_materialize_ms={:.3} child_active_ms={:.3} recursive_ms={:.3} reachable_interval_ms={:.3} child_precompute_ms={:.3} classes_built={}", timings.segment_table_ms, timings.signature_hash_ms, timings.map_materialize_ms, timings.child_active_ms, timings.recursive_ms, timings.reachable_interval_ms, timings.child_precompute_ms, timings.classes_built);
    }
    let profile = PossibleMatchesProfile { cache_entries: root_result.class_maps.len(), root_compute_ms, ..Default::default() };
    let mut state_classes = vec![u32::MAX; tokenizer.num_states() as usize];
    for (state_pos, &state) in entries.iter().enumerate() {
        if let Some(&class_id) = root_result.classes.get(state_pos) {
            if let Some(slot) = state_classes.get_mut(state as usize) { *slot = class_id; }
        }
    }
    (TrieClassBuildResult { state_classes, class_maps: root_result.class_maps.iter().cloned().collect() }, profile)
}
