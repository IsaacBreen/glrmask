//! Constraint-specific possible-match dense collector.
//!
//! Migrated from compiler::possible_matches so that the Constraint build path
//! does not depend on possible_matches.rs for its stored possible_matches.
//! Future constraint-specific optimizations belong here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::pm_profile::{PossibleMatchesProfile, elapsed_ms, profile_summary_enabled};
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
            let mut child_active_seen = vec![0u64; (tokenizer.num_states() as usize + 63) / 64];
            for segment_outcome in outcomes.iter() {
                let descend_end_state = if let Some(end_state) = segment_outcome.end_state {
                    if !is_end[end_state as usize]
                        && !subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                    {
                        let word = end_state as usize / 64;
                        let bit = 1u64 << (end_state % 64);
                        if child_active_seen[word] & bit == 0 {
                            child_active_seen[word] |= bit;
                            child_active_states.push(end_state);
                        }
                        end_state
                    } else {
                        u32::MAX
                    }
                } else {
                    u32::MAX
                };
                descend_end_states.push(descend_end_state);
            }
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
            let mut child_active_seen = vec![0u64; (tokenizer.num_states() as usize + 63) / 64];
            for segment_outcome in outcomes.iter() {
                let descend_end_state = if let Some(end_state) = segment_outcome.end_state {
                    if !is_end[end_state as usize]
                        && !subtree_bytes.is_subset(&self_loop_bytes[end_state as usize])
                    {
                        let word = end_state as usize / 64;
                        let bit = 1u64 << (end_state % 64);
                        if child_active_seen[word] & bit == 0 {
                            child_active_seen[word] |= bit;
                            child_active_states.push(end_state);
                        }
                        end_state
                    } else {
                        u32::MAX
                    }
                } else {
                    u32::MAX
                };
                descend_end_states.push(descend_end_state);
            }
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

