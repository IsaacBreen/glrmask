//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

pub(crate) mod max_length;

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Ranges key with pre-computed hash for O(1) HashMap lookups.
/// The hash is computed in the parallel traversal phase so the sequential
/// interning loop avoids re-hashing large range vectors.
#[derive(Clone)]
struct PreHashedRanges {
    hash: u64,
    ranges: Vec<(u32, u32)>,
}

impl PreHashedRanges {
    fn new(ranges: Vec<(u32, u32)>) -> Self {
        let mut h: u64 = ranges.len() as u64;
        for &(s, e) in &ranges {
            h = h.wrapping_mul(0x517cc1b727220a95) ^ ((s as u64) | ((e as u64) << 32));
        }
        Self { hash: h, ranges }
    }
}

impl PartialEq for PreHashedRanges {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.ranges == other.ranges
    }
}

impl Eq for PreHashedRanges {}

impl Hash for PreHashedRanges {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::compact::{compact_dwa_dimensions_fast, compact_dwa_dimensions_fast_with_stats};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::{
    LocalIdMapTerminalDwa, identity_original_to_local_state,
};
use crate::ds::weight::{Weight, shared_rangeset};
use crate::Vocab;

use super::l2p::equivalence_analysis::compat::TokenizerView;
use super::types::{TerminalColoring, compile_profile_enabled, debug_profile_enabled};

/// Maximum L1 equivalence class count before falling back to L2+.
///
/// When the tokenizer DFA has more than this many distinct equivalence classes
/// for the active L1 terminals, the L1 trie traversal becomes more expensive
/// than L2P's NWA-based approach.
pub(crate) const MAX_L1_TSIDS: usize = 50;

/// Quickly count L1 equivalence classes for the given active terminals.
///
/// Used by the partition builder to decide whether L1 should be attempted
/// *before* launching the parallel L1/L2P build, avoiding a wasteful
/// L2P double-build when L1 would be skipped.
pub(crate) fn count_l1_equivalence_classes(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    active_terminals: &[bool],
) -> usize {
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let tokenizer_view = TokenizerView::new_filtered(tokenizer, active_terminals);
    let token_bytes: Vec<&[u8]> = vocab.entries.values().map(|b| b.as_slice()).collect();
    let equiv_mapping = max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
    );
    let mut seen = rustc_hash::FxHashSet::default();
    for &rep in &equiv_mapping {
        seen.insert(rep);
    }
    seen.len()
}

/// Build an L1 id_map and terminal DWA for the given vocab and terminal set.
///
/// Uses max-length state equivalence and an identity vocab map, then traverses
/// the vocab tree to accumulate `terminal -> Weight` before building the final
/// 2-state DWA directly.
///
/// Returns `None` if the vocab is empty or no terminal matches exist.
/// The caller should pre-check `count_l1_equivalence_classes()` and merge
/// L1 terminals into L2+ when the count exceeds `MAX_L1_TSIDS`.
pub(crate) fn build_l1_id_map_and_terminal_dwa(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    _terminal_coloring: &TerminalColoring,
    _use_terminal_coloring: bool,
    _ignore_terminal: Option<TerminalID>,
    grammar: &AnalyzedGrammar,
    active_terminals: &[bool],
) -> Option<LocalIdMapTerminalDwa> {
    if vocab.is_empty() {
        return None;
    }

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let (mut id_map, sorted_entries, state_to_rep, id_map_profile) = build_l1_id_map(tokenizer, vocab, active_terminals);
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let num_terminals = grammar.num_terminals as u32;
    let dwa_started_at = Instant::now();
    let (mut dwa, terminal_profile) = build_l1_terminal_dwa(
        tokenizer,
        sorted_entries,
        &mut id_map,
        &state_to_rep,
        num_terminals,
        active_terminals,
    )?;
    let terminal_build_ms = dwa_started_at.elapsed().as_secs_f64() * 1000.0;

    let profiling = compile_profile_enabled() || debug_profile_enabled();
    let tsids_before_compact = id_map.num_tsids();
    let tokens_before_compact = id_map.num_internal_tokens();

    let compact_started_at = Instant::now();
    let compact_report = if profiling {
        compact_dwa_dimensions_fast_with_stats(&mut dwa, &mut id_map)
    } else {
        compact_dwa_dimensions_fast(&mut dwa, &mut id_map)
    };
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;

    if profiling {
        let stats_str = if let Some(stats) = compact_report.profile_stats {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={} compact_weight_ranges_before={} compact_weight_ranges_after={} compact_token_ranges_before={} compact_token_ranges_after={}",
                stats.tsids_before, stats.tsids_after,
                stats.tokens_before, stats.tokens_after,
                stats.weight_ranges_before, stats.weight_ranges_after,
                stats.token_ranges_before, stats.token_ranges_after,
            )
        } else {
            format!(
                " compact_tsids_before={} compact_tsids_after={} compact_tokens_before={} compact_tokens_after={}",
                tsids_before_compact, id_map.num_tsids(),
                tokens_before_compact, id_map.num_internal_tokens(),
            )
        };
        eprintln!(
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} state_equiv_ms={:.3} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} token_set_intern_ms={:.3} tsid_profile_merge_ms={:.3} tsid_profile_merge_before={} tsid_profile_merge_after={} vocab_tree_traversal_ms={:.3} direct_terminal_dwa_ms={:.3} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}{}",
            partition_label,
            vocab.entries.len(),
            id_map.num_tsids(),
            id_map_profile.state_equiv_ms,
            id_map_profile.token_identity_map_ms,
            id_map_ms,
            terminal_profile.internal_vocab_ms,
            terminal_profile.vocab_tree_build_ms,
            terminal_profile.state_seed_ms,
            terminal_profile.token_set_intern_ms,
            terminal_profile.tsid_profile_merge_ms,
            terminal_profile.tsid_profile_merge_before,
            terminal_profile.tsid_profile_merge_after,
            terminal_profile.vocab_tree_traversal_ms,
            terminal_profile.direct_terminal_dwa_ms,
            terminal_build_ms,
            compact_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            stats_str,
        );
    }

    // Fast iteration: exit after L1 for a specific partition
    if let Ok(exit_label) = std::env::var("GLRMASK_EXIT_AFTER_L1") {
        if exit_label == partition_label {
            eprintln!("[glrmask/debug] EXIT_AFTER_L1={} triggered.", partition_label);
            std::process::exit(0);
        }
    }

    Some(LocalIdMapTerminalDwa {
        id_map,
        dwa,
        original_to_local_state: identity_original_to_local_state(tokenizer.num_states() as usize),
    })
}

fn build_l1_id_map<'a>(tokenizer: &Tokenizer, vocab: &'a Vocab, active_terminals: &[bool]) -> (InternalIdMap, Vec<(u32, &'a [u8])>, Vec<u32>, L1IdMapProfile) {
    let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

    // Max-length bounded state equivalence: merge DFA states that behave
    // identically when only tokens up to the max vocab token length are
    // considered. Filtering by active_terminals lets us also merge states
    // that differ only by inactive terminal finalizers/futures.
    let state_equiv_started_at = Instant::now();
    let tokenizer_view = TokenizerView::new_filtered(tokenizer, active_terminals);
    let token_bytes: Vec<&[u8]> = vocab
        .entries
        .values()
        .map(|b| b.as_slice())
        .collect();
    let equiv_mapping = max_length::find_state_equivalence_classes_byte_restricted(
        &tokenizer_view,
        &token_bytes,
        &states,
    );
    // Build representative → internal_id mapping
    let mut rep_to_internal: FxHashMap<usize, u32> = FxHashMap::default();
    let mut state_original_to_internal = vec![u32::MAX; states.len()];
    let mut state_representatives = Vec::new();
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        let state_id = states[i];
        let internal_id = *rep_to_internal.entry(rep).or_insert_with(|| {
            let id = state_representatives.len() as u32;
            state_representatives.push(rep as u32);
            id
        });
        state_original_to_internal[state_id] = internal_id;
    }
    // Build state_to_rep: original_state → representative_state (for trie traversal)
    let mut state_to_rep = vec![0u32; states.len()];
    for (i, &rep) in equiv_mapping.iter().enumerate() {
        state_to_rep[states[i]] = rep as u32;
    }
    let state_equiv_ms = state_equiv_started_at.elapsed().as_secs_f64() * 1000.0;

    // Sort token IDs by byte content so internal IDs follow DFS traversal order
    // in the VocabPrefixTree. This makes reachable_token_ids() contiguous ranges,
    // enabling O(1) RangeSetBlaze unions during self-loop optimization.
    let token_identity_started_at = Instant::now();
    let mut token_id_bytes: Vec<(u32, &[u8])> = vocab
        .entries
        .iter()
        .map(|(&id, bytes)| (id, bytes.as_slice()))
        .collect();
    token_id_bytes.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));
    let mut token_original_to_internal = vec![u32::MAX; vocab.max_token_id() as usize + 1];
    let token_ids_sorted: Vec<u32> = token_id_bytes
        .iter()
        .enumerate()
        .map(|(internal_id, &(original_id, _))| {
            token_original_to_internal[original_id as usize] = internal_id as u32;
            original_id
        })
        .collect();
    let token_identity_map_ms = token_identity_started_at.elapsed().as_secs_f64() * 1000.0;

    (
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                state_original_to_internal,
                state_representatives.len() as u32,
                state_representatives,
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_with_representatives(
                token_original_to_internal,
                token_ids_sorted.len() as u32,
                token_ids_sorted,
            ),
        },
        token_id_bytes,
        state_to_rep,
        L1IdMapProfile {
            state_equiv_ms,
            token_identity_map_ms,
        },
    )
}

fn build_l1_terminal_dwa(
    tokenizer: &Tokenizer,
    sorted_entries: Vec<(u32, &[u8])>,
    id_map: &mut InternalIdMap,
    state_to_rep: &[u32],
    num_terminals: u32,
    active_terminals: &[bool],
) -> Option<(DWA, L1TerminalBuildProfile)> {
    let total_started_at = std::time::Instant::now();
    let internal_vocab_ms = 0.0;

    if sorted_entries.is_empty() {
        return None;
    }

    let vocab_tree_build_ms = 0.0;

    let state_seed_started_at = Instant::now();
    let mut states_to_initial_tsids = FxHashMap::<u32, Vec<u32>>::default();
    for (internal_tsid, representative_state) in id_map.tokenizer_states.iter_representative_ids().enumerate() {
        states_to_initial_tsids
            .entry(representative_state)
            .or_default()
            .push(internal_tsid as u32);
    }
    let state_seed_ms = state_seed_started_at.elapsed().as_secs_f64() * 1000.0;

    // Batch simulation: for each unique start state, simulate all tokens through
    // the DFA and accumulate end_state_rep → (tsid → token_ids).
    // Parallelized across start states using rayon.
    let num_dfa_states = tokenizer.num_states() as usize;

    let traversal_started_at = Instant::now();

    let flat_trans = build_flat_transition_table(tokenizer);

    // Parallel traversal: each start_state processed independently.
    // Each (end_rep, tsid) pair is unique across start groups since TSIDs
    // partition deterministically into start groups. We exploit this by using
    // Arc from the start and skipping merging entirely.
    let start_states_list: Vec<(&u32, &Vec<u32>)> = states_to_initial_tsids.iter().collect();
    let mut empty_token_indices = Vec::<usize>::new();
    let mut token_indices_by_first_byte = vec![Vec::<usize>::new(); 256];
    for (internal_token_id, &(_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
        if let Some(&first_byte) = token_bytes.first() {
            token_indices_by_first_byte[first_byte as usize].push(internal_token_id);
        } else {
            empty_token_indices.push(internal_token_id);
        }
    }

    // Compute suffix_subtree_bytes per first_byte: the set of all bytes
    // appearing in suffixes (bytes[1..]) of tokens starting with that byte.
    // Used for self-loop optimization in the walk cache.
    let mut suffix_subtree_bytes: Vec<[u64; 4]> = vec![[0u64; 4]; 256];
    for &(_original_id, token_bytes) in &sorted_entries {
        if let Some(&first) = token_bytes.first() {
            for &byte in &token_bytes[1..] {
                suffix_subtree_bytes[first as usize][byte as usize >> 6] |= 1u64 << (byte & 63);
            }
        }
    }

    let phase1_wall_ms: f64;
    let skipped_walks = std::sync::atomic::AtomicUsize::new(0);
    let skipped_tokens = std::sync::atomic::AtomicUsize::new(0);
    let per_thread_results: Vec<Vec<(u32, u32, PreHashedRanges)>> = {
        // Walk cache: compute once per unique (first_byte, target) and cache
        // the raw merged ranges. Self-loop optimization: if the target state
        // has self-loops on all suffix bytes, all tokens end at the target
        // state and the walk can be skipped entirely.
        let dead = u32::MAX;

        // Phase 1: Identify unique (first_byte, target) pairs
        let mut unique_targets: FxHashMap<(u8, u32), ()> = FxHashMap::default();
        for (&start_state, _) in &states_to_initial_tsids {
            for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                if token_ids.is_empty() { continue; }
                let target = flat_trans[start_state as usize * 256 + byte];
                if target != dead {
                    unique_targets.entry((byte as u8, target)).or_default();
                }
            }
        }
        let unique_walk_keys: Vec<(u8, u32)> = unique_targets.into_keys().collect();

        // Precompute self-loop mask per target state.
        let mut self_loop_masks: FxHashMap<u32, [u64; 4]> = FxHashMap::default();
        for &(_, target) in &unique_walk_keys {
            self_loop_masks.entry(target).or_insert_with(|| {
                let mut mask = [0u64; 4];
                let base = target as usize * 256;
                for byte in 0..=255u8 {
                    if flat_trans[base + byte as usize] == target {
                        mask[byte as usize >> 6] |= 1u64 << (byte & 63);
                    }
                }
                mask
            });
        }

        // Parallel walk per unique (first_byte, target). Store raw merged ranges.
        let walk_cache: FxHashMap<(u8, u32), Vec<(u32, Vec<(u32, u32)>)>> = unique_walk_keys
            .par_iter()
            .map(|&(first_byte, first_target)| {
                let token_ids = &token_indices_by_first_byte[first_byte as usize];

                // Self-loop skip: if the target state has self-loops on all
                // suffix bytes for this first_byte, all tokens end at first_target.
                let mask = &self_loop_masks[&first_target];
                let subtree = &suffix_subtree_bytes[first_byte as usize];
                let can_skip = (subtree[0] & !mask[0]) == 0
                    && (subtree[1] & !mask[1]) == 0
                    && (subtree[2] & !mask[2]) == 0
                    && (subtree[3] & !mask[3]) == 0;

                if can_skip {
                    skipped_walks.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    skipped_tokens.fetch_add(token_ids.len(), std::sync::atomic::Ordering::Relaxed);
                    let end_rep = state_to_rep[first_target as usize];
                    let first = *token_ids.first().unwrap() as u32;
                    let last = *token_ids.last().unwrap() as u32;
                    return ((first_byte, first_target), vec![(end_rep, vec![(first, last)])]);
                }

                let mut end_rep_token_ranges = FxHashMap::<u32, Vec<(u32, u32)>>::default();

                let mut previous_suffix: &[u8] = &[];
                let mut suffix_states: Vec<u32> = vec![first_target];
                for &internal_token_id in token_ids {
                    let token_bytes = sorted_entries[internal_token_id].1;
                    let suffix_bytes = &token_bytes[1..];
                    let lcp_len = common_prefix_len(previous_suffix, suffix_bytes);
                    suffix_states.truncate(lcp_len + 1);
                    let mut state = *suffix_states.last().unwrap();
                    if state == dead {
                        suffix_states.resize(suffix_bytes.len() + 1, dead);
                    } else {
                        for &byte in &suffix_bytes[lcp_len..] {
                            state = flat_trans[state as usize * 256 + byte as usize];
                            suffix_states.push(state);
                            if state == dead {
                                suffix_states.resize(suffix_bytes.len() + 1, dead);
                                break;
                            }
                        }
                    }
                    let final_state = suffix_states[suffix_bytes.len()];
                    if final_state != dead {
                        let end_rep = state_to_rep[final_state as usize];
                        append_token_id_range(
                            end_rep_token_ranges.entry(end_rep).or_default(),
                            internal_token_id as u32,
                        );
                    }
                    previous_suffix = suffix_bytes;
                }

                let results: Vec<(u32, Vec<(u32, u32)>)> = end_rep_token_ranges
                    .into_iter()
                    .map(|(end_rep, mut ranges)| {
                        merge_ranges_in_place(&mut ranges);
                        (end_rep, ranges)
                    })
                    .collect();
                ((first_byte, first_target), results)
            })
            .collect();

        let phase1_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;
        phase1_wall_ms = phase1_ms;

        // Debug: count unique transition signatures among start_states.
        // Signature = sorted Vec of (first_byte, target) for non-dead transitions.
        if debug_profile_enabled() {
            let dead2 = u32::MAX;
            let mut sig_counts: FxHashMap<Vec<(u8, u32)>, usize> = FxHashMap::default();
            for (&start_state, _) in &states_to_initial_tsids {
                let mut sig: Vec<(u8, u32)> = Vec::new();
                for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                    if token_ids.is_empty() { continue; }
                    let target = flat_trans[start_state as usize * 256 + byte];
                    if target != dead2 {
                        sig.push((byte as u8, target));
                    }
                }
                *sig_counts.entry(sig).or_default() += 1;
            }
            let unique_sigs = sig_counts.len();
            let max_group = sig_counts.values().max().copied().unwrap_or(0);
            let groups_gt1: usize = sig_counts.values().filter(|&&v| v > 1).count();
            eprintln!(
                "[glrmask/debug][l1_signatures] total_start_states={} unique_signatures={} max_group_size={} groups_with_multiple={}",
                start_states_list.len(), unique_sigs, max_group, groups_gt1
            );
        }

        // Phase 2: For each start_state, collect cached walk results across all
        // first bytes, merge per end_rep, build PreHashedRanges.
        // Token IDs are assigned in byte-sorted order and we iterate
        // first bytes 0→255, so accumulated ranges per end_rep are
        // already sorted. We skip the sort and only do the linear merge.
        start_states_list
            .par_iter()
            .map(|&(&start_state, ref initial_tsids)| {
                let mut end_rep_token_ranges = FxHashMap::<u32, Vec<(u32, u32)>>::default();

                // Empty tokens: end_rep = start_rep
                // (Empty byte sequences sort first, so these IDs are smallest)
                let start_rep = state_to_rep[start_state as usize];
                for &internal_token_id in &empty_token_indices {
                    append_token_id_range(
                        end_rep_token_ranges.entry(start_rep).or_default(),
                        internal_token_id as u32,
                    );
                }

                // Collect cached walk results in byte order (0→255).
                for (byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                    if token_ids.is_empty() { continue; }
                    let target = flat_trans[start_state as usize * 256 + byte];
                    if target == dead { continue; }
                    if let Some(results) = walk_cache.get(&(byte as u8, target)) {
                        for (end_rep, ranges) in results {
                            end_rep_token_ranges.entry(*end_rep).or_default()
                                .extend_from_slice(ranges);
                        }
                    }
                }

                // Build one entry per (end_rep, tsid).
                let mut result: Vec<(u32, u32, PreHashedRanges)> = Vec::new();
                for (end_rep, mut token_ranges) in end_rep_token_ranges {
                    merge_sorted_ranges_in_place(&mut token_ranges);
                    let prehashed_key = PreHashedRanges::new(token_ranges);
                    for &tsid in initial_tsids.iter() {
                        result.push((end_rep, tsid, prehashed_key.clone()));
                    }
                }
                result
            })
            .collect()
    };

    let phase2_wall_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0 - phase1_wall_ms;

    // Sort-based intern: sort entries by hash, find hash-group boundaries,
    // then verify and build Arcs in parallel.
    let token_set_intern_started_at = Instant::now();

    // Flatten all thread results into a single Vec.
    let mut all_entries: Vec<(u32, u32, PreHashedRanges)> =
        per_thread_results.into_iter().flatten().collect();

    // Sort by hash (fast u64 comparison). Equal hashes → same group candidate.
    all_entries.sort_unstable_by_key(|entry| entry.2.hash);

    // Find hash-group boundaries (sequential, fast — u64 comparison only).
    let mut hash_group_starts: Vec<usize> = vec![0];
    for k in 1..all_entries.len() {
        if all_entries[k].2.hash != all_entries[k - 1].2.hash {
            hash_group_starts.push(k);
        }
    }
    hash_group_starts.push(all_entries.len());

    // Process hash groups in parallel. For each group, verify ranges equality
    // (handles hash collisions), build Arc, and collect (end_rep, tsid, Arc).
    let group_results: Vec<Vec<(usize, u32, Arc<RangeSetBlaze<u32>>)>> = hash_group_starts
        .par_windows(2)
        .map(|w| {
            let start = w[0];
            let end = w[1];
            let mut out = Vec::new();
            // Within [start..end): same hash. Sub-group by ranges for correctness.
            let mut sub_start = start;
            while sub_start < end {
                let mut sub_end = sub_start + 1;
                while sub_end < end
                    && all_entries[sub_end].2.ranges == all_entries[sub_start].2.ranges
                {
                    sub_end += 1;
                }
                let arc: Arc<RangeSetBlaze<u32>> = Arc::new(
                    all_entries[sub_start]
                        .2
                        .ranges
                        .iter()
                        .map(|&(s, e)| s..=e)
                        .collect(),
                );
                for k in sub_start..sub_end {
                    out.push((
                        all_entries[k].0 as usize,
                        all_entries[k].1,
                        arc.clone(),
                    ));
                }
                sub_start = sub_end;
            }
            out
        })
        .collect();

    // Merge parallel results into deferred_arced (sequential).
    let mut deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        vec![Vec::new(); num_dfa_states];
    // Count unique range sets: each distinct Arc pointer = one unique set.
    let unique_range_set_count;
    {
        let mut seen_arcs: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
        for result in &group_results {
            for &(_, _, ref arc) in result {
                seen_arcs.insert(Arc::as_ptr(arc) as usize);
            }
        }
        unique_range_set_count = seen_arcs.len();
    }
    for result in group_results {
        for (end_rep, tsid, arc) in result {
            deferred_arced[end_rep].push((tsid, arc));
        }
    }

    let token_set_intern_ms = token_set_intern_started_at.elapsed().as_secs_f64() * 1000.0;
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;
    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l1_intern_detail] entries={} unique_sets={} hash_groups={} intern_ms={:.1}",
            all_entries.len(), unique_range_set_count, hash_group_starts.len() - 1, token_set_intern_ms,
        );
        eprintln!(
            "[glrmask/debug][l1_traversal] start_states={} phase1_walk_ms={:.1} phase2_assembly_ms={:.1} intern_ms={:.1} unique_range_sets={} skipped_walks={} skipped_tokens={} total_vt_ms={:.1}",
            start_states_list.len(), phase1_wall_ms, phase2_wall_ms, token_set_intern_ms,
            unique_range_set_count,
            skipped_walks.load(std::sync::atomic::Ordering::Relaxed),
            skipped_tokens.load(std::sync::atomic::Ordering::Relaxed),
            traversal_ms,
        );
    }

    let tsid_profile_merge_started_at = Instant::now();
    let tsid_profile_merge_before = id_map.num_tsids() as usize;
    let tsid_profile_merge_report = merge_deferred_equivalent_tsids(id_map, &mut deferred_arced);
    let tsid_profile_merge_after = tsid_profile_merge_report.tsids_after;
    let tsid_profile_merge_ms = tsid_profile_merge_started_at.elapsed().as_secs_f64() * 1000.0;
    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][l1_tsid_profile_merge] before={} after={} unique_arc_token_sets={} unique_range_token_sets={} profile_build_ms={:.3} group_ms={:.3} remap_ms={:.3} total_ms={:.3}",
            tsid_profile_merge_before,
            tsid_profile_merge_after,
            tsid_profile_merge_report.unique_arc_token_sets,
            tsid_profile_merge_report.unique_range_token_sets,
            tsid_profile_merge_report.profile_build_ms,
            tsid_profile_merge_report.group_ms,
            tsid_profile_merge_report.remap_ms,
            tsid_profile_merge_ms,
        );
    }

    let distribute_started_at = Instant::now();
    let arc_wrap_ms = 0.0; // Arc wrapping is now done inside the traversal

    // Build terminal → sorted deduped set of active DFA states (mapped to representatives)
    let inverse_started_at = Instant::now();
    let mut terminal_to_active_states: Vec<Vec<u32>> = vec![Vec::new(); num_terminals as usize];
    for state in 0..num_dfa_states {
        let state_u32 = state as u32;
        let rep = state_to_rep[state];
        for tid in tokenizer.dfa.finalizers(state_u32).iter() {
            if active_terminals.get(tid).copied().unwrap_or(false) {
                terminal_to_active_states[tid].push(rep);
            }
        }
        for tid in tokenizer.tokens_accessible_from_state(state_u32).iter() {
            if active_terminals.get(tid).copied().unwrap_or(false) {
                terminal_to_active_states[tid].push(rep);
            }
        }
    }
    for states in &mut terminal_to_active_states {
        states.sort_unstable();
        states.dedup();
    }
    let inverse_map_ms = inverse_started_at.elapsed().as_secs_f64() * 1000.0;

    // Pre-compute per-TSID full token set unions and contributing end_reps.
    // For each terminal, TSIDs whose contributing end_reps are all active reuse
    // the precomputed Arc; only TSIDs with some inactive end_reps are recomputed.
    let merge_started_at = Instant::now();

    let num_tsids = id_map.num_tsids() as usize;

    // Build per-TSID: full ranges union + list of contributing end_reps
    let mut tsid_full_ranges: Vec<Vec<(u32, u32)>> = (0..num_tsids).map(|_| Vec::new()).collect();
    let mut tsid_end_reps: Vec<Vec<u32>> = (0..num_tsids).map(|_| Vec::new()).collect();
    for (end_rep, entries) in deferred_arced.iter().enumerate() {
        for &(tsid, ref arc) in entries {
            tsid_end_reps[tsid as usize].push(end_rep as u32);
            for r in arc.ranges() {
                tsid_full_ranges[tsid as usize].push((*r.start(), *r.end()));
            }
        }
    }
    for reps in &mut tsid_end_reps {
        reps.sort_unstable();
        reps.dedup();
    }
    // Sort/merge ranges per TSID → full union Arcs (parallel)
    let tsid_full_arcs: Vec<Option<Arc<RangeSetBlaze<u32>>>> = tsid_full_ranges
        .par_iter_mut()
        .map(|ranges| {
            if ranges.is_empty() { return None; }
            ranges.sort_unstable();
            let mut w = 0;
            for r in 1..ranges.len() {
                if ranges[r].0 <= ranges[w].1.saturating_add(1) {
                    ranges[w].1 = ranges[w].1.max(ranges[r].1);
                } else { w += 1; ranges[w] = ranges[r]; }
            }
            ranges.truncate(w + 1);
            Some(shared_rangeset(ranges.iter().map(|&(s, e)| s..=e).collect()))
        })
        .collect();
    drop(tsid_full_ranges);

    // Group terminals by active_states to deduplicate identical computation
    let mut active_tids: Vec<usize> = (0..terminal_to_active_states.len())
        .filter(|&i| !terminal_to_active_states[i].is_empty())
        .collect();
    active_tids.sort_unstable_by(|&a, &b|
        terminal_to_active_states[a].cmp(&terminal_to_active_states[b]));
    let mut unique_groups: Vec<Vec<usize>> = Vec::new();
    for &tid in &active_tids {
        if let Some(last) = unique_groups.last_mut() {
            if terminal_to_active_states[last[0]] == terminal_to_active_states[tid] {
                last.push(tid); continue;
            }
        }
        unique_groups.push(vec![tid]);
    }

    // Compute weights per unique group in parallel
    let group_results: Vec<Option<(Vec<usize>, Weight)>> = unique_groups
        .par_iter()
        .map(|tids| {
            let active_states = &terminal_to_active_states[tids[0]];
            let active_set: rustc_hash::FxHashSet<u32> =
                active_states.iter().copied().collect();

            let mut weight_entries: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
            let mut affected_tsids: Vec<u32> = Vec::new();

            for tsid in 0..num_tsids as u32 {
                let reps = &tsid_end_reps[tsid as usize];
                if reps.is_empty() { continue; }
                if reps.iter().all(|r| active_set.contains(r)) {
                    if let Some(ref arc) = tsid_full_arcs[tsid as usize] {
                        weight_entries.push((tsid, Arc::clone(arc)));
                    }
                } else if reps.iter().any(|r| active_set.contains(r)) {
                    affected_tsids.push(tsid);
                }
            }

            // Recompute affected TSIDs from active end_reps only
            if !affected_tsids.is_empty() {
                let affected_set: rustc_hash::FxHashSet<u32> =
                    affected_tsids.iter().copied().collect();
                let mut tsid_ranges: Vec<Vec<(u32, u32)>> =
                    (0..num_tsids).map(|_| Vec::new()).collect();
                for &state in active_states {
                    for &(tsid, ref arc) in &deferred_arced[state as usize] {
                        if !affected_set.contains(&tsid) { continue; }
                        for r in arc.ranges() {
                            tsid_ranges[tsid as usize].push((*r.start(), *r.end()));
                        }
                    }
                }
                for &tsid in &affected_tsids {
                    let slot = &mut tsid_ranges[tsid as usize];
                    if slot.is_empty() { continue; }
                    slot.sort_unstable();
                    let mut w = 0;
                    for r in 1..slot.len() {
                        if slot[r].0 <= slot[w].1.saturating_add(1) {
                            slot[w].1 = slot[w].1.max(slot[r].1);
                        } else { w += 1; slot[w] = slot[r]; }
                    }
                    slot.truncate(w + 1);
                    weight_entries.push((tsid, shared_rangeset(
                        slot.iter().map(|&(s, e)| s..=e).collect())));
                }
                weight_entries.sort_unstable_by_key(|&(t, _)| t);
            }

            if weight_entries.is_empty() { return None; }
            let weight = Weight::from_per_tsid_shared(
                weight_entries.iter().map(|(t, a)| (*t, Arc::clone(a))));
            if weight.is_empty() { return None; }
            Some((tids.clone(), weight))
        })
        .collect();

    // Sequential DWA construction from grouped results
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    let mut num_transitions = 0usize;

    for result in group_results.into_iter().flatten() {
        let (tids, weight) = result;
        for &tid in &tids {
            dwa.add_transition(dwa.start_state, tid as i32, end_state, weight.clone());
            num_transitions += 1;
        }
    }

    if num_transitions == 0 {
        return None;
    }

    let merge_ms = merge_started_at.elapsed().as_secs_f64() * 1000.0;
    let direct_terminal_dwa_ms = merge_ms;
    let distribute_ms = distribute_started_at.elapsed().as_secs_f64() * 1000.0;
    let vocab_tree_traversal_ms = traversal_ms;

    if debug_profile_enabled() {
        eprintln!(
            "[glrmask/debug][terminal_dwa] partition_build_l1_batch vocab={} tsids={} transitions={} traversal_ms={:.1} arc_wrap_ms={:.1} inverse_map_ms={:.1} merge_ms={:.1} distribute_ms={:.1} total_ms={:.1}",
            sorted_entries.len(),
            id_map.num_tsids(),
            num_transitions,
            traversal_ms,
            arc_wrap_ms,
            inverse_map_ms,
            merge_ms,
            distribute_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((
        dwa,
        L1TerminalBuildProfile {
            internal_vocab_ms,
            vocab_tree_build_ms,
            state_seed_ms,
            token_set_intern_ms,
            tsid_profile_merge_ms,
            tsid_profile_merge_before,
            tsid_profile_merge_after,
            vocab_tree_traversal_ms,
            direct_terminal_dwa_ms,
        },
    ))
}

fn build_flat_transition_table(tokenizer: &Tokenizer) -> Vec<u32> {
    let dead = u32::MAX;
    let mut flat_trans = vec![dead; tokenizer.num_states() as usize * 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        let base = state_idx * 256;
        for (byte, &target) in dfa_state.transitions.iter() {
            flat_trans[base + byte as usize] = target;
        }
    }
    flat_trans
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    let limit = left.len().min(right.len());
    let mut index = 0usize;
    while index < limit && left[index] == right[index] {
        index += 1;
    }
    index
}

fn append_token_id_range(token_ranges: &mut Vec<(u32, u32)>, token_id: u32) {
    if let Some((_, end)) = token_ranges.last_mut() {
        if end.saturating_add(1) == token_id {
            *end = token_id;
            return;
        }
    }
    token_ranges.push((token_id, token_id));
}

fn collect_l1_root_ranges_by_first_byte_lcp(
    start_state: u32,
    sorted_entries: &[(u32, &[u8])],
    empty_token_indices: &[usize],
    token_indices_by_first_byte: &[Vec<usize>],
    flat_trans: &[u32],
    state_to_rep: &[u32],
    end_rep_token_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
) {
    let dead = u32::MAX;
    let start_rep = state_to_rep[start_state as usize];
    for &internal_token_id in empty_token_indices {
        append_token_id_range(
            end_rep_token_ranges.entry(start_rep).or_default(),
            internal_token_id as u32,
        );
    }

    for (first_byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
        if token_ids.is_empty() {
            continue;
        }

        let first_target = flat_trans[start_state as usize * 256 + first_byte];
        if first_target == dead {
            continue;
        }

        let mut previous_suffix: &[u8] = &[];
        let mut suffix_states = vec![first_target];
        for &internal_token_id in token_ids {
            let token_bytes = sorted_entries[internal_token_id].1;
            let suffix_bytes = &token_bytes[1..];
            let lcp_len = common_prefix_len(previous_suffix, suffix_bytes);
            suffix_states.truncate(lcp_len + 1);

            let mut state = *suffix_states.last().unwrap_or(&first_target);
            if state == dead {
                suffix_states.resize(suffix_bytes.len() + 1, dead);
            } else {
                for &byte in &suffix_bytes[lcp_len..] {
                    state = flat_trans[state as usize * 256 + byte as usize];
                    suffix_states.push(state);
                    if state == dead {
                        suffix_states.resize(suffix_bytes.len() + 1, dead);
                        break;
                    }
                }
            }

            let final_state = suffix_states[suffix_bytes.len()];
            if final_state != dead {
                let end_rep = state_to_rep[final_state as usize];
                append_token_id_range(
                    end_rep_token_ranges.entry(end_rep).or_default(),
                    internal_token_id as u32,
                );
            }

            previous_suffix = suffix_bytes;
        }
    }
}

fn merge_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {
    if ranges.is_empty() {
        return;
    }

    ranges.sort_unstable();
    let mut write_index = 0usize;
    for read_index in 1..ranges.len() {
        if ranges[read_index].0 <= ranges[write_index].1.saturating_add(1) {
            ranges[write_index].1 = ranges[write_index].1.max(ranges[read_index].1);
        } else {
            write_index += 1;
            ranges[write_index] = ranges[read_index];
        }
    }
    ranges.truncate(write_index + 1);
}

/// Like merge_ranges_in_place but assumes the input is already sorted.
/// Used in Phase 2 assembly where byte-order iteration guarantees sorted IDs.
fn merge_sorted_ranges_in_place(ranges: &mut Vec<(u32, u32)>) {
    if ranges.len() <= 1 {
        return;
    }
    let mut write_index = 0usize;
    for read_index in 1..ranges.len() {
        if ranges[read_index].0 <= ranges[write_index].1.saturating_add(1) {
            ranges[write_index].1 = ranges[write_index].1.max(ranges[read_index].1);
        } else {
            write_index += 1;
            ranges[write_index] = ranges[read_index];
        }
    }
    ranges.truncate(write_index + 1);
}

fn merge_deferred_equivalent_tsids(
    id_map: &mut InternalIdMap,
    deferred_arced: &mut [Vec<(u32, Arc<RangeSetBlaze<u32>>)>],
) -> L1TsidProfileMergeReport {
    let num_tsids = id_map.num_tsids() as usize;
    if num_tsids <= 1 {
        return L1TsidProfileMergeReport {
            tsids_after: num_tsids,
            unique_arc_token_sets: 0,
            unique_range_token_sets: 0,
            profile_build_ms: 0.0,
            group_ms: 0.0,
            remap_ms: 0.0,
        };
    }

    let profile_build_started_at = Instant::now();
    let mut profiles = vec![Vec::<(u32, u32)>::new(); num_tsids];
    let mut token_ctx_by_arc = FxHashMap::<usize, u32>::default();
    let mut next_token_ctx = 0u32;
    for (end_rep, entries) in deferred_arced.iter().enumerate() {
        for &(tsid, ref token_set) in entries {
            let arc_ptr = Arc::as_ptr(token_set) as usize;
            let token_ctx = *token_ctx_by_arc.entry(arc_ptr).or_insert_with(|| {
                let ctx = next_token_ctx;
                next_token_ctx += 1;
                ctx
            });
            profiles[tsid as usize].push((end_rep as u32, token_ctx));
        }
    }
    let profile_build_ms = profile_build_started_at.elapsed().as_secs_f64() * 1000.0;

    let group_started_at = Instant::now();
    let mut sorted_tsids: Vec<usize> = (0..num_tsids).collect();
    sorted_tsids.sort_by(|&left, &right| profiles[left].cmp(&profiles[right]));

    let mut tsid_perm = vec![0u32; num_tsids];
    let mut new_count = 1usize;
    tsid_perm[sorted_tsids[0]] = 0;
    for pair in sorted_tsids.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        if profiles[previous] != profiles[current] {
            new_count += 1;
        }
        tsid_perm[current] = (new_count - 1) as u32;
    }
    let group_ms = group_started_at.elapsed().as_secs_f64() * 1000.0;

    if new_count == num_tsids {
        return L1TsidProfileMergeReport {
            tsids_after: num_tsids,
            unique_arc_token_sets: token_ctx_by_arc.len(),
            unique_range_token_sets: token_ctx_by_arc.len(),
            profile_build_ms,
            group_ms,
            remap_ms: 0.0,
        };
    }

    let remap_started_at = Instant::now();
    apply_tsid_perm_to_id_map(&mut id_map.tokenizer_states, &tsid_perm, new_count);
    remap_deferred_arced_tsids(deferred_arced, &tsid_perm);
    let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    L1TsidProfileMergeReport {
        tsids_after: new_count,
        unique_arc_token_sets: token_ctx_by_arc.len(),
        unique_range_token_sets: token_ctx_by_arc.len(),
        profile_build_ms,
        group_ms,
        remap_ms,
    }
}

fn remap_deferred_arced_tsids(
    deferred_arced: &mut [Vec<(u32, Arc<RangeSetBlaze<u32>>)>],
    tsid_perm: &[u32],
) {
    for entries in deferred_arced {
        if entries.is_empty() {
            continue;
        }

        let mut remapped: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = std::mem::take(entries)
            .into_iter()
            .map(|(tsid, token_set)| (tsid_perm[tsid as usize], token_set))
            .collect();
        remapped.sort_unstable_by_key(|(tsid, _)| *tsid);

        let mut merged_entries = Vec::with_capacity(remapped.len());
        let mut idx = 0usize;
        while idx < remapped.len() {
            let tsid = remapped[idx].0;
            let token_set = Arc::clone(&remapped[idx].1);
            idx += 1;
            while idx < remapped.len() && remapped[idx].0 == tsid {
                idx += 1;
            }
            merged_entries.push((tsid, token_set));
        }

        *entries = merged_entries;
    }
}

fn apply_tsid_perm_to_id_map(id_map: &mut ManyToOneIdMap, perm: &[u32], new_count: usize) {
    let old_internal_to_originals = std::mem::take(&mut id_map.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.representative_original_ids);

    for internal in &mut id_map.original_to_internal {
        if *internal != u32::MAX {
            *internal = perm[*internal as usize];
        }
    }

    let mut new_internal_to_originals = vec![Vec::new(); new_count];
    let mut new_representatives = vec![u32::MAX; new_count];
    for (old_internal, originals) in old_internal_to_originals.into_iter().enumerate() {
        let new_internal = perm[old_internal] as usize;
        new_internal_to_originals[new_internal].extend(originals);
        if new_representatives[new_internal] == u32::MAX {
            new_representatives[new_internal] = old_representatives[old_internal];
        }
    }

    id_map.internal_to_originals = new_internal_to_originals;
    id_map.representative_original_ids = new_representatives;
}

struct L1IdMapProfile {
    state_equiv_ms: f64,
    token_identity_map_ms: f64,
}

struct L1TsidProfileMergeReport {
    tsids_after: usize,
    unique_arc_token_sets: usize,
    unique_range_token_sets: usize,
    profile_build_ms: f64,
    group_ms: f64,
    remap_ms: f64,
}

struct L1TerminalBuildProfile {
    internal_vocab_ms: f64,
    vocab_tree_build_ms: f64,
    state_seed_ms: f64,
    token_set_intern_ms: f64,
    tsid_profile_merge_ms: f64,
    tsid_profile_merge_before: usize,
    tsid_profile_merge_after: usize,
    vocab_tree_traversal_ms: f64,
    direct_terminal_dwa_ms: f64,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::automata::lexer::ast::{byte, star};
    use crate::compiler::compile::build_tokenizer_from_exprs;

    fn naive_end_rep_sets(
        tokenizer: &Tokenizer,
        sorted_entries: &[(u32, &[u8])],
        start_state: u32,
        state_to_rep: &[u32],
    ) -> BTreeMap<u32, RangeSetBlaze<u32>> {
        let mut out = BTreeMap::new();
        for (internal_token_id, &(_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
            let mut state = start_state;
            let mut blocked = false;
            for &byte in token_bytes {
                let Some(next_state) = tokenizer.step(state, byte) else {
                    blocked = true;
                    break;
                };
                state = next_state;
            }
            if !blocked {
                out.entry(state_to_rep[state as usize])
                    .or_insert_with(RangeSetBlaze::new)
                    .insert(internal_token_id as u32);
            }
        }
        out
    }

    #[test]
    fn test_l1_lexicographic_root_traversal_matches_naive_simulation() {
        let tokenizer = build_tokenizer_from_exprs(&[star(byte(b'a'))]);
        let mut token_entries = vec![
            (10u32, b"a".to_vec()),
            (11u32, b"a".to_vec()),
            (12u32, b"aa".to_vec()),
            (13u32, b"aaa".to_vec()),
            (14u32, b"b".to_vec()),
        ];
        token_entries.sort_unstable_by(|left, right| left.1.cmp(&right.1));
        let sorted_entries: Vec<(u32, &[u8])> = token_entries
            .iter()
            .map(|(token_id, bytes)| (*token_id, bytes.as_slice()))
            .collect();

        let flat_trans = build_flat_transition_table(&tokenizer);
        let state_to_rep: Vec<u32> = (0..tokenizer.num_states()).collect();
        let mut end_rep_ranges = FxHashMap::<u32, Vec<(u32, u32)>>::default();
        let mut empty_token_indices = Vec::<usize>::new();
        let mut token_indices_by_first_byte = vec![Vec::<usize>::new(); 256];
        for (internal_token_id, &(_original_id, token_bytes)) in sorted_entries.iter().enumerate() {
            if let Some(&first_byte) = token_bytes.first() {
                token_indices_by_first_byte[first_byte as usize].push(internal_token_id);
            } else {
                empty_token_indices.push(internal_token_id);
            }
        }

        collect_l1_root_ranges_by_first_byte_lcp(
            tokenizer.initial_state(),
            &sorted_entries,
            &empty_token_indices,
            &token_indices_by_first_byte,
            &flat_trans,
            &state_to_rep,
            &mut end_rep_ranges,
        );

        let actual: BTreeMap<u32, RangeSetBlaze<u32>> = end_rep_ranges
            .into_iter()
            .map(|(end_rep, ranges)| {
                (
                    end_rep,
                    ranges.into_iter().map(|(start, end)| start..=end).collect(),
                )
            })
            .collect();
        let expected = naive_end_rep_sets(
            &tokenizer,
            &sorted_entries,
            tokenizer.initial_state(),
            &state_to_rep,
        );

        assert_eq!(actual, expected);
    }
}
