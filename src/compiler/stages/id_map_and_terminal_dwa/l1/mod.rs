//! L1 terminal DWA: direct 2-state construction for terminals with max path
//! length ≤ 1.

pub(crate) mod max_length;

use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::compact::{compact_dwa_dimensions, compact_dwa_dimensions_fast};
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::merge::{
    LocalIdMapTerminalDwa, identity_original_to_local_state,
};
use crate::ds::u8set::U8Set;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
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
        compact_dwa_dimensions(&mut dwa, &mut id_map, true)
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
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} state_equiv_ms={:.3} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} token_set_intern_ms={:.3} tsid_profile_merge_ms={:.3} tsid_profile_merge_before={} tsid_profile_merge_after={} vocab_tree_traversal_ms={:.3} self_loop_subtrees_skipped={} direct_terminal_dwa_ms={:.3} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}{}",
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
            terminal_profile.self_loop_subtrees_skipped,
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

    let vocab_tree_started_at = Instant::now();
    let (grouped_entries, group_internal_ranges) = group_sorted_entries_by_bytes(&sorted_entries);
    let vocab_tree = VocabPrefixTree::build_presorted(&grouped_entries);
    let vocab_tree_build_ms = vocab_tree_started_at.elapsed().as_secs_f64() * 1000.0;

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

    let per_thread_results: Vec<(Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)>, L1TrieTraversalProfile)> = start_states_list
        .par_iter()
        .map(|&(&start_state, ref initial_tsids)| {
            let mut end_rep_ranges = FxHashMap::<u32, Vec<(u32, u32)>>::default();
            let mut self_loop_bytes = FxHashMap::<u32, U8Set>::default();
            let mut traversal_profile = L1TrieTraversalProfile::default();
            collect_l1_end_rep_ranges_from_root(
                &vocab_tree.root,
                start_state,
                &flat_trans,
                state_to_rep,
                &group_internal_ranges,
                tokenizer,
                &mut self_loop_bytes,
                &mut end_rep_ranges,
                &mut traversal_profile,
            );

            // Phase 2: Build Arc'd RangeSets per (end_rep, tsid).
            // Arc::clone is ~5ns vs deep clone ~200ns.
            let mut result: Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
            for (end_rep, mut token_ranges) in end_rep_ranges {
                merge_ranges_in_place(&mut token_ranges);
                let range_set: Arc<RangeSetBlaze<u32>> = Arc::new(
                    token_ranges.into_iter().map(|(start, end)| start..=end).collect(),
                );

                for &tsid in initial_tsids.iter() {
                    result.push((end_rep, tsid, Arc::clone(&range_set)));
                }
            }

            (result, traversal_profile)
        })
        .collect();

    let mut traversal_profile = L1TrieTraversalProfile::default();

    // Canonicalize identical token sets across start groups so later stages can
    // key off shared Arc identity instead of rebuilding content-based keys.
    let token_set_intern_started_at = Instant::now();
    let mut interned_arc_by_ptr = FxHashMap::<usize, Arc<RangeSetBlaze<u32>>>::default();
    let mut interned_arc_by_ranges = FxHashMap::<Vec<(u32, u32)>, Arc<RangeSetBlaze<u32>>>::default();

    // Concatenate per-thread results. No merge needed since each (end_rep, tsid)
    // pair appears in exactly one thread.
    let mut deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        vec![Vec::new(); num_dfa_states];
    for (thread_result, thread_profile) in per_thread_results {
        traversal_profile.child_segments_visited += thread_profile.child_segments_visited;
        traversal_profile.byte_steps += thread_profile.byte_steps;
        traversal_profile.blocked_segments += thread_profile.blocked_segments;
        traversal_profile.recursive_descents += thread_profile.recursive_descents;
        traversal_profile.self_loop_subtrees_skipped += thread_profile.self_loop_subtrees_skipped;
        for (end_rep, tsid, arc) in thread_result {
            let arc_ptr = Arc::as_ptr(&arc) as usize;
            let canonical_arc = if let Some(existing) = interned_arc_by_ptr.get(&arc_ptr) {
                Arc::clone(existing)
            } else {
                let ranges: Vec<(u32, u32)> = arc
                    .ranges()
                    .map(|range| (*range.start(), *range.end()))
                    .collect();
                let canonical = interned_arc_by_ranges
                    .entry(ranges)
                    .or_insert_with(|| Arc::clone(&arc))
                    .clone();
                interned_arc_by_ptr.insert(arc_ptr, Arc::clone(&canonical));
                canonical
            };
            deferred_arced[end_rep as usize].push((tsid, canonical_arc));
        }
    }
    let token_set_intern_ms = token_set_intern_started_at.elapsed().as_secs_f64() * 1000.0;
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

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

    // Per-terminal distribution: collect deferred entries into per-tsid slots,
    // sort occupied tsids, merge per-tsid, build Weight.
    // Parallelized across terminals using rayon.
    let merge_started_at = Instant::now();

    let num_tsids = id_map.num_tsids() as usize;

    // Compute per-terminal weights in parallel
    let terminal_weights: Vec<(usize, Weight)> = terminal_to_active_states
        .par_iter()
        .enumerate()
        .filter_map(|(tid, active_states)| {
            if active_states.is_empty() {
                return None;
            }

            // Thread-local buffers for this terminal
            let mut tsid_ranges: Vec<Vec<(u32, u32)>> = (0..num_tsids).map(|_| Vec::new()).collect();
            let mut tsid_single_arc: Vec<Option<Arc<RangeSetBlaze<u32>>>> = vec![None; num_tsids];
            let mut occupied_tsids: Vec<u32> = Vec::new();

            for &state in active_states {
                for &(tsid, ref arc) in &deferred_arced[state as usize] {
                    let tsid_idx = tsid as usize;
                    if tsid_ranges[tsid_idx].is_empty() && tsid_single_arc[tsid_idx].is_none() {
                        tsid_single_arc[tsid_idx] = Some(Arc::clone(arc));
                        occupied_tsids.push(tsid);
                    } else if let Some(first_arc) = tsid_single_arc[tsid_idx].take() {
                        for r in first_arc.ranges() {
                            tsid_ranges[tsid_idx].push((*r.start(), *r.end()));
                        }
                        for r in arc.ranges() {
                            tsid_ranges[tsid_idx].push((*r.start(), *r.end()));
                        }
                    } else {
                        for r in arc.ranges() {
                            tsid_ranges[tsid_idx].push((*r.start(), *r.end()));
                        }
                    }
                }
            }

            if occupied_tsids.is_empty() {
                return None;
            }

            occupied_tsids.sort_unstable();

            let mut weight_entries: Vec<(u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
            for &tsid in &occupied_tsids {
                let tsid_idx = tsid as usize;
                if let Some(arc) = tsid_single_arc[tsid_idx].take() {
                    weight_entries.push((tsid, arc));
                } else {
                    let slot = &mut tsid_ranges[tsid_idx];
                    slot.sort_unstable();
                    if !slot.is_empty() {
                        let mut write = 0;
                        for read in 1..slot.len() {
                            if slot[read].0 <= slot[write].1.saturating_add(1) {
                                slot[write].1 = slot[write].1.max(slot[read].1);
                            } else {
                                write += 1;
                                slot[write] = slot[read];
                            }
                        }
                        slot.truncate(write + 1);
                    }
                    let merged: RangeSetBlaze<u32> = slot.iter()
                        .map(|&(s, e)| s..=e)
                        .collect();
                    weight_entries.push((tsid, shared_rangeset(merged)));
                }
            }

            let weight = Weight::from_per_tsid_shared(
                weight_entries.iter().map(|(tsid, arc)| (*tsid, Arc::clone(arc))),
            );
            if weight.is_empty() {
                return None;
            }

            Some((tid, weight))
        })
        .collect();

    // Sequential DWA construction from parallel results
    let mut dwa = DWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let end_state = dwa.add_state();
    dwa.set_final_weight(end_state, Weight::all());
    let mut num_transitions = 0usize;

    for (tid, weight) in terminal_weights {
        dwa.add_transition(dwa.start_state, tid as i32, end_state, weight);
        num_transitions += 1;
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
            "[glrmask/debug][terminal_dwa] partition_build_l1_batch vocab={} tsids={} transitions={} child_segments={} byte_steps={} recursive_descents={} self_loop_subtrees_skipped={} traversal_ms={:.1} arc_wrap_ms={:.1} inverse_map_ms={:.1} merge_ms={:.1} distribute_ms={:.1} total_ms={:.1}",
            sorted_entries.len(),
            id_map.num_tsids(),
            num_transitions,
            traversal_profile.child_segments_visited,
            traversal_profile.byte_steps,
            traversal_profile.recursive_descents,
            traversal_profile.self_loop_subtrees_skipped,
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
            self_loop_subtrees_skipped: traversal_profile.self_loop_subtrees_skipped,
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

fn group_sorted_entries_by_bytes<'a>(
    sorted_entries: &[(u32, &'a [u8])],
) -> (Vec<(usize, &'a [u8])>, Vec<(u32, u32)>) {
    let mut grouped_entries = Vec::new();
    let mut group_internal_ranges = Vec::new();
    let mut index = 0usize;

    while index < sorted_entries.len() {
        let bytes = sorted_entries[index].1;
        let start = index as u32;
        index += 1;
        while index < sorted_entries.len() && sorted_entries[index].1 == bytes {
            index += 1;
        }
        let end = index as u32 - 1;
        let group_id = group_internal_ranges.len();
        grouped_entries.push((group_id, bytes));
        group_internal_ranges.push((start, end));
    }

    (grouped_entries, group_internal_ranges)
}

fn append_group_span(
    token_ranges: &mut Vec<(u32, u32)>,
    group_internal_ranges: &[(u32, u32)],
    start_group_id: usize,
    end_group_id: usize,
) {
    if start_group_id > end_group_id {
        return;
    }
    token_ranges.push((
        group_internal_ranges[start_group_id].0,
        group_internal_ranges[end_group_id].1,
    ));
}

fn append_group_token_range(
    token_ranges: &mut Vec<(u32, u32)>,
    group_internal_ranges: &[(u32, u32)],
    group_id: usize,
) {
    let (start, end) = group_internal_ranges[group_id];
    token_ranges.push((start, end));
}

fn append_reachable_group_ranges_excluding(
    token_ranges: &mut Vec<(u32, u32)>,
    node: &VocabPrefixTreeNode,
    group_internal_ranges: &[(u32, u32)],
    exclude_group_id: Option<usize>,
) {
    for group_range in node.reachable_token_ids().ranges() {
        let start_group_id = *group_range.start();
        let end_group_id = *group_range.end();
        if let Some(exclude_group_id) = exclude_group_id {
            if start_group_id <= exclude_group_id && exclude_group_id <= end_group_id {
                if start_group_id < exclude_group_id {
                    append_group_span(
                        token_ranges,
                        group_internal_ranges,
                        start_group_id,
                        exclude_group_id - 1,
                    );
                }
                if exclude_group_id < end_group_id {
                    append_group_span(
                        token_ranges,
                        group_internal_ranges,
                        exclude_group_id + 1,
                        end_group_id,
                    );
                }
                continue;
            }
        }
        append_group_span(
            token_ranges,
            group_internal_ranges,
            start_group_id,
            end_group_id,
        );
    }
}

fn can_skip_self_loop_subtree(
    tokenizer: &Tokenizer,
    node: &VocabPrefixTreeNode,
    tokenizer_state: u32,
    self_loop_bytes: &mut FxHashMap<u32, U8Set>,
) -> bool {
    let self_loop_bytes = self_loop_bytes.entry(tokenizer_state).or_insert_with(|| {
        let state = &tokenizer.dfa.states()[tokenizer_state as usize];
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

fn collect_l1_end_rep_ranges_from_root(
    root: &VocabPrefixTreeNode,
    start_state: u32,
    flat_trans: &[u32],
    state_to_rep: &[u32],
    group_internal_ranges: &[(u32, u32)],
    tokenizer: &Tokenizer,
    self_loop_bytes: &mut FxHashMap<u32, U8Set>,
    end_rep_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
    profile: &mut L1TrieTraversalProfile,
) {
    collect_l1_end_rep_ranges_for_node(
        root,
        start_state,
        flat_trans,
        state_to_rep,
        group_internal_ranges,
        tokenizer,
        self_loop_bytes,
        end_rep_ranges,
        profile,
    );
}

fn collect_l1_end_rep_ranges_for_node(
    node: &VocabPrefixTreeNode,
    tokenizer_state: u32,
    flat_trans: &[u32],
    state_to_rep: &[u32],
    group_internal_ranges: &[(u32, u32)],
    tokenizer: &Tokenizer,
    self_loop_bytes: &mut FxHashMap<u32, U8Set>,
    end_rep_ranges: &mut FxHashMap<u32, Vec<(u32, u32)>>,
    profile: &mut L1TrieTraversalProfile,
) {
    if node.has_token() {
        let end_rep = state_to_rep[tokenizer_state as usize];
        append_group_token_range(
            end_rep_ranges.entry(end_rep).or_default(),
            group_internal_ranges,
            node.token_id(),
        );
    }

    for (segment_bytes, child) in node.iter_children() {
        profile.child_segments_visited += 1;
        let mut current_state = tokenizer_state;
        let mut segment_blocked = false;

        for &byte in segment_bytes {
            profile.byte_steps += 1;
            let next_state = flat_trans[current_state as usize * 256 + byte as usize];
            if next_state == u32::MAX {
                segment_blocked = true;
                break;
            }
            current_state = next_state;
        }

        if segment_blocked {
            profile.blocked_segments += 1;
            continue;
        }

        if child.children().is_empty() {
            if child.has_token() {
                let end_rep = state_to_rep[current_state as usize];
                append_group_token_range(
                    end_rep_ranges.entry(end_rep).or_default(),
                    group_internal_ranges,
                    child.token_id(),
                );
            }
            continue;
        }

        if can_skip_self_loop_subtree(tokenizer, child, current_state, self_loop_bytes) {
            let end_rep = state_to_rep[current_state as usize];
            let token_ranges = end_rep_ranges.entry(end_rep).or_default();
            if child.has_token() {
                append_group_token_range(token_ranges, group_internal_ranges, child.token_id());
            }
            append_reachable_group_ranges_excluding(
                token_ranges,
                child,
                group_internal_ranges,
                child.has_token().then_some(child.token_id()),
            );
            profile.self_loop_subtrees_skipped += 1;
            continue;
        }

        profile.recursive_descents += 1;
        collect_l1_end_rep_ranges_for_node(
            child,
            current_state,
            flat_trans,
            state_to_rep,
            group_internal_ranges,
            tokenizer,
            self_loop_bytes,
            end_rep_ranges,
            profile,
        );
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
    self_loop_subtrees_skipped: u64,
    direct_terminal_dwa_ms: f64,
}

#[derive(Clone, Copy, Default)]
struct L1TrieTraversalProfile {
    child_segments_visited: u64,
    byte_steps: u64,
    blocked_segments: u64,
    recursive_descents: u64,
    self_loop_subtrees_skipped: u64,
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
    fn test_l1_trie_traversal_matches_flat_simulation_with_duplicate_bytes() {
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

        let (grouped_entries, group_internal_ranges) = group_sorted_entries_by_bytes(&sorted_entries);
        let vocab_tree = VocabPrefixTree::build_presorted(&grouped_entries);
        let flat_trans = build_flat_transition_table(&tokenizer);
        let state_to_rep: Vec<u32> = (0..tokenizer.num_states()).collect();

        let mut end_rep_ranges = FxHashMap::<u32, Vec<(u32, u32)>>::default();
        let mut self_loop_bytes = FxHashMap::<u32, U8Set>::default();
        let mut profile = L1TrieTraversalProfile::default();
        collect_l1_end_rep_ranges_from_root(
            &vocab_tree.root,
            tokenizer.initial_state(),
            &flat_trans,
            &state_to_rep,
            &group_internal_ranges,
            &tokenizer,
            &mut self_loop_bytes,
            &mut end_rep_ranges,
            &mut profile,
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
        assert!(profile.self_loop_subtrees_skipped > 0);
    }
}
