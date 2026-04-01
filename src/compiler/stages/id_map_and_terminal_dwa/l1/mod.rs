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
        &id_map,
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
            "[glrmask/profile][l1] partition={} vocab_tokens={} tsids={} state_equiv_ms={:.3} token_identity_map_ms={:.3} id_map_ms={:.3} internal_vocab_ms={:.3} vocab_tree_build_ms={:.3} state_seed_ms={:.3} vocab_tree_traversal_ms={:.3} direct_terminal_dwa_ms={:.3} terminal_build_ms={:.3} compact_ms={:.3} determinize=none minimize=none prune=none total_ms={:.3}{}",
            partition_label,
            vocab.entries.len(),
            id_map.num_tsids(),
            id_map_profile.state_equiv_ms,
            id_map_profile.token_identity_map_ms,
            id_map_ms,
            terminal_profile.internal_vocab_ms,
            terminal_profile.vocab_tree_build_ms,
            terminal_profile.state_seed_ms,
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
    id_map: &InternalIdMap,
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

    // Build flat transition table for O(1) DFA step lookups.
    let dead = u32::MAX;
    let mut flat_trans = vec![dead; num_dfa_states * 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        let base = state_idx * 256;
        for (byte, &target) in dfa_state.transitions.iter() {
            flat_trans[base + byte as usize] = target;
        }
    }

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

    let per_thread_results: Vec<Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)>> = start_states_list
        .par_iter()
        .map(|&(&start_state, ref initial_tsids)| {
            let mut end_rep_tokens = FxHashMap::<u32, Vec<u32>>::default();

            // Phase 1: Simulate all tokens from this start state.
            for &internal_token_id in &empty_token_indices {
                let end_rep = state_to_rep[start_state as usize];
                end_rep_tokens
                    .entry(end_rep)
                    .or_default()
                    .push(internal_token_id as u32);
            }

            for (first_byte, token_ids) in token_indices_by_first_byte.iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                let first_target = flat_trans[start_state as usize * 256 + first_byte];
                if first_target == dead {
                    continue;
                }

                for &internal_token_id in token_ids {
                    let token_bytes = sorted_entries[internal_token_id].1;
                    let mut state = first_target;
                    for &byte in &token_bytes[1..] {
                        let next = flat_trans[state as usize * 256 + byte as usize];
                        if next == dead {
                            state = dead;
                            break;
                        }
                        state = next;
                    }
                    if state != dead {
                        let end_rep = state_to_rep[state as usize];
                        end_rep_tokens
                            .entry(end_rep)
                            .or_default()
                            .push(internal_token_id as u32);
                    }
                }
            }

            // Phase 2: Build Arc'd RangeSets per (end_rep, tsid).
            // Arc::clone is ~5ns vs deep clone ~200ns.
            let mut result: Vec<(u32, u32, Arc<RangeSetBlaze<u32>>)> = Vec::new();
            for (end_rep, token_ids) in end_rep_tokens {
                let range_set: Arc<RangeSetBlaze<u32>> = Arc::new(
                    token_ids.into_iter().map(|x| x..=x).collect(),
                );

                for &tsid in initial_tsids.iter() {
                    result.push((end_rep, tsid, Arc::clone(&range_set)));
                }
            }

            result
        })
        .collect();

    // Concatenate per-thread results. No merge needed since each (end_rep, tsid)
    // pair appears in exactly one thread.
    let mut deferred_arced: Vec<Vec<(u32, Arc<RangeSetBlaze<u32>>)>> =
        vec![Vec::new(); num_dfa_states];
    for thread_result in per_thread_results {
        for (end_rep, tsid, arc) in thread_result {
            deferred_arced[end_rep as usize].push((tsid, arc));
        }
    }
    let traversal_ms = traversal_started_at.elapsed().as_secs_f64() * 1000.0;

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
            vocab_tree_traversal_ms,
            direct_terminal_dwa_ms,
        },
    ))
}

struct L1IdMapProfile {
    state_equiv_ms: f64,
    token_identity_map_ms: f64,
}

struct L1TerminalBuildProfile {
    internal_vocab_ms: f64,
    vocab_tree_build_ms: f64,
    state_seed_ms: f64,
    vocab_tree_traversal_ms: f64,
    direct_terminal_dwa_ms: f64,
}
