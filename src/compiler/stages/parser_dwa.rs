use std::collections::{hash_map::Entry, BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
#[cfg(test)]
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::{minimize_fast, minimize_from_env};
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
use crate::compiler::glr::table::GLRTable;
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
#[cfg(test)]
use crate::compiler::stages::terminal_dwa_compat::build_terminal_dwa_for_existing_id_map;
use crate::compiler::stages::templates::Templates;
#[cfg(test)]
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::ds::bitset::BitSet;
use crate::ds::weight::Weight;

type TerminalBundle = BTreeMap<TerminalID, Weight>;
type BundleSignature = Vec<(TerminalID, Weight)>;
type ComposeMemo = Vec<Option<Option<NwaBody>>>;
type ConcatMemo = FxHashMap<(usize, u32), Option<NwaBody>>;
type TargetContribs = SmallVec<[(u32, Weight); 4]>;

fn add_target_contribution(contribs: &mut TargetContribs, target: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    if let Some((_, existing)) = contribs.iter_mut().find(|(existing_target, _)| *existing_target == target) {
        *existing = existing.union(&add);
    } else {
        contribs.push((target, add));
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ParserDwaPhaseProfile {
    build_state_summaries_ms: f64,
    compose_state_ms: f64,
    resolve_negatives_ms: f64,
    viable_suffix_ms: f64,
    determinize_supports_ms: f64,
    optimize_defaults_ms: f64,
    subtract_final_ms: f64,
    fallback_to_nwa_ms: f64,
    ensure_fallback_ms: f64,
    determinize_fallback_ms: f64,
    determinize_after_defaults_ms: f64,
    minimize_ms: f64,
    total_ms: f64,
}

fn parser_dwa_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some()
}

fn parser_dwa_bundle_determinize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_PARSER_DWA_BUNDLE_DETERMINIZE").is_some()
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

#[derive(Debug, Clone, Copy)]
struct PendingBranch {
    target: u32,
    bundle_id: usize,
}

#[derive(Debug, Clone)]
struct Branch {
    target: u32,
    bundle_id: usize,
    bundle: Arc<NWA>,
}

#[derive(Debug, Clone)]
struct StateSummary {
    final_weight: Option<Weight>,
    branches: Vec<Branch>,
}

#[derive(Debug, Clone)]
struct DeterminizedDwaWithSupports {
    dwa: DWA,
    supports: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
struct CachedClosure {
    canon: Vec<(u32, Weight)>,
    edge_weight: Weight,
}

fn accepting_nwa(final_weight: &Weight) -> Option<NWA> {
    if final_weight.is_empty() {
        return None;
    }

    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states_mut().push(start);
    nwa.set_final_weight(start, final_weight.clone());
    Some(nwa)
}

fn group_terminal_edges_by_target(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, TerminalBundle> {
    let Some(state) = terminal_dwa.states().get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut bundles_by_target = BTreeMap::<u32, TerminalBundle>::new();
    for (&label, (target, weight)) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        bundles_by_target
            .entry(*target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }

    bundles_by_target
}

fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {
    bundle
        .iter()
        .map(|(&terminal, weight)| (terminal, weight.clone()))
        .collect()
}

fn build_state_summaries(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> Vec<StateSummary> {
    let mut pending_branches_by_state: Vec<Vec<PendingBranch>> =
        Vec::with_capacity(terminal_dwa.states().len());
    let mut bundle_ids_by_signature: FxHashMap<BundleSignature, usize> = FxHashMap::default();
    let mut unique_bundles: Vec<TerminalBundle> = Vec::new();

    for (state_id, _state) in terminal_dwa.states().iter().enumerate() {
        let bundles_by_target = group_terminal_edges_by_target(terminal_dwa, grammar, state_id as u32);
        let mut pending_branches = Vec::with_capacity(bundles_by_target.len());
        for (target, bundle) in bundles_by_target {
            let signature = bundle_signature(&bundle);
            let bundle_id = if let Some(&bundle_id) = bundle_ids_by_signature.get(&signature) {
                bundle_id
            } else {
                let bundle_id = unique_bundles.len();
                bundle_ids_by_signature.insert(signature, bundle_id);
                unique_bundles.push(bundle.clone());
                bundle_id
            };
            pending_branches.push(PendingBranch { target, bundle_id });
        }
        pending_branches_by_state.push(pending_branches);
    }

    if parser_dwa_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_dwa_bundles] terminal_dwa_states={} unique_bundles={} total_branches={}",
            terminal_dwa.states().len(),
            unique_bundles.len(),
            pending_branches_by_state.iter().map(|b| b.len()).sum::<usize>(),
        );
    }

    let built_bundles: Vec<Arc<NWA>> = {
        use rayon::prelude::*;
        let profile_enabled = parser_dwa_profile_enabled();
        let det_profile_enabled = parser_dwa_bundle_determinize_profile_enabled();
        unique_bundles
            .par_iter()
            .enumerate()
            .map(|(bundle_id, bundle)| {
                if !profile_enabled {
                    return Arc::new(templates.build_bundle(bundle));
                }

                let build_started_at = Instant::now();
                let (built_nwa, bundle_profile) = templates.build_bundle_profiled(bundle);
                let nwa = Arc::new(built_nwa);
                {
                    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
                    let nwa_transitions: usize = nwa
                        .states()
                        .iter()
                        .map(|s| {
                            s.transitions.values().map(|v| v.len()).sum::<usize>()
                                + s.epsilons.len()
                        })
                        .sum();
                    eprintln!(
                        "[glrmask/profile][parser_dwa_bundle] bundle={:>4} terminals={:>4} nwa_states={:>5} nwa_trans={:>6} build_ms={:>7.1}",
                        bundle_id,
                        bundle.len(),
                        nwa.states().len(),
                        nwa_transitions,
                        build_ms,
                    );
                    eprintln!(
                        concat!(
                            "[glrmask/profile][parser_dwa_bundle_detail] ",
                            "bundle={:>4} input_terminals={:>4} nonempty_terminals={:>4} ",
                            "weight_groups={:>4} singleton_groups={:>4} multi_groups={:>4} largest_group={:>4} ",
                            "group_dfas_ms={:>7.1} union_groups_ms={:>7.1} ",
                            "slowest_group_terminals={:>4} slowest_group_dfa_states={:>5} slowest_group_dfa_trans={:>6} slowest_group_ms={:>7.1} ",
                            "determinize_bundle_ms={:>7.1} minimize_ms={:>7.1} dwa_to_nwa_ms={:>7.1} ",
                            "result_dwa_states={:>5} result_dwa_trans={:>6} result_nwa_states={:>5} result_nwa_trans={:>6} ",
                            "fast_path={} total_ms={:>7.1}"
                        ),
                        bundle_id,
                        bundle_profile.input_terminals,
                        bundle_profile.nonempty_terminals,
                        bundle_profile.weight_groups,
                        bundle_profile.singleton_groups,
                        bundle_profile.multi_terminal_groups,
                        bundle_profile.largest_weight_group,
                        bundle_profile.build_group_dfas_ms,
                        bundle_profile.union_groups_ms,
                        bundle_profile.slowest_group_terminals,
                        bundle_profile.slowest_group_dfa_states,
                        bundle_profile.slowest_group_dfa_transitions,
                        bundle_profile.slowest_group_ms,
                        bundle_profile.determinize_bundle_ms,
                        bundle_profile.minimize_ms,
                        bundle_profile.dwa_to_nwa_ms,
                        bundle_profile.result_dwa_states,
                        bundle_profile.result_dwa_transitions,
                        bundle_profile.result_nwa_states,
                        bundle_profile.result_nwa_transitions,
                        bundle_profile.used_single_terminal_fast_path,
                        bundle_profile.total_ms,
                    );

                    if det_profile_enabled {
                        eprintln!(
                            concat!(
                                "[glrmask/profile][parser_dwa_bundle_determinize] ",
                                "bundle={:>4} det_pop_state_ms={:>7.1} det_alive_ms={:>7.1} det_effective_ms={:>7.1} ",
                                "det_final_ms={:>7.1} det_labels_ms={:>7.1} det_next_ms={:>7.1} det_edge_weight_ms={:>7.1} det_lookup_ms={:>7.1} det_add_transition_ms={:>7.1} ",
                                "det_states={:>5} det_labels={:>6} det_transitions={:>6} det_worklist_peak={:>5} det_cache_entries={:>5} ",
                                "edge_subset_total={:>6} edge_subset_max={:>4} edge_cache_hits={:>5} edge_cache_hit_subset_total={:>6} ",
                                "edge_cache_misses={:>5} edge_cache_miss_subset_total={:>6}"
                            ),
                            bundle_id,
                            bundle_profile.determinize_pop_state_ms,
                            bundle_profile.determinize_alive_groups_ms,
                            bundle_profile.determinize_effective_weights_ms,
                            bundle_profile.determinize_final_weight_ms,
                            bundle_profile.determinize_collect_labels_ms,
                            bundle_profile.determinize_next_state_ms,
                            bundle_profile.determinize_edge_weight_ms,
                            bundle_profile.determinize_state_lookup_ms,
                            bundle_profile.determinize_add_transition_ms,
                            bundle_profile.determinize_states_visited,
                            bundle_profile.determinize_labels_processed,
                            bundle_profile.determinize_transitions_added,
                            bundle_profile.determinize_worklist_peak,
                            bundle_profile.determinize_cache_entries,
                            bundle_profile.determinize_edge_subset_total,
                            bundle_profile.determinize_edge_subset_max,
                            bundle_profile.determinize_edge_cache_hits,
                            bundle_profile.determinize_edge_cache_hit_subset_total,
                            bundle_profile.determinize_edge_cache_misses,
                            bundle_profile.determinize_edge_cache_miss_subset_total,
                        );
                    }
                }
                nwa
            })
            .collect()
    };

    terminal_dwa
        .states()
        .iter()
        .enumerate()
        .map(|(state_id, state)| {
            let branches = pending_branches_by_state[state_id]
                .iter()
                .map(|pending| {
                    let built_bundle = Arc::clone(&built_bundles[pending.bundle_id]);
                    Branch {
                        target: pending.target,
                        bundle_id: pending.bundle_id,
                        bundle: built_bundle,
                    }
                })
                .collect();

            StateSummary {
                final_weight: state.final_weight.clone(),
                branches,
            }
        })
        .collect()
}

fn compose_state(
    state_id: u32,
    states: &[StateSummary],
    arena: &mut NWA,
    body_memo: &mut ComposeMemo,
    concatenated_branches: &mut ConcatMemo,
) -> Option<NwaBody> {
    if let Some(Some(cached)) = body_memo.get(state_id as usize) {
        return cached.clone();
    }

    let Some(state) = states.get(state_id as usize) else {
        return None;
    };

    let mut composed_body = state
        .final_weight
        .as_ref()
        .and_then(accepting_nwa)
        .map(|accepting| arena.append_with_body(&accepting));

    for branch in &state.branches {
        let Some(continuation) = compose_state(
            branch.target,
            states,
            arena,
            body_memo,
            concatenated_branches,
        ) else {
            continue;
        };

        let concat_key = (branch.bundle_id, branch.target);
        let branch_with_continuation = if let Some(cached) = concatenated_branches.get(&concat_key) {
            cached.clone()
        } else {
            let built = Some(arena.concatenate_in_place(branch.bundle.as_ref(), &continuation));
            concatenated_branches.insert(concat_key, built.clone());
            built
        };
        let Some(branch_with_continuation) = branch_with_continuation else {
            continue;
        };

        composed_body = Some(match composed_body {
            Some(existing) => NwaBody::union(&existing, &branch_with_continuation),
            None => branch_with_continuation,
        });
    }

    if let Some(entry) = body_memo.get_mut(state_id as usize) {
        *entry = Some(composed_body.clone());
    }
    composed_body
}

fn union_final_weight(slot: &mut Option<Weight>, add: Weight) -> bool {
    if add.is_empty() {
        return false;
    }

    match slot {
        Some(existing) => {
            let updated = existing.union(&add);
            if updated != *existing {
                *existing = updated;
                true
            } else {
                false
            }
        }
        None => {
            *slot = Some(add);
            true
        }
    }
}

fn add_or_union_transition(
    state: &mut crate::automata::weighted::nwa::NWAState,
    label: i32,
    target: u32,
    add: Weight,
) -> bool {
    if add.is_empty() {
        return false;
    }

    let targets = state.transitions.entry(label).or_default();
    for (existing_target, existing_weight) in targets.iter_mut() {
        if *existing_target == target {
            let updated = existing_weight.union(&add);
            if updated != *existing_weight {
                *existing_weight = updated;
                return true;
            }
            return false;
        }
    }

    targets.push((target, add));
    true
}

fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {
    if label >= 0 && (label as u32) < num_parser_states {
        Some(label as u32)
    } else {
        None
    }
}

enum PossibleOutgoingIds {
    Empty,
    All,
    Some(BitSet),
}

fn build_possible_outgoing_ids_by_state(
    parser_nwa: &NWA,
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    enum OutgoingIds {
        Empty,
        All,
        Some(Vec<u32>),
    }

    let num_parser_states = num_parser_states as usize;
    let all_parser_states = BitSet::all(num_parser_states);
    let state_outgoing_ids: Vec<OutgoingIds> = parser_nwa
        .states()
        .iter()
        .map(|state| {
            let mut ids = Vec::new();
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    return OutgoingIds::All;
                }
                if let Some(parser_state_id) = parser_state_label(label, num_parser_states as u32) {
                    ids.push(parser_state_id);
                }
            }
            if ids.is_empty() {
                OutgoingIds::Empty
            } else {
                OutgoingIds::Some(ids)
            }
        })
        .collect();

    state_supports
        .iter()
        .map(|support| {
            if support.len() == 1 {
                let state_id = support[0] as usize;
                return match state_outgoing_ids.get(state_id) {
                    Some(OutgoingIds::Empty) => PossibleOutgoingIds::Empty,
                    Some(OutgoingIds::All) => PossibleOutgoingIds::All,
                    Some(OutgoingIds::Some(ids)) => {
                        let mut bitset = BitSet::new(num_parser_states);
                        for &parser_state_id in ids {
                            bitset.set(parser_state_id as usize);
                        }
                        if bitset == all_parser_states {
                            PossibleOutgoingIds::All
                        } else {
                            PossibleOutgoingIds::Some(bitset)
                        }
                    }
                    None => PossibleOutgoingIds::Empty,
                };
            }

            let mut ids = BitSet::new(num_parser_states);
            for &state_id in support {
                let Some(state_ids) = state_outgoing_ids.get(state_id as usize) else {
                    continue;
                };
                match state_ids {
                    OutgoingIds::Empty => {}
                    OutgoingIds::All => return PossibleOutgoingIds::All,
                    OutgoingIds::Some(state_ids) => {
                        for &parser_state_id in state_ids {
                            ids.set(parser_state_id as usize);
                        }
                        if ids == all_parser_states {
                            break;
                        }
                    }
                }
            }
            if ids.is_empty() {
                PossibleOutgoingIds::Empty
            } else if ids == all_parser_states {
                PossibleOutgoingIds::All
            } else {
                PossibleOutgoingIds::Some(ids)
            }
        })
        .collect()
}

fn local_epsilon_closure(
    nwa: &NWA,
    weight_by_state: &mut Vec<Option<Weight>>,
    closure_queue: &mut VecDeque<u32>,
    seed: &mut FxHashMap<u32, Weight>,
) {
    let mut seed_states: Vec<u32> = Vec::new();
    for (&state_id, weight) in seed.iter() {
        weight_by_state[state_id as usize] = Some(weight.clone());
        closure_queue.push_back(state_id);
        seed_states.push(state_id);
    }
    if seed.len() == 1 {
        let state_id = seed_states[0];
        if let Some(state) = nwa.states().get(state_id as usize) {
            if state.epsilons.is_empty() {
                closure_queue.clear();
                for &s in &seed_states {
                    weight_by_state[s as usize] = None;
                }
                return;
            }
        }
    }
    while let Some(state_id) = closure_queue.pop_front() {
        let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
            continue;
        };
        let Some(state) = nwa.states().get(state_id as usize) else {
            continue;
        };
        for (target, edge_weight) in &state.epsilons {
            let contribution = current_weight.intersection(edge_weight);
            if contribution.is_empty() {
                continue;
            }
            let target_idx = *target as usize;
            if let Some(existing) = &weight_by_state[target_idx] {
                if !contribution.is_subset(existing) {
                    weight_by_state[target_idx] = Some(existing.union(&contribution));
                    closure_queue.push_back(*target);
                    if !seed_states.contains(target) {
                        seed_states.push(*target);
                    }
                }
            } else {
                weight_by_state[target_idx] = Some(contribution);
                closure_queue.push_back(*target);
                seed_states.push(*target);
            }
        }
    }
    seed.clear();
    for &s in &seed_states {
        if let Some(w) = weight_by_state[s as usize].take() {
            seed.insert(s, w);
        }
    }
}

fn determinize_with_supports(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
) -> DeterminizedDwaWithSupports {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let profile_enabled = parser_dwa_profile_enabled();
    let mut prof_iterations: u64 = 0;
    let mut prof_total_subset_entries: u64 = 0;
    let mut prof_max_subset_size: usize = 0;
    let mut prof_intersection_calls: u64 = 0;
    let mut prof_eps_closure_calls: u64 = 0;
    let mut prof_labels_processed: u64 = 0;
    let mut prof_eps_closure_ns: u64 = 0;
    let mut prof_intersection_ns: u64 = 0;
    let mut prof_target_build_ns: u64 = 0;
    let mut prof_target_drain_ns: u64 = 0;
    let mut prof_target_filter_ns: u64 = 0;
    let mut prof_target_edge_weight_ns: u64 = 0;
    let mut prof_target_pre_closure_key_ns: u64 = 0;
    let mut prof_target_closure_lookup_insert_ns: u64 = 0;
    let mut prof_target_subset_lookup_ns: u64 = 0;
    let mut prof_target_add_transition_ns: u64 = 0;
    let mut prof_max_raw_target_entries_per_iter: u64 = 0;
    let mut prof_max_labels_per_iter: u64 = 0;
    let mut prof_single_target_no_epsilon_fast_path: u64 = 0;
    let mut prof_target_subset_len_1: u64 = 0;
    let mut prof_target_subset_len_2_4: u64 = 0;
    let mut prof_target_subset_len_5_16: u64 = 0;
    let mut prof_target_subset_len_gt_16: u64 = 0;
    let mut prof_subset_map_hits: u64 = 0;
    let mut prof_subset_map_inserts: u64 = 0;
    let mut prof_dense_labels_processed: u64 = 0;
    let mut prof_default_labels_processed: u64 = 0;
    let mut prof_sparse_labels_processed: u64 = 0;
    let mut prof_edge_weight_cache_hits: u64 = 0;
    let mut prof_edge_weight_cache_misses: u64 = 0;
    let determinize_started_at = Instant::now();
    let mut last_progress_log_at = determinize_started_at;

    let num_nwa_states = nwa.states().len();

    // Use flat arrays for epsilon closure when NWA is small enough.
    // weight_by_state[i] = Some(weight) means state i is in the closure.
    let mut weight_by_state: Vec<Option<Weight>> = vec![None; num_nwa_states];
    let mut closure_queue: VecDeque<u32> = VecDeque::new();
    // Reusable buffer for canonicalized entries.
    let mut canon_buf: Vec<(u32, Weight)> = Vec::new();

    // Epsilon closure using flat arrays instead of FxHashMap.
    let epsilon_closure = |weight_by_state: &mut Vec<Option<Weight>>,
                           closure_queue: &mut VecDeque<u32>,
                           seed: &mut FxHashMap<u32, Weight>| {
        // Initialize flat array from seed.
        let mut seed_states: Vec<u32> = Vec::new();
        for (&state_id, weight) in seed.iter() {
            weight_by_state[state_id as usize] = Some(weight.clone());
            closure_queue.push_back(state_id);
            seed_states.push(state_id);
        }

        // Fast path: single seed with no epsilons.
        if seed.len() == 1 {
            let state_id = seed_states[0];
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    // Clean up and return early — seed is already populated.
                    closure_queue.clear();
                    for &s in &seed_states {
                        weight_by_state[s as usize] = None;
                    }
                    return;
                }
            }
        }

        while let Some(state_id) = closure_queue.pop_front() {
            let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (target, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let target_idx = *target as usize;
                if let Some(existing) = &weight_by_state[target_idx] {
                    if !contribution.is_subset(existing) {
                        weight_by_state[target_idx] = Some(existing.union(&contribution));
                        closure_queue.push_back(*target);
                    }
                } else {
                    weight_by_state[target_idx] = Some(contribution);
                    closure_queue.push_back(*target);
                    seed_states.push(*target);
                }
            }
        }

        // Write results back to seed map.
        seed.clear();
        for &s in &seed_states {
            if let Some(w) = weight_by_state[s as usize].take() {
                seed.insert(s, w);
            }
        }
    };

    // Canonicalize from FxHashMap into reusable buffer.
    let canonicalize_into =
        |map: &FxHashMap<u32, Weight>, buf: &mut Vec<(u32, Weight)>| {
            buf.clear();
            for (&state_id, weight) in map.iter() {
                if !weight.is_empty() {
                    buf.push((state_id, weight.clone()));
                }
            }
            buf.sort_unstable_by_key(|(state_id, _)| *state_id);
        };

    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![Vec::new()];

    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    epsilon_closure(&mut weight_by_state, &mut closure_queue, &mut start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    canonicalize_into(&start_subset, &mut canon_buf);
    supports[0] = canon_buf.iter().map(|(state_id, _)| *state_id).collect();

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&canon_buf), dwa.start_state());
    worklist.push_back(canon_buf.clone());

    let dense_label_limit = dense_positive_label_limit.map(|n| n as usize).unwrap_or(0);
    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    // Memoize local epsilon-closure outputs keyed by pre-closure weighted subsets.
    let mut closure_cache: FxHashMap<Vec<(u32, usize)>, CachedClosure> = FxHashMap::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut prof_final_weight_ns: u64 = 0;
    let mut prof_subset_key_ns: u64 = 0;
    let mut prof_total_raw_target_entries: u64 = 0;

    // Deferred final weight computation: store subset entries for each DWA state
    // and compute final weights in parallel after the main loop.
    let mut deferred_final_entries: Vec<(u32, Vec<(u32, Weight)>)> = Vec::new();

    while let Some(subset_entries) = worklist.pop_front() {
        let t_sk = std::time::Instant::now();
        let from_state = subset_map[&subset_key(&subset_entries)];
        if profile_enabled { prof_subset_key_ns += t_sk.elapsed().as_nanos() as u64; }

        if profile_enabled {
            prof_iterations += 1;
            prof_total_subset_entries += subset_entries.len() as u64;
            if subset_entries.len() > prof_max_subset_size {
                prof_max_subset_size = subset_entries.len();
            }
            if last_progress_log_at.elapsed().as_secs_f64() >= 5.0 {
                eprintln!(
                    "[glrmask/profile][determinize_supports_progress] elapsed_ms={:.1} iterations={} dwa_states={} pending_worklist={} subset_cache={} closure_cache={} max_subset_size={} labels_processed={} raw_target_entries={} avg_labels_per_iter={:.1} avg_raw_targets_per_iter={:.1} max_labels_per_iter={} max_raw_target_entries_per_iter={} single_target_no_epsilon_fast_path={} subset_len_1={} subset_len_2_4={} subset_len_5_16={} subset_len_gt_16={} subset_map_hits={} subset_map_inserts={} dense_labels={} default_labels={} sparse_labels={} edge_weight_cache_hits={} edge_weight_cache_misses={} subset_key_ms={:.1} collect_finals_ms={:.1} intersection_ms={:.1} eps_closure_ms={:.1} target_build_ms={:.1} target_drain_ms={:.1} target_filter_ms={:.1} target_edge_weight_ms={:.1} target_pre_closure_key_ms={:.1} target_closure_lookup_insert_ms={:.1} target_subset_lookup_ms={:.1} target_add_transition_ms={:.1}",
                    elapsed_ms(determinize_started_at),
                    prof_iterations,
                    dwa.states().len(),
                    worklist.len(),
                    subset_map.len(),
                    closure_cache.len(),
                    prof_max_subset_size,
                    prof_labels_processed,
                    prof_total_raw_target_entries,
                    if prof_iterations > 0 { prof_labels_processed as f64 / prof_iterations as f64 } else { 0.0 },
                    if prof_iterations > 0 { prof_total_raw_target_entries as f64 / prof_iterations as f64 } else { 0.0 },
                    prof_max_labels_per_iter,
                    prof_max_raw_target_entries_per_iter,
                    prof_single_target_no_epsilon_fast_path,
                    prof_target_subset_len_1,
                    prof_target_subset_len_2_4,
                    prof_target_subset_len_5_16,
                    prof_target_subset_len_gt_16,
                    prof_subset_map_hits,
                    prof_subset_map_inserts,
                    prof_dense_labels_processed,
                    prof_default_labels_processed,
                    prof_sparse_labels_processed,
                    prof_edge_weight_cache_hits,
                    prof_edge_weight_cache_misses,
                    prof_subset_key_ns as f64 / 1_000_000.0,
                    prof_final_weight_ns as f64 / 1_000_000.0,
                    prof_intersection_ns as f64 / 1_000_000.0,
                    prof_eps_closure_ns as f64 / 1_000_000.0,
                    prof_target_build_ns as f64 / 1_000_000.0,
                    prof_target_drain_ns as f64 / 1_000_000.0,
                    prof_target_filter_ns as f64 / 1_000_000.0,
                    prof_target_edge_weight_ns as f64 / 1_000_000.0,
                    prof_target_pre_closure_key_ns as f64 / 1_000_000.0,
                    prof_target_closure_lookup_insert_ns as f64 / 1_000_000.0,
                    prof_target_subset_lookup_ns as f64 / 1_000_000.0,
                    prof_target_add_transition_ns as f64 / 1_000_000.0,
                );
                last_progress_log_at = Instant::now();
            }
        }

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let t_fw = std::time::Instant::now();
        let has_finals: Vec<(u32, Weight)> = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states()[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }
        if profile_enabled { prof_final_weight_ns += t_fw.elapsed().as_nanos() as u64; }

        let labels_processed_before_iter = prof_labels_processed;

        let t_intersect = std::time::Instant::now();
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    if profile_enabled { prof_intersection_calls += 1; }
                    let next_weight = path_weight.intersection(transition_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                        let label_idx = label as usize;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        &mut dense_raw_targets[label_idx]
                    } else if label == DEFAULT_LABEL {
                        default_touched = true;
                        &mut default_raw_targets
                    } else {
                        sparse_raw_targets.entry(label).or_default()
                    };
                    add_target_contribution(target_weights, *target, next_weight);
                }
            }
        }
        if profile_enabled { prof_intersection_ns += t_intersect.elapsed().as_nanos() as u64; }

        let t_target = std::time::Instant::now();
        if profile_enabled {
            let raw_target_entries_this_iter = touched_dense_labels.iter()
                .map(|&label_idx| dense_raw_targets[label_idx].len() as u64)
                .sum::<u64>()
                + if default_touched { default_raw_targets.len() as u64 } else { 0 }
                + sparse_raw_targets.values().map(|contribs| contribs.len() as u64).sum::<u64>();
            prof_total_raw_target_entries += raw_target_entries_this_iter;
            if raw_target_entries_this_iter > prof_max_raw_target_entries_per_iter {
                prof_max_raw_target_entries_per_iter = raw_target_entries_this_iter;
            }
        }

        let mut pre_closure_key: Vec<(u32, usize)> = Vec::new();

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));
            if profile_enabled {
                prof_target_filter_ns += 0;
            }

            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            if profile_enabled {
                match contribs.len() {
                    1 => prof_target_subset_len_1 += 1,
                    2..=4 => prof_target_subset_len_2_4 += 1,
                    5..=16 => prof_target_subset_len_5_16 += 1,
                    _ => prof_target_subset_len_gt_16 += 1,
                }
            }

            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                if nwa.states()[*only_state as usize].epsilons.is_empty() {
                    let t_subset_lookup = std::time::Instant::now();
                    key_buf.clear();
                    key_buf.push((*only_state, only_weight.ptr_key()));
                    let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                        if profile_enabled {
                            prof_subset_map_hits += 1;
                        }
                        existing
                    } else {
                        let new_state = dwa.add_state();
                        subset_map.insert(key_buf.clone(), new_state);
                        worklist.push_back(vec![(*only_state, only_weight.clone())]);
                        supports.push(vec![*only_state]);
                        if profile_enabled {
                            prof_subset_map_inserts += 1;
                        }
                        new_state
                    };
                    if profile_enabled {
                        prof_target_subset_lookup_ns += t_subset_lookup.elapsed().as_nanos() as u64;
                    }
                    let t_add_transition = std::time::Instant::now();
                    dwa.add_transition(from_state, label, to_state, only_weight.clone());
                    if profile_enabled {
                        prof_target_add_transition_ns += t_add_transition.elapsed().as_nanos() as u64;
                        prof_labels_processed += 1;
                        prof_single_target_no_epsilon_fast_path += 1;
                        if label >= 0 && (label as usize) < dense_label_limit {
                            prof_dense_labels_processed += 1;
                        } else if label == DEFAULT_LABEL {
                            prof_default_labels_processed += 1;
                        } else {
                            prof_sparse_labels_processed += 1;
                        }
                    }
                    return;
                }
            }

            let t_pre_closure_key = std::time::Instant::now();
            pre_closure_key.clear();
            pre_closure_key.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            if profile_enabled {
                prof_target_pre_closure_key_ns += t_pre_closure_key.elapsed().as_nanos() as u64;
            }

            let t_eps = std::time::Instant::now();
            let t_closure_lookup_insert = std::time::Instant::now();
            let cached = match closure_cache.entry(pre_closure_key.clone()) {
                Entry::Occupied(entry) => {
                    if profile_enabled {
                        prof_edge_weight_cache_hits += 1;
                    }
                    entry.into_mut()
                }
                Entry::Vacant(entry) => {
                    let t_edge_weight = std::time::Instant::now();
                    let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
                    if profile_enabled {
                        prof_target_edge_weight_ns += t_edge_weight.elapsed().as_nanos() as u64;
                    }
                    if edge_weight.is_empty() {
                        return;
                    }
                    let mut target_subset: FxHashMap<u32, Weight> = contribs
                        .iter()
                        .map(|(state_id, weight)| (*state_id, weight.clone()))
                        .collect();
                    local_epsilon_closure(
                        nwa,
                        &mut weight_by_state,
                        &mut closure_queue,
                        &mut target_subset,
                    );
                    if target_subset.is_empty() {
                        return;
                    }
                    let mut canon: Vec<(u32, Weight)> = target_subset
                        .iter()
                        .filter(|(_, w)| !w.is_empty())
                        .map(|(id, w)| (*id, w.clone()))
                        .collect();
                    canon.sort_unstable_by_key(|(state_id, _)| *state_id);
                    if canon.is_empty() {
                        return;
                    }
                    if profile_enabled {
                        prof_edge_weight_cache_misses += 1;
                    }
                    entry.insert(CachedClosure { canon, edge_weight })
                }
            };
            if profile_enabled {
                prof_target_closure_lookup_insert_ns += t_closure_lookup_insert.elapsed().as_nanos() as u64;
                prof_labels_processed += 1;
                prof_eps_closure_calls += 1;
                prof_eps_closure_ns += t_eps.elapsed().as_nanos() as u64;
                if label >= 0 && (label as usize) < dense_label_limit {
                    prof_dense_labels_processed += 1;
                } else if label == DEFAULT_LABEL {
                    prof_default_labels_processed += 1;
                } else {
                    prof_sparse_labels_processed += 1;
                }
            }

            let t_subset_lookup = std::time::Instant::now();
            key_buf.clear();
            key_buf.extend(cached.canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                if profile_enabled {
                    prof_subset_map_hits += 1;
                }
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(key_buf.clone(), new_state);
                worklist.push_back(cached.canon.clone());
                supports.push(cached.canon.iter().map(|(sid, _)| *sid).collect());
                if profile_enabled {
                    prof_subset_map_inserts += 1;
                }
                new_state
            };
            if profile_enabled {
                prof_target_subset_lookup_ns += t_subset_lookup.elapsed().as_nanos() as u64;
            }
            let t_add_transition = std::time::Instant::now();
            dwa.add_transition(from_state, label, to_state, cached.edge_weight.clone());
            if profile_enabled {
                prof_target_add_transition_ns += t_add_transition.elapsed().as_nanos() as u64;
            }
        };

        let t_drain = std::time::Instant::now();
        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            process_label(label_idx as i32, std::mem::take(&mut dense_raw_targets[label_idx]));
        }

        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }

        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
        if profile_enabled {
            prof_target_drain_ns += t_drain.elapsed().as_nanos() as u64;
        }
        if profile_enabled {
            let labels_processed_this_iter = prof_labels_processed - labels_processed_before_iter;
            if labels_processed_this_iter > prof_max_labels_per_iter {
                prof_max_labels_per_iter = labels_processed_this_iter;
            }
        }
        if profile_enabled { prof_target_build_ns += t_target.elapsed().as_nanos() as u64; }
    }

    // Compute final weights in parallel using rayon.
    let t_parallel_fw = std::time::Instant::now();
    {
        use rayon::prelude::*;
        let final_weights: Vec<(u32, Weight)> = deferred_final_entries
            .par_iter()
            .filter_map(|(state_id, entries)| {
                // Group by final weight pointer to leverage distributivity.
                let mut final_groups: SmallVec<[(usize, &Weight, SmallVec<[&Weight; 4]>); 4]> = SmallVec::new();
                for (nwa_state_id, path_weight) in entries {
                    if let Some(state_final) = nwa.states()[*nwa_state_id as usize].final_weight.as_ref() {
                        let key = state_final.ptr_key();
                        if let Some(group) = final_groups.iter_mut().find(|(k, _, _)| *k == key) {
                            group.2.push(path_weight);
                        } else {
                            let mut pws = SmallVec::new();
                            pws.push(path_weight);
                            final_groups.push((key, state_final, pws));
                        }
                    }
                }
                let final_contributions: SmallVec<[Weight; 4]> = final_groups.into_iter()
                    .filter_map(|(_, final_w, path_weights)| {
                        let pw_union = Weight::union_all(path_weights.into_iter());
                        let contribution = pw_union.intersection(final_w);
                        if contribution.is_empty() { None } else { Some(contribution) }
                    })
                    .collect();
                let final_weight = Weight::union_all(final_contributions.iter());
                if final_weight.is_empty() { None } else { Some((*state_id, final_weight)) }
            })
            .collect();
        for (state_id, weight) in final_weights {
            dwa.set_final_weight(state_id, weight);
        }
    }
    let parallel_fw_ms = t_parallel_fw.elapsed().as_millis();

    if profile_enabled {
        let avg_subset = if prof_iterations > 0 { prof_total_subset_entries as f64 / prof_iterations as f64 } else { 0.0 };
        let avg_labels = if prof_iterations > 0 { prof_labels_processed as f64 / prof_iterations as f64 } else { 0.0 };
        let avg_raw_targets = if prof_iterations > 0 { prof_total_raw_target_entries as f64 / prof_iterations as f64 } else { 0.0 };
        eprintln!(
            "[glrmask/profile][determinize_supports] iterations={} total_subset_entries={} avg_subset_size={:.1} max_subset_size={} intersection_calls={} eps_closure_calls={} labels_processed={} avg_labels_per_iter={:.1} max_labels_per_iter={} raw_target_entries={} avg_raw_targets_per_iter={:.1} max_raw_target_entries_per_iter={} single_target_no_epsilon_fast_path={} subset_len_1={} subset_len_2_4={} subset_len_5_16={} subset_len_gt_16={} subset_map_hits={} subset_map_inserts={} dense_labels={} default_labels={} sparse_labels={} edge_weight_cache_hits={} edge_weight_cache_misses={} deferred_finals={} parallel_final_weight_ms={} subset_key_ms={:.1} collect_finals_ms={:.1} intersection_ms={:.1} eps_closure_ms={:.1} target_build_ms={:.1} target_drain_ms={:.1} target_filter_ms={:.1} target_edge_weight_ms={:.1} target_pre_closure_key_ms={:.1} target_closure_lookup_insert_ms={:.1} target_subset_lookup_ms={:.1} target_add_transition_ms={:.1}",
            prof_iterations, prof_total_subset_entries, avg_subset, prof_max_subset_size,
            prof_intersection_calls, prof_eps_closure_calls, prof_labels_processed,
            avg_labels, prof_max_labels_per_iter,
            prof_total_raw_target_entries,
            avg_raw_targets, prof_max_raw_target_entries_per_iter,
            prof_single_target_no_epsilon_fast_path,
            prof_target_subset_len_1,
            prof_target_subset_len_2_4,
            prof_target_subset_len_5_16,
            prof_target_subset_len_gt_16,
            prof_subset_map_hits,
            prof_subset_map_inserts,
            prof_dense_labels_processed,
            prof_default_labels_processed,
            prof_sparse_labels_processed,
            prof_edge_weight_cache_hits,
            prof_edge_weight_cache_misses,
            deferred_final_entries.len(),
            parallel_fw_ms,
            prof_subset_key_ns as f64 / 1_000_000.0,
            prof_final_weight_ns as f64 / 1_000_000.0,
            prof_intersection_ns as f64 / 1_000_000.0,
            prof_eps_closure_ns as f64 / 1_000_000.0,
            prof_target_build_ns as f64 / 1_000_000.0,
            prof_target_drain_ns as f64 / 1_000_000.0,
            prof_target_filter_ns as f64 / 1_000_000.0,
            prof_target_edge_weight_ns as f64 / 1_000_000.0,
            prof_target_pre_closure_key_ns as f64 / 1_000_000.0,
            prof_target_closure_lookup_insert_ns as f64 / 1_000_000.0,
            prof_target_subset_lookup_ns as f64 / 1_000_000.0,
            prof_target_add_transition_ns as f64 / 1_000_000.0,
        );
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

fn determinize_parser_dwa_with_fallbacks(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) -> DWA {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let dense_label_limit = num_parser_states as usize;
    let mut result = DWA::new(0, 0);

    let mut start_subset = FxHashMap::default();
    start_subset.insert(dwa.start_state(), Weight::all());

    let mut canon_buf: Vec<(u32, Weight)> = start_subset
        .iter()
        .map(|(state_id, weight)| (*state_id, weight.clone()))
        .collect();
    canon_buf.sort_unstable_by_key(|(state_id, _)| *state_id);

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&canon_buf), result.start_state());
    worklist.push_back(canon_buf.clone());

    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut final_contributions: Vec<Weight> = Vec::new();

    while let Some(subset_entries) = worklist.pop_front() {
        let from_state = subset_map[&subset_key(&subset_entries)];

        final_contributions.clear();
        for (state_id, path_weight) in &subset_entries {
            let Some(state_final) = dwa.states()[*state_id as usize].final_weight.as_ref() else {
                continue;
            };
            let contribution = path_weight.intersection(state_final);
            if !contribution.is_empty() {
                final_contributions.push(contribution);
            }
        }
        let final_weight = Weight::union_all(final_contributions.iter());
        if !final_weight.is_empty() {
            result.set_final_weight(from_state, final_weight);
        }

        for (dwa_state_id, path_weight) in &subset_entries {
            let state = &dwa.states()[*dwa_state_id as usize];

            for (&label, (target, transition_weight)) in &state.transitions {
                if label == DEFAULT_LABEL {
                    continue;
                }
                let next_weight = path_weight.intersection(transition_weight);
                if next_weight.is_empty() {
                    continue;
                }

                let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    &mut dense_raw_targets[label_idx]
                } else {
                    sparse_raw_targets.entry(label).or_default()
                };
                add_target_contribution(target_weights, *target, next_weight);
            }

            let Some((default_target, default_weight)) = state.transitions.get(&DEFAULT_LABEL) else {
                continue;
            };
            let fallback_weight = path_weight.intersection(default_weight);
            if fallback_weight.is_empty() {
                continue;
            }

            default_touched = true;
            add_target_contribution(&mut default_raw_targets, *default_target, fallback_weight.clone());

            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    let target_weights = &mut dense_raw_targets[label_idx];
                    add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                } else {
                    let target_weights = sparse_raw_targets.entry(label).or_default();
                    add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                }
            }

            match possible_by_state.get(*dwa_state_id as usize) {
                Some(PossibleOutgoingIds::All) => {
                    for parser_state_id in 0..num_parser_states {
                        let label_idx = parser_state_id as usize;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        let target_weights = &mut dense_raw_targets[label_idx];
                        add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                    }
                }
                Some(PossibleOutgoingIds::Some(ids)) => {
                    for parser_state_id in ids.iter_ones() {
                        let label_idx = parser_state_id;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        let target_weights = &mut dense_raw_targets[label_idx];
                        add_target_contribution(target_weights, *default_target, fallback_weight.clone());
                    }
                }
                Some(PossibleOutgoingIds::Empty) | None => {}
            }
        }

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));
            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
            if edge_weight.is_empty() {
                return;
            }

            key_buf.clear();
            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                key_buf.push((*only_state, only_weight.ptr_key()));
            } else {
                key_buf.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            }

            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                existing
            } else {
                let new_state = result.add_state();
                subset_map.insert(key_buf.clone(), new_state);
                let next_entries: Vec<(u32, Weight)> = contribs.into_iter().collect();
                worklist.push_back(next_entries);
                new_state
            };

            result.add_transition(from_state, label, to_state, edge_weight);
        };

        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            process_label(label_idx as i32, std::mem::take(&mut dense_raw_targets[label_idx]));
        }
        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }
        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
    }

    result
}

fn optimize_parser_dwa_defaults(
    dwa: &mut DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) {
    loop {
        let mut changed = false;

        for (state_id, possible_ids) in possible_by_state.iter().enumerate() {
            let possible_count = match possible_ids {
                PossibleOutgoingIds::Empty => 0,
                PossibleOutgoingIds::All => num_parser_states as usize,
                PossibleOutgoingIds::Some(ids) => ids.count_ones(),
            };
            if possible_count < 2 {
                continue;
            }

            let state = &dwa.states()[state_id];

            let mut actual_positive = BitSet::new(num_parser_states as usize);
            for &label in state.transitions.keys() {
                if let Some(ps) = parser_state_label(label, num_parser_states) {
                    actual_positive.set(ps as usize);
                }
            }
            match possible_ids {
                PossibleOutgoingIds::Empty => continue,
                PossibleOutgoingIds::All => {
                    if actual_positive.count_ones() != num_parser_states as usize {
                        continue;
                    }
                }
                PossibleOutgoingIds::Some(ids) => {
                    if actual_positive != *ids {
                        continue;
                    }
                }
            }

            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            match possible_ids {
                PossibleOutgoingIds::Empty => continue,
                PossibleOutgoingIds::All => {
                    for ps in 0..num_parser_states {
                        let label = ps as i32;
                        let Some((target, weight)) = state.transitions.get(&label) else {
                            valid = false;
                            break;
                        };
                        match shared_target {
                            Some(existing) if existing != *target => {
                                valid = false;
                                break;
                            }
                            None => shared_target = Some(*target),
                            _ => {}
                        }
                        default_weight = Some(match default_weight {
                            Some(existing) => existing.intersection(weight),
                            None => weight.clone(),
                        });
                    }
                }
                PossibleOutgoingIds::Some(ids) => {
                    for ps in ids.iter_ones() {
                        let label = ps as i32;
                        let Some((target, weight)) = state.transitions.get(&label) else {
                            valid = false;
                            break;
                        };
                        match shared_target {
                            Some(existing) if existing != *target => {
                                valid = false;
                                break;
                            }
                            None => shared_target = Some(*target),
                            _ => {}
                        }
                        default_weight = Some(match default_weight {
                            Some(existing) => existing.intersection(weight),
                            None => weight.clone(),
                        });
                    }
                }
            }

            let Some(target) = shared_target else { continue };
            let Some(default_weight) = default_weight else { continue };
            if !valid || default_weight.is_empty() {
                continue;
            }

            let state = &mut dwa.states_mut()[state_id];
            let entry = state.transitions.entry(DEFAULT_LABEL);
            match entry {
                std::collections::btree_map::Entry::Occupied(mut occ) => {
                    let (existing_target, existing_weight) = occ.get_mut();
                    if *existing_target == target {
                        let updated = existing_weight.union(&default_weight);
                        if updated != *existing_weight {
                            *existing_weight = updated;
                            changed = true;
                        }
                    }
                }
                std::collections::btree_map::Entry::Vacant(vac) => {
                    vac.insert((target, default_weight));
                    changed = true;
                }
            }
        }

        for state_id in 0..dwa.states().len() {
            let Some((default_target, default_weight)) =
                dwa.states()[state_id].transitions.get(&DEFAULT_LABEL).cloned()
            else {
                continue;
            };

            let target_final = dwa.states()[default_target as usize].final_weight.clone();
            let Some(target_final) = target_final else { continue };
            let lifted = default_weight.intersection(&target_final);
            if lifted.is_empty() {
                continue;
            }

            if union_final_weight(&mut dwa.states_mut()[state_id].final_weight, lifted.clone()) {
                changed = true;
            }

            let state = &mut dwa.states_mut()[state_id];
            let mut to_remove = Vec::new();
            for (&label, (_, weight)) in state.transitions.iter_mut() {
                let new_weight = weight.difference(&lifted);
                if new_weight != *weight {
                    *weight = new_weight;
                    changed = true;
                }
                if weight.is_empty() {
                    to_remove.push(label);
                }
            }
            for label in to_remove {
                state.transitions.remove(&label);
            }
        }

        for state_id in 0..dwa.states().len() {
            let Some(&(default_target, ref default_weight)) =
                dwa.states()[state_id].transitions.get(&DEFAULT_LABEL)
            else {
                continue;
            };
            let default_target = default_target;
            let default_weight = default_weight.clone();

            let state = &mut dwa.states_mut()[state_id];
            let mut to_remove = Vec::new();
            for (&label, (target, weight)) in state.transitions.iter_mut() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if *target != default_target {
                    continue;
                }
                let new_weight = weight.difference(&default_weight);
                if new_weight != *weight {
                    *weight = new_weight;
                    changed = true;
                }
                if weight.is_empty() {
                    to_remove.push(label);
                }
            }
            for label in to_remove {
                state.transitions.remove(&label);
            }
        }

        if !changed {
            break;
        }
    }
}

fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {
    for state_id in 0..dwa.states().len() {
        let Some(final_weight) = dwa.states()[state_id].final_weight.clone() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        let state = &mut dwa.states_mut()[state_id];
        let mut to_remove = Vec::new();
        for (&label, (_, weight)) in state.transitions.iter_mut() {
            let new_weight = weight.difference(&final_weight);
            if new_weight != *weight {
                *weight = new_weight;
            }
            if weight.is_empty() {
                to_remove.push(label);
            }
        }
        for label in to_remove {
            state.transitions.remove(&label);
        }
    }
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    *nwa.states_mut() = vec![crate::automata::weighted::nwa::NWAState::default(); dwa.states().len()];
    nwa.set_start_states(vec![dwa.start_state()]);

    for (state_id, state) in dwa.states().iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.states_mut()[state_id].final_weight = Some(final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.states_mut()[state_id]
                .transitions
                .entry(label)
                .or_default()
                .push((*target, weight.clone()));
        }
    }

    nwa
}

fn ensure_default_transitions_are_fallbacks(
    nwa: &mut NWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) {
    for state_id in 0..nwa.states().len() {
        let state = &nwa.states()[state_id];

        let Some(default_targets) = state.transitions.get(&DEFAULT_LABEL).cloned() else {
            continue;
        };

        if default_targets.is_empty() {
            continue;
        }

        let mut fallback_labels = BTreeSet::new();
        for &label in state.transitions.keys() {
            if label != DEFAULT_LABEL {
                fallback_labels.insert(label);
            }
        }

        match possible_by_state.get(state_id) {
            Some(PossibleOutgoingIds::Empty) | None => {}
            Some(PossibleOutgoingIds::All) => {
                for parser_state_id in 0..num_parser_states {
                    fallback_labels.insert(encode_positive_label(parser_state_id));
                }
            }
            Some(PossibleOutgoingIds::Some(ids)) => {
                for parser_state_id in ids.iter_ones() {
                    fallback_labels.insert(encode_positive_label(parser_state_id as u32));
                }
            }
        }

        for label in fallback_labels {
            let state_mut = &mut nwa.states_mut()[state_id];
            for (dst, weight) in &default_targets {
                add_or_union_transition(state_mut, label, *dst, weight.clone());
            }
        }
    }
}

fn build_parser_nwa_from_terminal_dwa(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: Templates,
) -> Option<(NWA, ParserDwaPhaseProfile)> {
    let mut profile = ParserDwaPhaseProfile::default();

    let build_state_summaries_started_at = Instant::now();
    let states = build_state_summaries(terminal_dwa, grammar, &templates);
    profile.build_state_summaries_ms = elapsed_ms(build_state_summaries_started_at);

    let mut arena = NWA::new(0, 0);
    let mut body_memo = vec![None; states.len()];
    let mut concatenated_branches: ConcatMemo = FxHashMap::default();
    let compose_started_at = Instant::now();
    let parser_body = compose_state(
        terminal_dwa.start_state(),
        &states,
        &mut arena,
        &mut body_memo,
        &mut concatenated_branches,
    )?;
    profile.compose_state_ms = elapsed_ms(compose_started_at);

    arena.set_start_states(parser_body.start_states.clone());

    Some((arena, profile))
}

#[cfg(test)]
pub(crate) fn debug_build_parser_nwa_from_terminal_dwa(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: Templates,
) -> Option<NWA> {
    build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates).map(|(nwa, _)| nwa)
}

#[cfg(test)]
pub(crate) fn build_parser_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    let (terminal_dwa, templates) = rayon::join(
        || build_terminal_dwa_for_existing_id_map(grammar, tokenizer, vocab, id_map, ignore_terminal),
        || {
            let characterizations = characterize_terminals(table, grammar);
            Templates::from_characterizations(&characterizations)
        },
    );

    let mut parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
        table,
        grammar,
        &terminal_dwa,
        templates,
        vocab,
        id_map,
    );
    parser_dwa.clip_weights(id_map.max_internal_token_id());
    parser_dwa
}

pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal_dwa: &DWA,
    templates: Templates,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    let total_started_at = Instant::now();
    let Some((mut parser_nwa, mut profile)) =
        build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates)
    else {
        return DWA::new(0, 0);
    };

    if parser_dwa_profile_enabled() {
        let nwa_transitions: usize = parser_nwa.states().iter()
            .map(|s| s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] phase=pre_resolve_negatives nwa_states={} nwa_transitions={} terminal_dwa_states={}",
            parser_nwa.states().len(), nwa_transitions, terminal_dwa.states().len(),
        );
    }

    let resolve_negatives_started_at = Instant::now();
    resolve_negative_codes_in_nwa(&mut parser_nwa);
    profile.resolve_negatives_ms = elapsed_ms(resolve_negatives_started_at);

    if parser_dwa_profile_enabled() {
        let nwa_transitions: usize = parser_nwa.states().iter()
            .map(|s| s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] phase=post_resolve_negatives nwa_states={} nwa_transitions={} terminal_dwa_states={}",
            parser_nwa.states().len(), nwa_transitions, terminal_dwa.states().len(),
        );
    }

    let determinize_supports_started_at = Instant::now();
    let determinized = determinize_with_supports(&parser_nwa, Some(table.num_states));
    profile.determinize_supports_ms = elapsed_ms(determinize_supports_started_at);
    let mut parser_dwa_pre_minimize = determinized.dwa;

    if parser_dwa_profile_enabled() {
        let dwa_transitions: usize = parser_dwa_pre_minimize.states().iter()
            .map(|s| s.transitions.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] dwa_states={} dwa_transitions={} minimized_later",
            parser_dwa_pre_minimize.states().len(), dwa_transitions,
        );
    }

    let viable_suffix_started_at = Instant::now();
    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        table.num_states,
    );
    profile.viable_suffix_ms = elapsed_ms(viable_suffix_started_at);

    let optimize_defaults_started_at = Instant::now();
    if std::env::var_os("GLRMASK_DISABLE_PARSER_DWA_DEFAULTS_OPT").is_none() {
        optimize_parser_dwa_defaults(
            &mut parser_dwa_pre_minimize,
            &possible_by_state,
            table.num_states,
        );
    }
    profile.optimize_defaults_ms = elapsed_ms(optimize_defaults_started_at);

    let subtract_final_started_at = Instant::now();
    if std::env::var_os("GLRMASK_DISABLE_PARSER_DWA_SUBTRACT_FINAL").is_none() {
        subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    }
    profile.subtract_final_ms = elapsed_ms(subtract_final_started_at);

    let ensure_fallback_started_at = Instant::now();
    if std::env::var_os("GLRMASK_USE_NWA_FALLBACK_DETERMINIZE").is_some() {
        let mut nwa_for_fallback = parser_dwa_pre_minimize.to_nwa();
        profile.fallback_to_nwa_ms = elapsed_ms(ensure_fallback_started_at);

        let ensure_fallback_step_started_at = Instant::now();
        ensure_default_transitions_are_fallbacks(&mut nwa_for_fallback, &possible_by_state, table.num_states);
        profile.ensure_fallback_ms = elapsed_ms(ensure_fallback_step_started_at);

        let determinize_fallback_started_at = Instant::now();
        parser_dwa_pre_minimize = determinize_with_supports(&nwa_for_fallback, Some(table.num_states)).dwa;
        profile.determinize_fallback_ms = elapsed_ms(determinize_fallback_started_at);
    } else {
        profile.fallback_to_nwa_ms = 0.0;
        profile.ensure_fallback_ms = 0.0;

        let determinize_fallback_started_at = Instant::now();
        parser_dwa_pre_minimize = determinize_parser_dwa_with_fallbacks(
            &parser_dwa_pre_minimize,
            &possible_by_state,
            table.num_states,
        );
        profile.determinize_fallback_ms = elapsed_ms(determinize_fallback_started_at);
    }

    profile.determinize_after_defaults_ms = elapsed_ms(ensure_fallback_started_at);

    let minimize_started_at = Instant::now();
    let minimized = if std::env::var_os("GLRMASK_DISABLE_PARSER_DWA_MINIMIZE").is_some() {
        parser_dwa_pre_minimize.clone()
    } else {
        minimize_from_env(
            &parser_dwa_pre_minimize,
            "GLRMASK_MINIMIZE_PARSER_DWA",
            minimize_fast,
        )
    };
    profile.minimize_ms = elapsed_ms(minimize_started_at);
    profile.total_ms = elapsed_ms(total_started_at);

    if parser_dwa_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_dwa] build_state_summaries_ms={:.3} compose_state_ms={:.3} resolve_negatives_ms={:.3} viable_suffix_ms={:.3} determinize_supports_ms={:.3} optimize_defaults_ms={:.3} subtract_final_ms={:.3} fallback_to_nwa_ms={:.3} ensure_fallback_ms={:.3} determinize_fallback_ms={:.3} determinize_after_defaults_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            profile.build_state_summaries_ms,
            profile.compose_state_ms,
            profile.resolve_negatives_ms,
            profile.viable_suffix_ms,
            profile.determinize_supports_ms,
            profile.optimize_defaults_ms,
            profile.subtract_final_ms,
            profile.fallback_to_nwa_ms,
            profile.ensure_fallback_ms,
            profile.determinize_fallback_ms,
            profile.determinize_after_defaults_ms,
            profile.minimize_ms,
            profile.total_ms,
        );
    }

    if std::env::var("GLRMASK_DEBUG_PARSER_DWA_DUMP").map_or(false, |v| v == "1") {
        emit_parser_dwa_token_map(&minimized, vocab, id_map);
        emit_parser_dwa_debug_dump(&minimized);
    }

    minimized
}

fn emit_parser_dwa_token_map(dwa: &DWA, vocab: &Vocab, id_map: &InternalIdMap) {
    use super::id_map_and_terminal_dwa::l2p::nwa_builder::internal_vocab_entries;
    let internal_vocab = internal_vocab_entries(vocab, id_map);
    let internal_bytes: std::collections::BTreeMap<u32, &[u8]> =
        internal_vocab.iter().map(|(id, bytes)| (*id, bytes.as_slice())).collect();
    let mut referenced_tokens = std::collections::BTreeSet::new();
    for state in dwa.states() {
        for (_, (_, weight)) in &state.transitions {
            for tid in weight.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
        if let Some(fw) = &state.final_weight {
            for tid in fw.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
    }
    for tid in &referenced_tokens {
        if let Some(bytes) = internal_bytes.get(tid) {
            let originals = id_map.vocab_tokens.internal_to_originals.get(*tid as usize)
                .map(|v| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .unwrap_or_else(|| "?".into());
            eprintln!(
                "[glrmask/debug][parser_dwa][token_map] internal={} originals=[{}] bytes={:?}",
                tid, originals, String::from_utf8_lossy(bytes)
            );
        }
    }
}

fn emit_parser_dwa_debug_dump(dwa: &DWA) {
    let num_states = dwa.num_states() as usize;
    let start_state = dwa.start_state() as usize;
    let mut incoming_counts = vec![0usize; num_states];
    let mut outgoing_counts = vec![0usize; num_states];
    let mut final_states = 0usize;
    let mut self_loops = 0usize;
    let mut transitions_to_start = 0usize;
    let mut transitions_from_start = 0usize;
    let mut transitions_from_start_to_start = 0usize;

    for (state_id, state) in dwa.states().iter().enumerate() {
        outgoing_counts[state_id] = state.transitions.len();
        for (_, (target, _)) in &state.transitions {
            incoming_counts[*target as usize] += 1;
            if *target as usize == start_state {
                transitions_to_start += 1;
            }
            if state_id == start_state {
                transitions_from_start += 1;
            }
            if state_id == start_state && *target as usize == start_state {
                transitions_from_start_to_start += 1;
            }
            if *target as usize == state_id {
                self_loops += 1;
            }
        }
        if state.final_weight.is_some() {
            final_states += 1;
        }
    }

    eprintln!(
        "[glrmask/debug][parser_dwa][dump] states={} final_states={} self_loops={} to_start={} from_start={} from_start_to_start={}",
        num_states, final_states, self_loops, transitions_to_start, transitions_from_start, transitions_from_start_to_start,
    );

    for (state_id, state) in dwa.states().iter().enumerate() {
        let incoming = incoming_counts[state_id];
        let outgoing = outgoing_counts[state_id];
        let to_start = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == start_state)
            .count();
        let self_loop_count = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == state_id)
            .count();
        let final_weight = state
            .final_weight
            .as_ref()
            .map(|weight| format!("{weight}"))
            .unwrap_or_else(|| "none".to_string());
        let start_mark = if state_id == start_state {
            " [START]"
        } else {
            ""
        };

        eprintln!(
            "[glrmask/debug][parser_dwa][state] id={}{} incoming={} outgoing={} to_start={} self_loops={} final={}",
            state_id,
            start_mark,
            incoming,
            outgoing,
            to_start,
            self_loop_count,
            final_weight,
        );

        for (label, (target, weight)) in &state.transitions {
            eprintln!("    {label} -> State {target}");
            eprintln!("      weight: {weight}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constraint;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::labels::{DEFAULT_LABEL, encode_positive_label};
    use crate::compiler::glr::parser::stack_may_advance_on;
    use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
    use crate::grammar::flat::GrammarDef;
    use crate::grammar::flat::tests::*;
    use range_set_blaze::RangeSetBlaze;

    fn token_weight(tokens: &[u32]) -> Weight {
        Weight::from_token_set_for_tsid(
            0,
            RangeSetBlaze::from_iter(tokens.iter().copied().map(|token| token..=token)),
        )
    }

    fn make_vocab_and_preprocessing(gdef: &GrammarDef) -> (Vocab, Tokenizer, InternalIdMap) {
        let tok = crate::compiler::compile::build_tokenizer(gdef);

        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (i, td) in gdef.terminals.iter().enumerate() {
            entries.push((i as u32, td.name().as_bytes().to_vec()));
        }
        let vocab = Vocab::new(entries, None);
        let id_map = crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences(&tok, &vocab, &std::collections::BTreeMap::new(), None, None);
        (vocab, tok, id_map)
    }

    #[test]
    fn test_build_parser_dwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp, None);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_build_parser_dwa_choice() {
        let gdef = choice_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp, None);
        assert!(dwa.num_states() > 0);
    }

    #[ignore = "diagnostic for whether the minimized split-boundary token is lost before or during determinization"]
    #[test]
    fn diagnose_minimized_split_boundary_nwa_vs_determinized() {
        let grammar_str = r#"
start start;
t A_EXACT ::= "a"{32};
t A_UP_TO_32 ::= "a"{1,2} "\"";
nt start ::= (A_EXACT{4} | A_EXACT{5}) A_UP_TO_32;
"#;
        let named = crate::grammar::glrm::from_glrm(grammar_str).unwrap();
        let factored = crate::grammar::factoring::factor_named_grammar(named);
        let gdef = crate::grammar::ast::lower(&factored).unwrap();
        let analyzed = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&analyzed);
        let tokenizer = crate::compiler::compile::build_tokenizer(&gdef);
        let vocab = Vocab::new(vec![(0, b"aa\"".to_vec())], None);
        let id_map = crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences(
            &tokenizer,
            &vocab,
            &std::collections::BTreeMap::new(),
            None,
            None,
        );
        let terminal_dwa = build_terminal_dwa_for_existing_id_map(&analyzed, &tokenizer, &vocab, &id_map, None);
        let templates = Templates::from_characterizations(&characterize_terminals(&table, &analyzed));
        let parser_nwa = debug_build_parser_nwa_from_terminal_dwa(&terminal_dwa, &analyzed, templates.clone())
            .expect("parser NWA should build");
        let determinized = determinize_with_supports(&parser_nwa, None);

        let constraint = Constraint::from_glrm_grammar(grammar_str, &vocab).unwrap();
        let prefix = [b'a'; 159];
        let mut mask_state = constraint.start();
        mask_state.commit_bytes(&prefix).unwrap();
        let (&tokenizer_state, gss) = mask_state.state.iter().next().expect("live prefix tokenizer state");
        let (chain_states, _acc, _tail) = gss.extract_chain_and_tail().expect("expected chain-shaped live GSS");
        let internal_tsid = constraint.internal_tsid_for_state(tokenizer_state);
        let internal_token_0 = constraint.original_token_to_internal[0usize];

        let candidate_terminals: Vec<_> = constraint
            .possible_matches_for_state(tokenizer_state)
            .into_iter()
            .filter(|(terminal, tokens)| tokens.contains(0) && stack_may_advance_on(&constraint.table, gss, *terminal))
            .map(|(terminal, _)| terminal)
            .collect();
        let mut template_accepts_candidate = false;
        for terminal in &candidate_terminals {
            let template = templates.by_terminal.get(terminal).expect("candidate terminal template");
            let mut template_state = template.start_state;
            let mut alive = true;
            for (index, parser_state) in chain_states.iter().copied().enumerate() {
                if index == 0 && template_state == template.start_state && parser_state == 0 {
                    continue;
                }
                let node = &template.states[template_state as usize];
                let label = encode_positive_label(parser_state);
                let Some(&target) = node.transitions.get(&label).or_else(|| node.transitions.get(&DEFAULT_LABEL)) else {
                    alive = false;
                    break;
                };
                template_state = target;
            }
            if alive && template.states[template_state as usize].is_accepting {
                template_accepts_candidate = true;
            }
        }

        let mut current_subset = rustc_hash::FxHashMap::<u32, Weight>::default();
        for &state_id in parser_nwa.start_states() {
            current_subset.insert(state_id, Weight::all());
        }
        local_epsilon_closure(&parser_nwa, &mut vec![None; parser_nwa.states().len()], &mut std::collections::VecDeque::new(), &mut current_subset);

        let mut nwa_token_0_reachable = false;
        for (index, parser_state) in chain_states.iter().copied().enumerate() {
            if index == 0 && parser_state == 0 {
                continue;
            }

            let label = encode_positive_label(parser_state);
            let mut next_subset = rustc_hash::FxHashMap::<u32, Weight>::default();
            for (&nwa_state_id, path_weight) in &current_subset {
                let state = &parser_nwa.states()[nwa_state_id as usize];
                for candidate_label in [label, DEFAULT_LABEL] {
                    let Some(targets) = state.transitions.get(&candidate_label) else {
                        continue;
                    };
                    for (target, edge_weight) in targets {
                        let contribution = path_weight.intersection(edge_weight);
                        if contribution.is_empty() {
                            continue;
                        }
                        let entry = next_subset.entry(*target).or_insert_with(Weight::empty);
                        *entry = entry.union(&contribution);
                    }
                }
            }

            local_epsilon_closure(
                &parser_nwa,
                &mut vec![None; parser_nwa.states().len()],
                &mut std::collections::VecDeque::new(),
                &mut next_subset,
            );
            current_subset = next_subset;

            for (&nwa_state_id, path_weight) in &current_subset {
                let Some(final_weight) = parser_nwa.states()[nwa_state_id as usize].final_weight.as_ref() else {
                    continue;
                };
                let contribution = path_weight.intersection(final_weight);
                if contribution.tokens_for_tsid(internal_tsid).contains(internal_token_0) {
                    nwa_token_0_reachable = true;
                }
            }
        }

        let mut dwa_token_0_reachable = false;
        let parser_dwa = &determinized.dwa;
        let mut wa_state = parser_dwa.start_state();
        for (index, parser_state) in chain_states.iter().copied().enumerate() {
            if index == 0 && parser_state == 0 {
                continue;
            }
            let state = &parser_dwa.states()[wa_state as usize];
            let label = encode_positive_label(parser_state);
            let Some((target, _weight)) = state.transitions.get(&label).or_else(|| state.transitions.get(&DEFAULT_LABEL)) else {
                break;
            };
            wa_state = *target;
            let final_tokens = parser_dwa.states()[wa_state as usize]
                .final_weight
                .as_ref()
                .map(|weight| weight.tokens_for_tsid(internal_tsid))
                .unwrap_or_default();
            if final_tokens.contains(internal_token_0) {
                dwa_token_0_reachable = true;
            }
        }

        assert_eq!(
            (nwa_token_0_reachable, dwa_token_0_reachable),
            (false, false),
            "the minimized witness token is already lost before determinization"
        );
        assert!(!candidate_terminals.is_empty(), "expected at least one live candidate terminal carrying token 0");
        assert!(
            !template_accepts_candidate,
            "candidate terminal template unexpectedly accepts the live chain; candidate_terminals={candidate_terminals:?}"
        );
    }

    #[ignore = "diagnostic for sparse o82710 reachability across resolved parser NWA and DWA stages"]
    #[test]
    fn diagnose_sparse_o82710_resolved_nwa_vs_dwas() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "aside": { "type": "boolean" },
                "autoplay": { "type": "boolean" },
                "css_class": {
                    "type": "string",
                    "pattern": "^[\\w\\s-]+$"
                },
                "description": {
                    "type": "string",
                    "minLength": 0,
                    "maxLength": 5000
                }
            },
            "required": ["id"],
            "additionalProperties": true
        }"##;

        fn dwa_reaches_token_along_chain(
            dwa: &DWA,
            chain_states: &[u32],
            internal_tsid: u32,
            internal_token: u32,
        ) -> bool {
            let mut wa_state = dwa.start_state();
            for (index, parser_state) in chain_states.iter().copied().enumerate() {
                if index == 0 && parser_state == 0 {
                    continue;
                }
                let state = &dwa.states()[wa_state as usize];
                let label = encode_positive_label(parser_state);
                let Some((target, _weight)) = state.transitions.get(&label).or_else(|| state.transitions.get(&DEFAULT_LABEL)) else {
                    return false;
                };
                wa_state = *target;
                let final_tokens = dwa.states()[wa_state as usize]
                    .final_weight
                    .as_ref()
                    .map(|weight| weight.tokens_for_tsid(internal_tsid))
                    .unwrap_or_default();
                if final_tokens.contains(internal_token) {
                    return true;
                }
            }
            false
        }

        fn nwa_reaches_token_along_chain(
            nwa: &NWA,
            chain_states: &[u32],
            internal_tsid: u32,
            internal_token: u32,
        ) -> bool {
            let mut current_subset = rustc_hash::FxHashMap::<u32, Weight>::default();
            for &state_id in nwa.start_states() {
                current_subset.insert(state_id, Weight::all());
            }
            local_epsilon_closure(
                nwa,
                &mut vec![None; nwa.states().len()],
                &mut std::collections::VecDeque::new(),
                &mut current_subset,
            );

            for (index, parser_state) in chain_states.iter().copied().enumerate() {
                if index == 0 && parser_state == 0 {
                    continue;
                }

                let label = encode_positive_label(parser_state);
                let mut next_subset = rustc_hash::FxHashMap::<u32, Weight>::default();
                for (&nwa_state_id, path_weight) in &current_subset {
                    let state = &nwa.states()[nwa_state_id as usize];
                    for candidate_label in [label, DEFAULT_LABEL] {
                        let Some(targets) = state.transitions.get(&candidate_label) else {
                            continue;
                        };
                        for (target, edge_weight) in targets {
                            let contribution = path_weight.intersection(edge_weight);
                            if contribution.is_empty() {
                                continue;
                            }
                            let entry = next_subset.entry(*target).or_insert_with(Weight::empty);
                            *entry = entry.union(&contribution);
                        }
                    }
                }

                local_epsilon_closure(
                    nwa,
                    &mut vec![None; nwa.states().len()],
                    &mut std::collections::VecDeque::new(),
                    &mut next_subset,
                );
                current_subset = next_subset;

                for (&nwa_state_id, path_weight) in &current_subset {
                    let Some(final_weight) = nwa.states()[nwa_state_id as usize].final_weight.as_ref() else {
                        continue;
                    };
                    let contribution = path_weight.intersection(final_weight);
                    if contribution.tokens_for_tsid(internal_tsid).contains(internal_token) {
                        return true;
                    }
                }
            }
            false
        }

        let grammar = crate::import::json_schema::json_schema_to_grammar(schema)
            .expect("schema should lower to a grammar");
        let (prepared, _prepared_tokenizer) =
            crate::compiler::grammar::transforms::prepare_grammar_for_compile(&grammar);
        let analyzed = AnalyzedGrammar::from_grammar_def(&prepared);
        let table = GLRTable::build(&analyzed);
        let tokenizer = crate::compiler::compile::build_tokenizer(&prepared);
        let vocab = Vocab::new(
            vec![(68439u32, b"'];?>\"".to_vec()), (99925u32, b" Vimeo".to_vec())],
            None,
        );
        let id_map = crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::combined::analyze_equivalences(
            &tokenizer,
            &vocab,
            &std::collections::BTreeMap::new(),
            None,
            None,
        );
        let terminal_dwa = build_terminal_dwa_for_existing_id_map(
            &analyzed,
            &tokenizer,
            &vocab,
            &id_map,
            None,
        );
        let templates = Templates::from_characterizations(&characterize_terminals(&table, &analyzed));
        let mut resolved_parser_nwa = debug_build_parser_nwa_from_terminal_dwa(&terminal_dwa, &analyzed, templates.clone())
            .expect("parser NWA should build");
        resolve_negative_codes_in_nwa(&mut resolved_parser_nwa);

        let specialized_dwa = determinize_with_supports(&resolved_parser_nwa, None).dwa;
        let generic_dwa = crate::automata::weighted_u32::determinize::determinize(&resolved_parser_nwa)
            .expect("generic determinization should succeed");
        let full_pipeline_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
            &table,
            &analyzed,
            &terminal_dwa,
            templates,
            &vocab,
            &id_map,
        );

        let constraint = Constraint::from_json_schema(schema, &vocab).unwrap();
        let mut prefix = Vec::from(
            b"{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"".as_slice(),
        );
        prefix.extend(std::iter::repeat(b"This is a Vimeo video block. ".as_slice()).take(79).flatten().copied());
        prefix.extend_from_slice(b"This is a");

        let mut state = constraint.start();
        state.commit_bytes(&prefix).unwrap();
        let (&tokenizer_state, gss) = state.state.iter().next().expect("live prefix tokenizer state");
        let (chain_states, _acc, _tail) = gss.extract_chain_and_tail().expect("expected chain-shaped live GSS");
        let internal_tsid = constraint.internal_tsid_for_state(tokenizer_state);
        let internal_token = constraint.original_token_to_internal[68439usize];

        let resolved_nwa_reachable = nwa_reaches_token_along_chain(
            &resolved_parser_nwa,
            &chain_states,
            internal_tsid,
            internal_token,
        );
        let specialized_reachable = dwa_reaches_token_along_chain(
            &specialized_dwa,
            &chain_states,
            internal_tsid,
            internal_token,
        );
        let generic_reachable = dwa_reaches_token_along_chain(
            &generic_dwa,
            &chain_states,
            internal_tsid,
            internal_token,
        );
        let full_pipeline_reachable = dwa_reaches_token_along_chain(
            &full_pipeline_dwa,
            &chain_states,
            internal_tsid,
            internal_token,
        );

        assert_eq!(
            (
                resolved_nwa_reachable,
                specialized_reachable,
                generic_reachable,
                full_pipeline_reachable,
            ),
            (true, false, false, true),
            "chain_states={chain_states:?} tokenizer_state={tokenizer_state} internal_tsid={internal_tsid} internal_token={internal_token}"
        );
    }
}
