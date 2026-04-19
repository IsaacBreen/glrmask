use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[cfg(test)]
use crate::Vocab;
#[cfg(test)]
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::{minimize_fast, minimize_from_env};
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::GLRTable;
use crate::grammar::flat::TerminalID;
#[cfg(test)]
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
type ConcatMemo = HashMap<(usize, u32), Option<NwaBody>>;

#[derive(Debug, Clone, Copy, Default)]
struct ParserDwaPhaseProfile {
    build_state_summaries_ms: f64,
    compose_state_ms: f64,
    resolve_negatives_ms: f64,
    viable_suffix_ms: f64,
    determinize_supports_ms: f64,
    optimize_defaults_ms: f64,
    subtract_final_ms: f64,
    determinize_after_defaults_ms: f64,
    minimize_ms: f64,
    total_ms: f64,
}

fn parser_dwa_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some()
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

fn accepting_nwa(final_weight: &Weight) -> Option<NWA> {
    if final_weight.is_empty() {
        return None;
    }

    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states.push(start);
    nwa.set_final_weight(start, final_weight.clone());
    Some(nwa)
}

fn group_terminal_edges_by_target(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, TerminalBundle> {
    let Some(state) = terminal_dwa.states.get(state_id as usize) else {
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
        Vec::with_capacity(terminal_dwa.states.len());
    let mut bundle_ids_by_signature: HashMap<BundleSignature, usize> = HashMap::new();
    let mut unique_bundles: Vec<TerminalBundle> = Vec::new();

    for (state_id, _state) in terminal_dwa.states.iter().enumerate() {
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
            terminal_dwa.states.len(),
            unique_bundles.len(),
            pending_branches_by_state.iter().map(|b| b.len()).sum::<usize>(),
        );
    }

    let built_bundles: Vec<Arc<NWA>> = {
        use rayon::prelude::*;
        unique_bundles
            .par_iter()
            .enumerate()
            .map(|(bundle_id, bundle)| {
                let build_started_at = Instant::now();
                let nwa = Arc::new(templates.build_bundle(bundle));
                if parser_dwa_profile_enabled() {
                    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
                    let nwa_transitions: usize = nwa
                        .states
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
                        nwa.states.len(),
                        nwa_transitions,
                        build_ms,
                    );
                }
                nwa
            })
            .collect()
    };

    terminal_dwa
        .states
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
        .states
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
        if let Some(state) = nwa.states.get(state_id as usize) {
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
        let Some(state) = nwa.states.get(state_id as usize) else {
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

fn determinize_with_supports(nwa: &NWA) -> DeterminizedDwaWithSupports {
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

    let num_nwa_states = nwa.states.len();

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
            if let Some(state) = nwa.states.get(state_id as usize) {
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
            let Some(state) = nwa.states.get(state_id as usize) else {
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
    for &state_id in &nwa.start_states {
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
    subset_map.insert(subset_key(&canon_buf), dwa.start_state);
    worklist.push_back(canon_buf.clone());

    let mut raw_targets: FxHashMap<i32, FxHashMap<u32, Vec<Weight>>> = FxHashMap::default();
    // Reusable target subset map — cleared and reused each iteration.
    let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
    // Memoize local epsilon-closure outputs keyed by pre-closure weighted subsets.
    let mut closure_cache: FxHashMap<Vec<(u32, usize)>, Vec<(u32, Weight)>> = FxHashMap::default();
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
        }

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let t_fw = std::time::Instant::now();
        let has_finals: Vec<(u32, Weight)> = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }
        if profile_enabled { prof_final_weight_ns += t_fw.elapsed().as_nanos() as u64; }

        let t_intersect = std::time::Instant::now();
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    if profile_enabled { prof_intersection_calls += 1; }
                    let next_weight = path_weight.intersection(transition_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    raw_targets.entry(label).or_default().entry(*target).or_default().push(next_weight);
                }
            }
        }
        if profile_enabled { prof_intersection_ns += t_intersect.elapsed().as_nanos() as u64; }

        let t_target = std::time::Instant::now();
        let raw_target_entries: Vec<(i32, FxHashMap<u32, Vec<Weight>>)> =
            raw_targets.drain().collect();
        if profile_enabled {
            for (_, contribs) in &raw_target_entries {
                for (_, weights) in contribs {
                    prof_total_raw_target_entries += weights.len() as u64;
                }
            }
        }

        type LabelResult = (i32, Weight, Vec<(u32, Weight)>, Vec<u32>);
        let mut label_results: Vec<LabelResult> = Vec::with_capacity(raw_target_entries.len());
        let mut pre_closure_key: Vec<(u32, usize)> = Vec::new();

        for (label, contribs) in raw_target_entries {
            if contribs.is_empty() {
                continue;
            }

            target_subset.clear();
            for (dst, weights) in contribs {
                let combined = Weight::union_all(weights.iter());
                if !combined.is_empty() {
                    target_subset.insert(dst, combined);
                }
            }
            if target_subset.is_empty() {
                continue;
            }

            let edge_weight = Weight::union_all(target_subset.values());
            if edge_weight.is_empty() {
                continue;
            }

            pre_closure_key.clear();
            pre_closure_key.extend(target_subset.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            pre_closure_key.sort_unstable_by_key(|(sid, _)| *sid);

            let t_eps = std::time::Instant::now();
            let canon = if let Some(cached) = closure_cache.get(&pre_closure_key) {
                cached.clone()
            } else {
                local_epsilon_closure(nwa, &mut weight_by_state, &mut closure_queue, &mut target_subset);
                if target_subset.is_empty() {
                    continue;
                }
                let mut canon: Vec<(u32, Weight)> = target_subset
                    .iter()
                    .filter(|(_, w)| !w.is_empty())
                    .map(|(id, w)| (*id, w.clone()))
                    .collect();
                canon.sort_unstable_by_key(|(state_id, _)| *state_id);
                if canon.is_empty() {
                    continue;
                }
                closure_cache.insert(pre_closure_key.clone(), canon.clone());
                canon
            };
            if profile_enabled {
                prof_labels_processed += 1;
                prof_eps_closure_calls += 1;
                prof_eps_closure_ns += t_eps.elapsed().as_nanos() as u64;
            }

            let next_support: Vec<u32> = canon.iter().map(|(sid, _)| *sid).collect();
            label_results.push((label, edge_weight, canon, next_support));
        }

        for (label, edge_weight, canon, next_support) in label_results {
            key_buf.clear();
            key_buf.extend(canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(key_buf.clone(), new_state);
                worklist.push_back(canon);
                supports.push(next_support);
                new_state
            };
            dwa.add_transition(from_state, label, to_state, edge_weight);
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
                    if let Some(state_final) = nwa.states[*nwa_state_id as usize].final_weight.as_ref() {
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
        eprintln!(
            "[glrmask/profile][determinize_supports] iterations={} total_subset_entries={} avg_subset_size={:.1} max_subset_size={} intersection_calls={} eps_closure_calls={} labels_processed={} raw_target_entries={} deferred_finals={} parallel_final_weight_ms={} subset_key_ms={:.1} collect_finals_ms={:.1} intersection_ms={:.1} eps_closure_ms={:.1} target_build_ms={:.1}",
            prof_iterations, prof_total_subset_entries, avg_subset, prof_max_subset_size,
            prof_intersection_calls, prof_eps_closure_calls, prof_labels_processed,
            prof_total_raw_target_entries,
            deferred_final_entries.len(),
            parallel_fw_ms,
            prof_subset_key_ns as f64 / 1_000_000.0,
            prof_final_weight_ns as f64 / 1_000_000.0,
            prof_intersection_ns as f64 / 1_000_000.0,
            prof_eps_closure_ns as f64 / 1_000_000.0,
            prof_target_build_ns as f64 / 1_000_000.0,
        );
    }

    DeterminizedDwaWithSupports { dwa, supports }
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

            let state = &dwa.states[state_id];

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

            let state = &mut dwa.states[state_id];
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

        for state_id in 0..dwa.states.len() {
            let Some((default_target, default_weight)) =
                dwa.states[state_id].transitions.get(&DEFAULT_LABEL).cloned()
            else {
                continue;
            };

            let target_final = dwa.states[default_target as usize].final_weight.clone();
            let Some(target_final) = target_final else { continue };
            let lifted = default_weight.intersection(&target_final);
            if lifted.is_empty() {
                continue;
            }

            if union_final_weight(&mut dwa.states[state_id].final_weight, lifted.clone()) {
                changed = true;
            }

            let state = &mut dwa.states[state_id];
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

        for state_id in 0..dwa.states.len() {
            let Some(&(default_target, ref default_weight)) =
                dwa.states[state_id].transitions.get(&DEFAULT_LABEL)
            else {
                continue;
            };
            let default_target = default_target;
            let default_weight = default_weight.clone();

            let state = &mut dwa.states[state_id];
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
    for state_id in 0..dwa.states.len() {
        let Some(final_weight) = dwa.states[state_id].final_weight.clone() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        let state = &mut dwa.states[state_id];
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
    nwa.states = vec![crate::automata::weighted::nwa::NWAState::default(); dwa.states.len()];
    nwa.start_states = vec![dwa.start_state];

    for (state_id, state) in dwa.states.iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.states[state_id].final_weight = Some(final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.states[state_id]
                .transitions
                .entry(label)
                .or_default()
                .push((*target, weight.clone()));
        }
    }

    nwa
}

fn optimize_parser_default_transitions(
    nwa: &mut NWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) -> bool {
    fn subtract_weight_from_outgoing(
        state: &mut crate::automata::weighted::nwa::NWAState,
        weight_to_subtract: &Weight,
    ) -> bool {
        let mut changed = false;
        for (_, edge_weight) in &mut state.epsilons {
            let new_weight = edge_weight.difference(weight_to_subtract);
            if new_weight != *edge_weight {
                *edge_weight = new_weight;
                changed = true;
            }
        }
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets.iter_mut() {
                let new_weight = edge_weight.difference(weight_to_subtract);
                if new_weight != *edge_weight {
                    *edge_weight = new_weight;
                    changed = true;
                }
            }
            targets.retain(|(_, edge_weight)| !edge_weight.is_empty());
        }
        state.epsilons.retain(|(_, edge_weight)| !edge_weight.is_empty());
        state.transitions.retain(|_, targets| !targets.is_empty());
        changed
    }

    let mut any_changed = false;

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

            let mut actual_positive = BitSet::new(num_parser_states as usize);
            for &label in nwa.states[state_id].transitions.keys() {
                if let Some(parser_state_id) = parser_state_label(label, num_parser_states) {
                    actual_positive.set(parser_state_id as usize);
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
                    for parser_state_id in 0..num_parser_states as i32 {
                        let Some(targets) = nwa.states[state_id].transitions.get(&parser_state_id) else {
                            valid = false;
                            break;
                        };
                        if targets.len() != 1 {
                            valid = false;
                            break;
                        }

                        let (target, weight) = &targets[0];
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
                    for parser_state_id in ids.iter_ones().map(|state_id| state_id as i32) {
                        let Some(targets) = nwa.states[state_id].transitions.get(&parser_state_id) else {
                            valid = false;
                            break;
                        };
                        if targets.len() != 1 {
                            valid = false;
                            break;
                        }

                        let (target, weight) = &targets[0];
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

            let Some(target) = shared_target else {
                continue;
            };
            let Some(default_weight) = default_weight else {
                continue;
            };
            if !valid || default_weight.is_empty() {
                continue;
            }

            if add_or_union_transition(&mut nwa.states[state_id], DEFAULT_LABEL, target, default_weight) {
                changed = true;
            }
        }

        for state_id in 0..nwa.states.len() {
            let default_targets = nwa.states[state_id]
                .transitions
                .get(&DEFAULT_LABEL)
                .cloned()
                .unwrap_or_default();
            if default_targets.is_empty() {
                continue;
            }

            let mut lifted_final = Weight::empty();
            for (_index, (target, weight)) in default_targets.iter().enumerate() {
                let Some(target_final) = nwa.states[*target as usize].final_weight.as_ref() else {
                    continue;
                };
                let lifted = weight.intersection(target_final);
                if lifted.is_empty() {
                    continue;
                }
                lifted_final = lifted_final.union(&lifted);
            }

            if union_final_weight(&mut nwa.states[state_id].final_weight, lifted_final.clone()) {
                changed = true;
            }

            if subtract_weight_from_outgoing(&mut nwa.states[state_id], &lifted_final) {
                changed = true;
            }
        }

        for state in &mut nwa.states {
            let mut default_by_target: HashMap<u32, Weight> = HashMap::new();
            if let Some(default_targets) = state.transitions.get(&DEFAULT_LABEL) {
                for (target, weight) in default_targets {
                    default_by_target
                        .entry(*target)
                        .and_modify(|existing| *existing = existing.union(weight))
                        .or_insert_with(|| weight.clone());
                }
            }
            if default_by_target.is_empty() {
                continue;
            }

            for (&label, targets) in state.transitions.iter_mut() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                for (target, weight) in targets.iter_mut() {
                    let Some(default_weight) = default_by_target.get(target) else {
                        continue;
                    };
                    let new_weight = weight.difference(default_weight);
                    if new_weight != *weight {
                        *weight = new_weight;
                        changed = true;
                    }
                }
                targets.retain(|(_, weight)| !weight.is_empty());
            }
            state.transitions.retain(|_, targets| !targets.is_empty());
        }

        if !changed {
            break;
        }
        any_changed = true;
    }

    any_changed
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
    let mut concatenated_branches = HashMap::new();
    let compose_started_at = Instant::now();
    let parser_body = compose_state(
        terminal_dwa.start_state,
        &states,
        &mut arena,
        &mut body_memo,
        &mut concatenated_branches,
    )?;
    profile.compose_state_ms = elapsed_ms(compose_started_at);

    arena.start_states = parser_body.start_states.clone();

    Some((arena, profile))
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
    );
    parser_dwa.clip_weights(id_map.max_internal_token_id());
    parser_dwa
}

pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal_dwa: &DWA,
    templates: Templates,
) -> DWA {
    let total_started_at = Instant::now();
    let Some((mut parser_nwa, mut profile)) =
        build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates)
    else {
        return DWA::new(0, 0);
    };

    if parser_dwa_profile_enabled() {
        let nwa_transitions: usize = parser_nwa.states.iter()
            .map(|s| s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] phase=pre_resolve_negatives nwa_states={} nwa_transitions={} terminal_dwa_states={}",
            parser_nwa.states.len(), nwa_transitions, terminal_dwa.states.len(),
        );
    }

    let resolve_negatives_started_at = Instant::now();
    resolve_negative_codes_in_nwa(&mut parser_nwa);
    profile.resolve_negatives_ms = elapsed_ms(resolve_negatives_started_at);

    if parser_dwa_profile_enabled() {
        let nwa_transitions: usize = parser_nwa.states.iter()
            .map(|s| s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] phase=post_resolve_negatives nwa_states={} nwa_transitions={} terminal_dwa_states={}",
            parser_nwa.states.len(), nwa_transitions, terminal_dwa.states.len(),
        );
    }

    let determinize_supports_started_at = Instant::now();
    let determinized = determinize_with_supports(&parser_nwa);
    profile.determinize_supports_ms = elapsed_ms(determinize_supports_started_at);
    let mut parser_dwa_pre_minimize = determinized.dwa;

    if parser_dwa_profile_enabled() {
        let dwa_transitions: usize = parser_dwa_pre_minimize.states.iter()
            .map(|s| s.transitions.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] dwa_states={} dwa_transitions={} minimized_later",
            parser_dwa_pre_minimize.states.len(), dwa_transitions,
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
    optimize_parser_dwa_defaults(
        &mut parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );
    profile.optimize_defaults_ms = elapsed_ms(optimize_defaults_started_at);

    let subtract_final_started_at = Instant::now();
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    profile.subtract_final_ms = elapsed_ms(subtract_final_started_at);

    profile.determinize_after_defaults_ms = 0.0;

    let minimize_started_at = Instant::now();
    let minimized = minimize_from_env(
        &parser_dwa_pre_minimize,
        "GLRMASK_MINIMIZE_PARSER_DWA",
        minimize_fast,
    );
    profile.minimize_ms = elapsed_ms(minimize_started_at);
    profile.total_ms = elapsed_ms(total_started_at);

    if parser_dwa_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_dwa] build_state_summaries_ms={:.3} compose_state_ms={:.3} resolve_negatives_ms={:.3} viable_suffix_ms={:.3} determinize_supports_ms={:.3} optimize_defaults_ms={:.3} subtract_final_ms={:.3} determinize_after_defaults_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            profile.build_state_summaries_ms,
            profile.compose_state_ms,
            profile.resolve_negatives_ms,
            profile.viable_suffix_ms,
            profile.determinize_supports_ms,
            profile.optimize_defaults_ms,
            profile.subtract_final_ms,
            profile.determinize_after_defaults_ms,
            profile.minimize_ms,
            profile.total_ms,
        );
    }

    minimized
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
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

    #[test]
    fn test_optimize_parser_defaults_synthesizes_and_subtracts_same_destination_weight() {
        let mut dwa = DWA::new(0, 0);
        let accept = dwa.add_state();
        dwa.add_transition(0, 0, accept, token_weight(&[1, 2]));
        dwa.add_transition(0, 1, accept, token_weight(&[2, 3]));

        let mut nwa = dwa_to_nwa(&dwa);
        let mut start_ids = BitSet::new(2);
        start_ids.set(0);
        start_ids.set(1);
        let possible_by_state = vec![
            PossibleOutgoingIds::Some(start_ids),
            PossibleOutgoingIds::Empty,
        ];

        assert!(optimize_parser_default_transitions(&mut nwa, &possible_by_state, 2));

        let defaults = nwa.states[0].transitions.get(&DEFAULT_LABEL).expect("default edge");
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].0, accept);
        assert_eq!(defaults[0].1, token_weight(&[2]));

        let explicit_zero = &nwa.states[0].transitions[&0][0];
        let explicit_one = &nwa.states[0].transitions[&1][0];
        assert_eq!(explicit_zero.1, token_weight(&[1]));
        assert_eq!(explicit_one.1, token_weight(&[3]));
    }

    #[test]
    fn test_optimize_parser_defaults_lifts_default_weight_into_source_final() {
        let mut nwa = NWA::new(0, 0);
        nwa.states = vec![crate::automata::weighted::nwa::NWAState::default(); 2];
        nwa.start_states = vec![0];
        nwa.states[0]
            .transitions
            .entry(DEFAULT_LABEL)
            .or_default()
            .push((1, token_weight(&[1, 2])));
        nwa.states[1].final_weight = Some(token_weight(&[2, 3]));

        let possible_by_state = vec![PossibleOutgoingIds::Empty, PossibleOutgoingIds::Empty];

        assert!(optimize_parser_default_transitions(&mut nwa, &possible_by_state, 4));

        assert_eq!(nwa.states[0].final_weight, Some(token_weight(&[2])));
        let defaults = nwa.states[0].transitions.get(&DEFAULT_LABEL).expect("default edge");
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].1, token_weight(&[1]));
    }
}
