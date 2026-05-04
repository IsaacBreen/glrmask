use std::collections::{hash_map::Entry, BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_fast;
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::GLRTable;
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::Templates;
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

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
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

    let built_bundles: Vec<Arc<NWA>> = {
        use rayon::prelude::*;
        unique_bundles
            .par_iter()
            .map(|bundle| Arc::new(templates.build_bundle(bundle)))
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

    // Deferred final weight computation: store subset entries for each DWA state
    // and compute final weights in parallel after the main loop.
    let mut deferred_final_entries: Vec<(u32, Vec<(u32, Weight)>)> = Vec::new();

    while let Some(subset_entries) = worklist.pop_front() {
        let from_state = subset_map[&subset_key(&subset_entries)];

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let has_finals: Vec<(u32, Weight)> = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states()[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
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

        let mut pre_closure_key: Vec<(u32, usize)> = Vec::new();

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));

            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                if nwa.states()[*only_state as usize].epsilons.is_empty() {
                    key_buf.clear();
                    key_buf.push((*only_state, only_weight.ptr_key()));
                    let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                        existing
                    } else {
                        let new_state = dwa.add_state();
                        subset_map.insert(key_buf.clone(), new_state);
                        worklist.push_back(vec![(*only_state, only_weight.clone())]);
                        supports.push(vec![*only_state]);
                        new_state
                    };
                    dwa.add_transition(from_state, label, to_state, only_weight.clone());
                    return;
                }
            }

            pre_closure_key.clear();
            pre_closure_key.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));

            let cached = match closure_cache.entry(pre_closure_key.clone()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
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
                    entry.insert(CachedClosure { canon, edge_weight })
                }
            };

            key_buf.clear();
            key_buf.extend(cached.canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(key_buf.clone(), new_state);
                worklist.push_back(cached.canon.clone());
                supports.push(cached.canon.iter().map(|(sid, _)| *sid).collect());
                new_state
            };
            dwa.add_transition(from_state, label, to_state, cached.edge_weight.clone());
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

    // Compute final weights in parallel using rayon.
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

fn build_parser_nwa_from_terminal_dwa(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: Templates,
) -> Option<NWA> {
    let states = build_state_summaries(terminal_dwa, grammar, &templates);

    let mut arena = NWA::new(0, 0);
    let mut body_memo = vec![None; states.len()];
    let mut concatenated_branches: ConcatMemo = FxHashMap::default();
    let parser_body = compose_state(
        terminal_dwa.start_state(),
        &states,
        &mut arena,
        &mut body_memo,
        &mut concatenated_branches,
    )?;

    arena.set_start_states(parser_body.start_states.clone());

    Some(arena)
}

pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal_dwa: &DWA,
    templates: Templates,
    _vocab: &Vocab,
    _id_map: &InternalIdMap,
) -> DWA {
    let Some(mut parser_nwa) = build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates) else {
        return DWA::new(0, 0);
    };

    resolve_negative_codes_in_nwa(&mut parser_nwa);

    let determinized = determinize_with_supports(&parser_nwa, Some(table.num_states));
    let mut parser_dwa_pre_minimize = determinized.dwa;

    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        table.num_states,
    );
    optimize_parser_dwa_defaults(
        &mut parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    parser_dwa_pre_minimize = determinize_parser_dwa_with_fallbacks(
        &parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );

    minimize_fast(&parser_dwa_pre_minimize)
}

