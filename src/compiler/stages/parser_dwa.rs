use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

#[cfg(test)]
use crate::Vocab;
#[cfg(test)]
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_fast;
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::TerminalID;
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

fn leaf_continuation_final_weight(states: &[StateSummary], state_id: u32) -> Option<Weight> {
    let state = states.get(state_id as usize)?;
    if !state.branches.is_empty() {
        return None;
    }
    state.final_weight.clone().filter(|weight| !weight.is_empty())
}

fn build_leaf_specialized_bundle(bundle: &NWA, leaf_final_weight: &Weight) -> Option<NWA> {
    let mut specialized = bundle.clone();
    let mut has_live_final = false;

    for state in &mut specialized.states {
        let Some(final_weight) = state.final_weight.as_ref() else {
            continue;
        };
        let combined = final_weight.intersection(leaf_final_weight);
        if combined.is_empty() {
            state.final_weight = None;
        } else {
            state.final_weight = Some(combined);
            has_live_final = true;
        }
    }

    if !has_live_final {
        return None;
    }

    resolve_negative_codes_in_nwa(&mut specialized);
    Some(specialized)
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

    let leaf_targets: Vec<bool> = pending_branches_by_state
        .iter()
        .enumerate()
        .map(|(state_id, branches)| {
            branches.is_empty()
                && terminal_dwa
                    .states
                    .get(state_id)
                    .and_then(|state| state.final_weight.as_ref())
                    .is_some_and(|weight| !weight.is_empty())
        })
        .collect();
    let branches_to_leaf_targets = pending_branches_by_state
        .iter()
        .flat_map(|branches| branches.iter())
        .filter(|pending| leaf_targets.get(pending.target as usize).copied().unwrap_or(false))
        .count();

    if parser_dwa_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_dwa_bundles] terminal_dwa_states={} unique_bundles={} total_branches={}",
            terminal_dwa.states.len(),
            unique_bundles.len(),
            pending_branches_by_state.iter().map(|b| b.len()).sum::<usize>(),
        );
        eprintln!(
            "[glrmask/profile][parser_dwa_leaf_targets] leaf_states={} branches_to_leaf_targets={} states_with_branches={}",
            leaf_targets.iter().filter(|&&is_leaf| is_leaf).count(),
            branches_to_leaf_targets,
            pending_branches_by_state.iter().filter(|branches| !branches.is_empty()).count(),
        );
        for (i, bundle) in unique_bundles.iter().enumerate() {
            let terminals: Vec<_> = bundle.keys().collect();
            if terminals.len() > 5 || bundle.values().any(|w| !w.is_empty()) {
                eprintln!(
                    "[glrmask/profile][parser_dwa_bundles] bundle={} num_terminals={}",
                    i, terminals.len(),
                );
            }
        }
    }

    let built_bundles: Vec<Arc<NWA>> = {
        use rayon::prelude::*;
        unique_bundles
            .par_iter()
            .map(|bundle| {
                let nwa = Arc::new(templates.build_bundle(bundle));
                if parser_dwa_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][parser_dwa_bundle_built] bundle_terminals={} nwa_states={}",
                        bundle.len(), nwa.states.len(),
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
        let concat_key = (branch.bundle_id, branch.target);
        let branch_with_continuation = if let Some(cached) = concatenated_branches.get(&concat_key) {
            cached.clone()
        } else if let Some(leaf_final_weight) = leaf_continuation_final_weight(states, branch.target) {
            let built = build_leaf_specialized_bundle(branch.bundle.as_ref(), &leaf_final_weight)
                .map(|specialized| arena.append_with_body(&specialized));
            concatenated_branches.insert(concat_key, built.clone());
            built
        } else {
            let Some(continuation) = compose_state(
                branch.target,
                states,
                arena,
                body_memo,
                concatenated_branches,
            ) else {
                continue;
            };

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

fn build_possible_outgoing_ids_by_state(
    parser_nwa: &NWA,
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<BitSet> {
    let num_parser_states = num_parser_states as usize;
    let all_parser_states = BitSet::all(num_parser_states);
    let state_outgoing_ids: Vec<BitSet> = parser_nwa
        .states
        .iter()
        .map(|state| {
            let mut ids = BitSet::new(num_parser_states);
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    return all_parser_states.clone();
                }
                if let Some(parser_state_id) = parser_state_label(label, num_parser_states as u32) {
                    ids.set(parser_state_id as usize);
                }
            }
            ids
        })
        .collect();

    state_supports
        .iter()
        .map(|support| {
            let mut ids = BitSet::new(num_parser_states);
            for &state_id in support {
                let Some(state_ids) = state_outgoing_ids.get(state_id as usize) else {
                    continue;
                };
                ids.union_with(state_ids);
                if ids == all_parser_states {
                    break;
                }
            }
            ids
        })
        .collect()
}

fn determinize_with_supports(nwa: &NWA) -> DeterminizedDwaWithSupports {
    fn canonicalize(subset: &FxHashMap<u32, Weight>) -> Vec<(u32, Weight)> {
        let mut entries: Vec<_> = subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect();
        entries.sort_by_key(|(state_id, _)| *state_id);
        entries
    }

    /// Cheap key for subset_map: uses Arc pointer address instead of full weight hashing.
    /// Safe because weights are interned, so pointer equality ↔ content equality.
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    fn epsilon_closure(nwa: &NWA, seed: FxHashMap<u32, Weight>) -> FxHashMap<u32, Weight> {
        // Fast path: single-state seed with no epsilon transitions
        if seed.len() == 1 {
            let (&state_id, _) = seed.iter().next().unwrap();
            if let Some(state) = nwa.states.get(state_id as usize) {
                if state.epsilons.is_empty() {
                    return seed;
                }
            }
        }

        let mut closure = seed;
        let mut queue: VecDeque<u32> = closure.keys().copied().collect();

        while let Some(state_id) = queue.pop_front() {
            let Some(current_weight) = closure.get(&state_id).cloned() else {
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
                let existing = closure.get(target).cloned().unwrap_or_else(Weight::empty);
                if !contribution.is_subset(&existing) {
                    closure.insert(*target, existing.union(&contribution));
                    queue.push_back(*target);
                }
            }
        }

        closure
    }

    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![Vec::new()];

    let mut start_subset = FxHashMap::default();
    for &state_id in &nwa.start_states {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    let start_subset = epsilon_closure(nwa, start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    let start_entries = canonicalize(&start_subset);
    supports[0] = start_entries.iter().map(|(state_id, _)| *state_id).collect();

    // Use pointer-based keys for O(1) hashing instead of iterating Weight ranges
    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&start_entries), dwa.start_state);
    worklist.push_back(start_entries);

    let mut raw_targets: FxHashMap<i32, FxHashMap<u32, Vec<Weight>>> = FxHashMap::default();

    while let Some(subset_entries) = worklist.pop_front() {
        let from_state = subset_map[&subset_key(&subset_entries)];

        let mut final_weight = Weight::empty();
        for (nwa_state_id, path_weight) in &subset_entries {
            if let Some(state_final) = nwa.states[*nwa_state_id as usize].final_weight.as_ref() {
                final_weight = final_weight.union(&path_weight.intersection(state_final));
            }
        }
        if !final_weight.is_empty() {
            dwa.set_final_weight(from_state, final_weight);
        }

        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    let next_weight = path_weight.intersection(transition_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    raw_targets.entry(label).or_default().entry(*target).or_default().push(next_weight);
                }
            }
        }

        for (label, target_contributions) in raw_targets.drain() {
            if target_contributions.is_empty() {
                continue;
            }

            let mut target_subset: FxHashMap<u32, Weight> = FxHashMap::default();
            for (dst, weights) in target_contributions {
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

            let expanded = epsilon_closure(nwa, target_subset);
            if expanded.is_empty() {
                continue;
            }

            let edge_complement = edge_weight.complement();
            let normalized: FxHashMap<u32, Weight> = if edge_complement.is_empty() {
                expanded
            } else {
                expanded
                    .into_iter()
                    .filter_map(|(state_id, weight)| {
                        let normalized_weight = weight.union(&edge_complement);
                        (!normalized_weight.is_empty()).then_some((state_id, normalized_weight))
                    })
                    .collect()
            };

            let next_entries = canonicalize(&normalized);
            if next_entries.is_empty() {
                continue;
            }

            let next_support: Vec<u32> = next_entries.iter().map(|(state_id, _)| *state_id).collect();

            let next_key = subset_key(&next_entries);
            let to_state = if let Some(existing) = subset_map.get(&next_key).copied() {
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(next_key, new_state);
                worklist.push_back(next_entries);
                supports.push(next_support);
                new_state
            };

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

/// Apply default-transition optimization directly on a DWA, avoiding the
/// DWA→NWA→optimize→determinize round-trip. This replaces repeated per-parser-state
/// transitions (all going to the same target) with a single DEFAULT_LABEL transition,
/// lifts final weights from the default target, and subtracts covered weights.
fn optimize_parser_dwa_defaults(
    dwa: &mut DWA,
    possible_by_state: &[BitSet],
    num_parser_states: u32,
) {
    loop {
        let mut changed = false;

        // Phase 1: Identify states where all possible parser-state labels go to the same
        // target and create a DEFAULT_LABEL entry with the intersection weight.
        for (state_id, possible_ids) in possible_by_state.iter().enumerate() {
            if possible_ids.is_empty() || possible_ids.count_ones() < 2 {
                continue;
            }

            let state = &dwa.states[state_id];

            // Check that every possible parser_state label is present
            let mut actual_positive = BitSet::new(num_parser_states as usize);
            for &label in state.transitions.keys() {
                if let Some(ps) = parser_state_label(label, num_parser_states) {
                    actual_positive.set(ps as usize);
                }
            }
            if actual_positive != *possible_ids {
                continue;
            }

            // Check all parser-state transitions share the same target
            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            for ps in possible_ids.iter_ones() {
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

            let Some(target) = shared_target else { continue };
            let Some(default_weight) = default_weight else { continue };
            if !valid || default_weight.is_empty() {
                continue;
            }

            // Add or union DEFAULT_LABEL transition
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
                    // Different target — skip (cannot merge in DWA)
                }
                std::collections::btree_map::Entry::Vacant(vac) => {
                    vac.insert((target, default_weight));
                    changed = true;
                }
            }
        }

        // Phase 2: Lift final weights from default targets to source states.
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

            // Union lifted final weight into source state
            if union_final_weight(&mut dwa.states[state_id].final_weight, lifted.clone()) {
                changed = true;
            }

            // Subtract lifted weight from all outgoing transitions of this state
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

        // Phase 3: For each state with DEFAULT_LABEL, subtract default weight from
        // explicit transitions that share the same target.
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

/// Subtract final weights from all outgoing transitions (DWA version).
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
    possible_by_state: &[BitSet],
    num_parser_states: u32,
) -> bool {
    fn subtract_weight_from_outgoing(
        state: &mut crate::automata::weighted::nwa::NWAState,
        weight_to_subtract: &Weight,
    ) -> bool {
        if weight_to_subtract.is_empty() {
            return false;
        }

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
            if possible_ids.is_empty() {
                continue;
            }
            if possible_ids.count_ones() < 2 {
                continue;
            }

            let mut actual_positive = BitSet::new(num_parser_states as usize);
            for &label in nwa.states[state_id].transitions.keys() {
                if let Some(parser_state_id) = parser_state_label(label, num_parser_states) {
                    actual_positive.set(parser_state_id as usize);
                }
            }
            if actual_positive != *possible_ids {
                continue;
            }

            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            for parser_state_id in possible_ids.iter_ones().map(|state_id| state_id as i32) {
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

    let resolve_negatives_started_at = Instant::now();
    resolve_negative_codes_in_nwa(&mut parser_nwa);
    profile.resolve_negatives_ms = elapsed_ms(resolve_negatives_started_at);

    if parser_dwa_profile_enabled() {
        let nwa_transitions: usize = parser_nwa.states.iter()
            .map(|s| s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len())
            .sum();
        eprintln!(
            "[glrmask/profile][parser_dwa_scale] nwa_states={} nwa_transitions={} terminal_dwa_states={}",
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

    // determinize_after_defaults is no longer needed — optimization is done directly on the DWA
    profile.determinize_after_defaults_ms = 0.0;

    let minimize_started_at = Instant::now();
    let minimized = minimize_fast(&parser_dwa_pre_minimize);
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
    use crate::compiler::grammar::model::GrammarDef;
    use crate::compiler::grammar::model::tests::*;
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
        let possible_by_state = vec![start_ids, BitSet::new(2)];

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

        let possible_by_state = vec![BitSet::new(4), BitSet::new(4)];

        assert!(optimize_parser_default_transitions(&mut nwa, &possible_by_state, 4));

        assert_eq!(nwa.states[0].final_weight, Some(token_weight(&[2])));
        let defaults = nwa.states[0].transitions.get(&DEFAULT_LABEL).expect("default edge");
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].1, token_weight(&[1]));
    }
}
