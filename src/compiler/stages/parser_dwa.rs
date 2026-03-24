#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::sync::Arc;

#[cfg(test)]
use crate::Vocab;
#[cfg(test)]
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::TerminalID;
#[cfg(test)]
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
#[cfg(test)]
use crate::compiler::stages::terminal_dwa::build_terminal_dwa;
use crate::compiler::stages::templates::Templates;
#[cfg(test)]
use crate::compiler::stages::templates::characterize::characterize_terminals;
use crate::ds::weight::Weight;

type Bundle = BTreeMap<TerminalID, Weight>;

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

#[derive(Debug, Clone, Default)]
struct ViableSuffixRecognizer {
    subset_to_state: HashMap<Vec<u32>, u32>,
    possible_outgoing_ids: Vec<Vec<u32>>,
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
) -> BTreeMap<u32, Bundle> {
    let Some(state) = terminal_dwa.states.get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut groups = BTreeMap::<u32, Bundle>::new();
    for (&label, (target, weight)) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        groups
            .entry(*target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }

    groups
}

fn build_state_summaries(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> Vec<StateSummary> {
    let mut state_groups: Vec<Vec<(u32, Vec<(TerminalID, Weight)>, Bundle)>> = Vec::with_capacity(terminal_dwa.states.len());
    let mut unique_keys: HashMap<Vec<(TerminalID, Weight)>, usize> = HashMap::new();
    let mut unique_bundles_ordered: Vec<(usize, Vec<(TerminalID, Weight)>, Bundle)> = Vec::new();

    for (state_id, _state) in terminal_dwa.states.iter().enumerate() {
        let groups = group_terminal_edges_by_target(terminal_dwa, grammar, state_id as u32);
        let mut state_entries = Vec::with_capacity(groups.len());
        for (target, bundle) in groups {
            let bundle_key: Vec<(TerminalID, Weight)> = bundle
                .iter()
                .map(|(&terminal, weight)| (terminal, weight.clone()))
                .collect();
            if unique_keys.contains_key(&bundle_key) {
                state_entries.push((target, bundle_key, bundle));
            } else {
                let id = unique_bundles_ordered.len();
                unique_keys.insert(bundle_key.clone(), id);
                unique_bundles_ordered.push((id, bundle_key.clone(), bundle.clone()));
                state_entries.push((target, bundle_key, bundle));
            }
        }
        state_groups.push(state_entries);
    }

    let built_bundles: Vec<Arc<NWA>> = {
        use rayon::prelude::*;
        unique_bundles_ordered
            .par_iter()
            .map(|(_id, _key, bundle)| Arc::new(templates.build_bundle(bundle)))
            .collect()
    };

    terminal_dwa
        .states
        .iter()
        .enumerate()
        .map(|(state_id, state)| {
            let branches = state_groups[state_id]
                .iter()
                .map(|(target, bundle_key, _bundle)| {
                    let bundle_id = unique_keys[bundle_key];
                    let built_bundle = Arc::clone(&built_bundles[bundle_id]);
                    Branch {
                        target: *target,
                        bundle_id,
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
    memo: &mut Vec<Option<Option<crate::automata::weighted::nwa::NwaBody>>>,
    concat_memo: &mut HashMap<(usize, u32), Option<crate::automata::weighted::nwa::NwaBody>>,
) -> Option<crate::automata::weighted::nwa::NwaBody> {
    if let Some(Some(cached)) = memo.get(state_id as usize) {
        return cached.clone();
    }

    let Some(state) = states.get(state_id as usize) else {
        return None;
    };

    let mut composed = state
        .final_weight
        .as_ref()
        .and_then(accepting_nwa)
        .map(|accepting| arena.append_with_body(&accepting));

    for branch in &state.branches {
        let Some(continuation) = compose_state(
            branch.target,
            states,
            arena,
            memo,
            concat_memo,
        ) else {
            continue;
        };

        let concat_key = (branch.bundle_id, branch.target);
        let branch_with_continuation = if let Some(cached) = concat_memo.get(&concat_key) {
            cached.clone()
        } else {
            let built = Some(arena.concatenate_in_place(branch.bundle.as_ref(), &continuation));
            concat_memo.insert(concat_key, built.clone());
            built
        };
        let Some(branch_with_continuation) = branch_with_continuation else {
            continue;
        };

        composed = Some(match composed {
            Some(existing) => crate::automata::weighted::nwa::NwaBody::union(&existing, &branch_with_continuation),
            None => branch_with_continuation,
        });
    }

    if let Some(entry) = memo.get_mut(state_id as usize) {
        *entry = Some(composed.clone());
    }
    composed
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

fn nwa_to_unweighted(nwa: &NWA) -> UnweightedNfa {
    let mut out = UnweightedNfa::new_empty();
    out.states = vec![crate::automata::unweighted_u32::nfa::NFAState::default(); nwa.states.len()];
    out.start_states = nwa.start_states.clone();

    for (state_id, state) in nwa.states.iter().enumerate() {
        if state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()) {
            out.states[state_id].is_accepting = true;
        }

        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                if !weight.is_empty() {
                    out.states[state_id].transitions.entry(label).or_default().push(*target);
                }
            }
        }

        for (target, weight) in &state.epsilons {
            if !weight.is_empty() {
                out.states[state_id].epsilons.push(*target);
            }
        }
    }

    out
}

fn determinize_unweighted_with_subset_map(nfa: &UnweightedNfa) -> (UnweightedDfa, HashMap<Vec<u32>, u32>) {
    fn epsilon_closure(nfa: &UnweightedNfa, seeds: &[u32]) -> BTreeSet<u32> {
        let mut closed = BTreeSet::new();
        let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
        while let Some(state_id) = queue.pop_front() {
            if closed.insert(state_id) {
                for &target in &nfa.states[state_id as usize].epsilons {
                    if !closed.contains(&target) {
                        queue.push_back(target);
                    }
                }
            }
        }
        closed
    }

    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return (UnweightedDfa::new(), HashMap::new());
    }

    let mut dfa = UnweightedDfa::new();
    let mut subset_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    let start_closure = epsilon_closure(nfa, &nfa.start_states);
    let start_key: Vec<u32> = start_closure.iter().copied().collect();
    subset_map.insert(start_key.clone(), dfa.start_state);
    worklist.push_back(start_key);

    while let Some(subset_key) = worklist.pop_front() {
        let dfa_state = subset_map[&subset_key];
        if subset_key.iter().any(|&state_id| nfa.states[state_id as usize].is_accepting) {
            dfa.set_accepting(dfa_state, true);
        }

        let mut label_targets: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
        for &state_id in &subset_key {
            for (&label, targets) in &nfa.states[state_id as usize].transitions {
                let entry = label_targets.entry(label).or_default();
                for &target in targets {
                    entry.insert(target);
                }
            }
        }

        for (label, raw_targets) in label_targets {
            let closure = epsilon_closure(nfa, &raw_targets.iter().copied().collect::<Vec<_>>());
            let next_key: Vec<u32> = closure.iter().copied().collect();
            if next_key.is_empty() {
                continue;
            }

            let next_state = if let Some(&existing) = subset_map.get(&next_key) {
                existing
            } else {
                let new_state = dfa.add_state();
                subset_map.insert(next_key.clone(), new_state);
                worklist.push_back(next_key);
                new_state
            };
            dfa.add_transition(dfa_state, label, next_state);
        }
    }

    (dfa, subset_map)
}

fn build_viable_suffix_recognizer(nwa: &NWA, num_parser_states: u32) -> ViableSuffixRecognizer {
    let unweighted = nwa_to_unweighted(nwa);
    let (dfa, subset_to_state) = determinize_unweighted_with_subset_map(&unweighted);
    let possible_outgoing_ids = dfa
        .states
        .iter()
        .map(|state| {
            let mut ids = BTreeSet::new();
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    ids.extend(0..num_parser_states);
                } else if let Some(parser_state_id) = parser_state_label(label, num_parser_states) {
                    ids.insert(parser_state_id);
                }
            }
            ids.into_iter().collect()
        })
        .collect();

    ViableSuffixRecognizer {
        subset_to_state,
        possible_outgoing_ids,
    }
}

fn determinize_with_supports(nwa: &NWA) -> DeterminizedDwaWithSupports {
    fn canonicalize(subset: &HashMap<u32, Weight>) -> Vec<(u32, Weight)> {
        let mut entries: Vec<_> = subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect();
        entries.sort_by_key(|(state_id, _)| *state_id);
        entries
    }

    fn epsilon_closure(nwa: &NWA, seed: HashMap<u32, Weight>) -> HashMap<u32, Weight> {
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

    let mut start_subset = HashMap::new();
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

    let mut subset_map: HashMap<Vec<(u32, Weight)>, u32> = HashMap::new();
    let mut worklist: VecDeque<(Vec<(u32, Weight)>, Vec<(u32, Weight)>)> = VecDeque::new();
    let start_key = start_entries.clone();
    subset_map.insert(start_key.clone(), dwa.start_state);
    worklist.push_back((start_key, start_entries));

    while let Some((subset_key, subset_entries)) = worklist.pop_front() {
        let from_state = subset_map[&subset_key];

        let mut final_weight = Weight::empty();
        for (nwa_state_id, path_weight) in &subset_entries {
            if let Some(state_final) = nwa.states[*nwa_state_id as usize].final_weight.as_ref() {
                final_weight = final_weight.union(&path_weight.intersection(state_final));
            }
        }
        if !final_weight.is_empty() {
            dwa.set_final_weight(from_state, final_weight);
        }

        let mut raw_targets: HashMap<i32, HashMap<u32, Weight>> = HashMap::new();
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    let next_weight = path_weight.intersection(transition_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    let target_entry = raw_targets.entry(label).or_default();
                    target_entry
                        .entry(*target)
                        .and_modify(|existing| *existing = existing.union(&next_weight))
                        .or_insert(next_weight);
                }
            }
        }

        for (label, target_subset) in raw_targets {
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
            let normalized: HashMap<u32, Weight> = if edge_complement.is_empty() {
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

            let to_state = if let Some(existing) = subset_map.get(&next_entries).copied() {
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(next_entries.clone(), new_state);
                worklist.push_back((next_entries.clone(), next_entries));
                supports.push(next_support);
                new_state
            };

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
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
    state_supports: &[Vec<u32>],
    recognizer: &ViableSuffixRecognizer,
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

    let possible_by_state: Vec<Vec<u32>> = state_supports
        .iter()
        .map(|support| {
            recognizer
                .subset_to_state
                .get(support)
                .and_then(|state_id| recognizer.possible_outgoing_ids.get(*state_id as usize))
                .cloned()
                .unwrap_or_default()
        })
        .collect();

    let mut any_changed = false;

    loop {
        let mut changed = false;

        for (state_id, possible_ids) in possible_by_state.iter().enumerate() {
            if possible_ids.is_empty() {
                continue;
            }
            if possible_ids.len() < 2 {
                continue;
            }

            let possible_set: BTreeSet<u32> = possible_ids.iter().copied().collect();
            let actual_positive: BTreeSet<u32> = nwa.states[state_id]
                .transitions
                .iter()
                .filter_map(|(&label, _)| parser_state_label(label, num_parser_states))
                .collect();
            if actual_positive != possible_set {
                continue;
            }

            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            for parser_state_id in possible_ids {
                let Some(targets) = nwa.states[state_id].transitions.get(&(*parser_state_id as i32)) else {
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
        || build_terminal_dwa(grammar, tokenizer, vocab, id_map, ignore_terminal),
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
    let states = build_state_summaries(terminal_dwa, grammar, &templates);
    let mut arena = NWA::new(0, 0);
    let mut memo = vec![None; states.len()];
    let mut concat_memo = HashMap::new();
    let Some(parser_body) = compose_state(
        terminal_dwa.start_state,
        &states,
        &mut arena,
        &mut memo,
        &mut concat_memo,
    )
    else {
        return DWA::new(0, 0);
    };
    arena.start_states = parser_body.start_states.clone();
    let mut parser_nwa = arena;

    resolve_negative_codes_in_nwa(&mut parser_nwa);

    let viable_suffix_recognizer = build_viable_suffix_recognizer(&parser_nwa, table.num_states);

    let determinized = determinize_with_supports(&parser_nwa);
    let parser_dwa_pre_minimize = determinized.dwa;

    let mut optimized_parser_nwa = dwa_to_nwa(&parser_dwa_pre_minimize);
    let default_opt_enabled = std::env::var_os("GLRMASK_DISABLE_PARSER_DEFAULT_OPT").is_none();
    if default_opt_enabled {
        optimize_parser_default_transitions(
            &mut optimized_parser_nwa,
            &determinized.supports,
            &viable_suffix_recognizer,
            table.num_states,
        );
    }

    optimized_parser_nwa.subtract_final_weights_from_outgoing();

    let determinized_after_defaults = determinize(&optimized_parser_nwa)
        .expect("parser NWA determinization failed after default-transition optimization");
    minimize(&determinized_after_defaults)
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
        let id_map = InternalIdMap::build(&tok, &vocab, &std::collections::BTreeMap::new(), None);
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
        let supports = vec![vec![0], vec![1]];
        let recognizer = ViableSuffixRecognizer {
            subset_to_state: HashMap::from([(vec![0], 0), (vec![1], 1)]),
            possible_outgoing_ids: vec![vec![0, 1], vec![]],
        };

        assert!(optimize_parser_default_transitions(&mut nwa, &supports, &recognizer, 2));

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

        let supports = vec![vec![0], vec![1]];
        let recognizer = ViableSuffixRecognizer {
            subset_to_state: HashMap::from([(vec![0], 0), (vec![1], 1)]),
            possible_outgoing_ids: vec![vec![], vec![]],
        };

        assert!(optimize_parser_default_transitions(&mut nwa, &supports, &recognizer, 4));

        assert_eq!(nwa.states[0].final_weight, Some(token_weight(&[2])));
        let defaults = nwa.states[0].transitions.get(&DEFAULT_LABEL).expect("default edge");
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].1, token_weight(&[1]));
    }
}
