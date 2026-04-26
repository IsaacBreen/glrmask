//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::Instant;
use rustc_hash::FxHashMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as unweighted_minimize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::{minimize_fast, minimize_from_env};
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::Weight;

fn parser_dwa_bundle_determinize_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_PARSER_DWA_BUNDLE_DETERMINIZE").is_some()
}

fn empty_bundle_nwa() -> NWA {
    let mut nwa = NWA::new(0, 0);
    let start_state = nwa.add_state();
    nwa.start_states_mut().push(start_state);
    nwa
}

fn instantiate_weighted_nwa_from_skeleton(skeleton: &NWA, weight: &Weight) -> NWA {
    let mut bundle = skeleton.clone();
    for state in  bundle.states_mut() {
        if state.final_weight.is_some() {
            state.final_weight = Some(weight.clone());
        }
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets {
                *edge_weight = weight.clone();
            }
        }
        for (_, epsilon_weight) in &mut state.epsilons {
            *epsilon_weight = weight.clone();
        }
    }

    bundle
}

fn compute_effective_group_weights(alive_groups: &[usize], normalized_weights: &[Weight]) -> Vec<Weight> {
    let alive_union = Weight::union_all(alive_groups.iter().map(|&index| &normalized_weights[index]));
    let complement_alive = alive_union.complement();
    let mut is_alive = vec![false; normalized_weights.len()];
    for &index in alive_groups {
        is_alive[index] = true;
    }
    normalized_weights
        .iter()
        .enumerate()
        .map(|(index, weight)| {
            if is_alive[index] {
                weight.union(&complement_alive)
            } else {
                Weight::empty()
            }
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BundleBuildProfile {
    pub(crate) input_terminals: usize,
    pub(crate) nonempty_terminals: usize,
    pub(crate) weight_groups: usize,
    pub(crate) singleton_groups: usize,
    pub(crate) multi_terminal_groups: usize,
    pub(crate) largest_weight_group: usize,
    pub(crate) build_group_dfas_ms: f64,
    pub(crate) union_groups_ms: f64,
    pub(crate) slowest_group_terminals: usize,
    pub(crate) slowest_group_dfa_states: usize,
    pub(crate) slowest_group_dfa_transitions: usize,
    pub(crate) slowest_group_ms: f64,
    pub(crate) determinize_bundle_ms: f64,
    pub(crate) determinize_pop_state_ms: f64,
    pub(crate) determinize_alive_groups_ms: f64,
    pub(crate) determinize_effective_weights_ms: f64,
    pub(crate) determinize_final_weight_ms: f64,
    pub(crate) determinize_collect_labels_ms: f64,
    pub(crate) determinize_next_state_ms: f64,
    pub(crate) determinize_edge_weight_ms: f64,
    pub(crate) determinize_state_lookup_ms: f64,
    pub(crate) determinize_add_transition_ms: f64,
    pub(crate) determinize_states_visited: usize,
    pub(crate) determinize_labels_processed: usize,
    pub(crate) determinize_transitions_added: usize,
    pub(crate) determinize_worklist_peak: usize,
    pub(crate) determinize_cache_entries: usize,
    pub(crate) minimize_ms: f64,
    pub(crate) dwa_to_nwa_ms: f64,
    pub(crate) result_dwa_states: usize,
    pub(crate) result_dwa_transitions: usize,
    pub(crate) result_nwa_states: usize,
    pub(crate) result_nwa_transitions: usize,
    pub(crate) total_ms: f64,
    pub(crate) used_single_terminal_fast_path: bool,
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn count_unweighted_dfa_transitions(dfa: &UnweightedDfa) -> usize {
    dfa.states.iter().map(|state| state.transitions.len()).sum()
}

fn count_weighted_dwa_transitions(dwa: &DWA) -> usize {
    dwa.states().iter().map(|state| state.transitions.len()).sum()
}

fn count_nwa_transitions(nwa: &NWA) -> usize {
    nwa.states()
        .iter()
        .map(|state| state.transitions.values().map(|targets| targets.len()).sum::<usize>() + state.epsilons.len())
        .sum()
}

impl Templates {
    fn build_single_terminal_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> Option<NWA> {
        let (&terminal, weight) = terminal_weights.iter().next()?;
        if terminal_weights.len() != 1 {
            return None;
        }
        if weight.is_empty() {
            return Some(empty_bundle_nwa());
        }
        Some(
            self.by_terminal_nwa
                .get(&terminal)
                .map(|template_nwa| instantiate_weighted_nwa_from_skeleton(template_nwa, weight))
                .unwrap_or_else(empty_bundle_nwa),
        )
    }

    fn group_terminals_by_weight<'a>(
        &'a self,
        terminal_weights: &'a BTreeMap<TerminalID, Weight>,
    ) -> HashMap<&'a Weight, Vec<TerminalID>> {
        let mut weight_groups: HashMap<&Weight, Vec<TerminalID>> = HashMap::new();
        for (&terminal, weight) in terminal_weights {
            if weight.is_empty() || !self.by_terminal.contains_key(&terminal) {
                continue;
            }
            weight_groups.entry(weight).or_default().push(terminal);
        }
        weight_groups
    }

    fn build_group_dfas<'a>(
        &'a self,
        weight_groups: &'a HashMap<&'a Weight, Vec<TerminalID>>,
    ) -> Vec<(&'a Weight, UnweightedDfa)> {
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            if terminals.len() == 1 {
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, template.clone()));
                }
                continue;
            }

            let merged = union_unweighted_dfas(
                terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
            );
            group_dfas.push((*weight, merged));
        }
        group_dfas
    }

    fn build_group_dfas_profiled<'a>(
        &'a self,
        weight_groups: &'a HashMap<&'a Weight, Vec<TerminalID>>,
        profile: &mut BundleBuildProfile,
    ) -> Vec<(&'a Weight, UnweightedDfa)> {
        let build_started_at = Instant::now();
        let mut group_dfas = Vec::with_capacity(weight_groups.len());
        for (weight, terminals) in weight_groups {
            profile.nonempty_terminals += terminals.len();
            profile.largest_weight_group = profile.largest_weight_group.max(terminals.len());
            if terminals.len() == 1 {
                profile.singleton_groups += 1;
                if let Some(template) = self.by_terminal.get(&terminals[0]) {
                    group_dfas.push((*weight, template.clone()));
                }
                continue;
            }

            profile.multi_terminal_groups += 1;
            let group_started_at = Instant::now();
            let merged = union_unweighted_dfas(
                terminals.iter().filter_map(|terminal| self.by_terminal.get(terminal)),
            );
            let group_ms = elapsed_ms(group_started_at);
            profile.union_groups_ms += group_ms;

            if group_ms > profile.slowest_group_ms {
                profile.slowest_group_ms = group_ms;
                profile.slowest_group_terminals = terminals.len();
                profile.slowest_group_dfa_states = merged.states.len();
                profile.slowest_group_dfa_transitions = count_unweighted_dfa_transitions(&merged);
            }

            group_dfas.push((*weight, merged));
        }
        profile.build_group_dfas_ms = elapsed_ms(build_started_at);
        group_dfas
    }

    pub(crate) fn build_bundle_profiled(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> (NWA, BundleBuildProfile) {
        let total_started_at = Instant::now();
        let mut profile = BundleBuildProfile {
            input_terminals: terminal_weights.len(),
            ..BundleBuildProfile::default()
        };

        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            profile.used_single_terminal_fast_path = true;
            profile.result_nwa_states = bundle.states().len();
            profile.result_nwa_transitions = count_nwa_transitions(&bundle);
            profile.total_ms = elapsed_ms(total_started_at);
            return (bundle, profile);
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        profile.weight_groups = weight_groups.len();
        let group_dfas = self.build_group_dfas_profiled(&weight_groups, &mut profile);

        let determinize_started_at = Instant::now();
        let (bundle_dwa, determinize_profile) = if parser_dwa_bundle_determinize_profile_enabled() {
            determinize_bundle_groups_profiled(&group_dfas)
        } else {
            (determinize_bundle_groups(&group_dfas), DeterminizeBundleProfile::default())
        };
        profile.determinize_bundle_ms = elapsed_ms(determinize_started_at);
        profile.determinize_pop_state_ms = determinize_profile.pop_state_ms;
        profile.determinize_alive_groups_ms = determinize_profile.alive_groups_ms;
        profile.determinize_effective_weights_ms = determinize_profile.effective_weights_ms;
        profile.determinize_final_weight_ms = determinize_profile.final_weight_ms;
        profile.determinize_collect_labels_ms = determinize_profile.collect_labels_ms;
        profile.determinize_next_state_ms = determinize_profile.next_state_ms;
        profile.determinize_edge_weight_ms = determinize_profile.edge_weight_ms;
        profile.determinize_state_lookup_ms = determinize_profile.state_lookup_ms;
        profile.determinize_add_transition_ms = determinize_profile.add_transition_ms;
        profile.determinize_states_visited = determinize_profile.states_visited;
        profile.determinize_labels_processed = determinize_profile.labels_processed;
        profile.determinize_transitions_added = determinize_profile.transitions_added;
        profile.determinize_worklist_peak = determinize_profile.worklist_peak;
        profile.determinize_cache_entries = determinize_profile.cache_entries;
        profile.result_dwa_states = bundle_dwa.states().len();
        profile.result_dwa_transitions = count_weighted_dwa_transitions(&bundle_dwa);

        let minimize_started_at = Instant::now();
        let minimized = if profile.weight_groups > 1 {
            minimize_from_env(&bundle_dwa, "GLRMASK_MINIMIZE_BUNDLE", minimize_fast)
        } else {
            bundle_dwa
        };
        profile.minimize_ms = elapsed_ms(minimize_started_at);
        profile.result_dwa_states = minimized.states().len();
        profile.result_dwa_transitions = count_weighted_dwa_transitions(&minimized);

        let to_nwa_started_at = Instant::now();
        let nwa = dwa_to_nwa(&minimized);
        profile.dwa_to_nwa_ms = elapsed_ms(to_nwa_started_at);
        profile.result_nwa_states = nwa.states().len();
        profile.result_nwa_transitions = count_nwa_transitions(&nwa);
        profile.total_ms = elapsed_ms(total_started_at);

        (nwa, profile)
    }

    /// Assemble a weighted NWA for one bundle of (terminal, weight) entries.
    ///
    /// Pipeline: group by weight, merge each group, determinize the product,
    /// optionally minimize it, then convert back to an NWA.
    pub(crate) fn build_bundle(
        &self,
        terminal_weights: &BTreeMap<TerminalID, Weight>,
    ) -> NWA {
        if let Some(bundle) = self.build_single_terminal_bundle(terminal_weights) {
            return bundle;
        }

        let weight_groups = self.group_terminals_by_weight(terminal_weights);
        let num_groups = weight_groups.len();
        let group_dfas = self.build_group_dfas(&weight_groups);
        let bundle_dwa = determinize_bundle_groups(&group_dfas);

        let minimized = if num_groups > 1 {
            minimize_from_env(&bundle_dwa, "GLRMASK_MINIMIZE_BUNDLE", minimize_fast)
        } else {
            bundle_dwa
        };

        dwa_to_nwa(&minimized)
    }
}

/// Specialized weighted determinize for bundles.
fn determinize_bundle_groups(groups: &[(&Weight, UnweightedDfa)]) -> DWA {
    use crate::automata::weighted_u32::dwa::DWA;

    let n = groups.len();
    if n == 0 {
        return DWA::new(0, 0);
    }

    const DEAD: u32 = u32::MAX;

    let union_all = Weight::union_all(groups.iter().map(|(w, _)| *w));
    let complement_all = union_all.complement();
    let normalized_weights: Vec<Weight> = groups
        .iter()
        .map(|(w, _)| (*w).union(&complement_all))
        .collect();
    let start_effective_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();

    let mut alive_cache: FxHashMap<Vec<usize>, Vec<Weight>> = FxHashMap::default();

    let start_key: Vec<u32> = groups.iter().map(|(_, dfa)| dfa.start_state).collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key.clone());

    let mut is_start = true;
    let mut all_labels: BTreeSet<i32> = BTreeSet::new();
    let mut next_state: Vec<u32> = vec![DEAD; n];

    while let Some(product_state) = worklist.pop_front() {
        let dwa_state = state_map[&product_state];

        let alive_groups: Vec<usize> = (0..n)
            .filter(|&i| product_state[i] != DEAD)
            .collect();

        let effective_weights: &Vec<Weight> = if is_start {
            &start_effective_weights
        } else {
            alive_cache
                .entry(alive_groups.clone())
                .or_insert_with(|| {
                    compute_effective_group_weights(&alive_groups, &normalized_weights)
                })
        };

        let mut final_w = Weight::empty();
        for &i in &alive_groups {
            if groups[i].1.states[product_state[i] as usize].is_accepting {
                final_w = final_w.union(&effective_weights[i]);
            }
        }
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }

        all_labels.clear();
        for &i in &alive_groups {
            for &label in groups[i].1.states[product_state[i] as usize]
                .transitions
                .keys()
            {
                all_labels.insert(label);
            }
        }

        for &label in &all_labels {
            for i in 0..n {
                next_state[i] = if product_state[i] == DEAD {
                    DEAD
                } else if let Some(&target) = groups[i]
                    .1
                    .states[product_state[i] as usize]
                    .transitions
                    .get(&label)
                {
                    target
                } else {
                    DEAD
                };
            }

            let mut edge_w = Weight::empty();
            for i in 0..n {
                if next_state[i] != DEAD {
                    edge_w = edge_w.union(&effective_weights[i]);
                }
            }
            if edge_w.is_empty() {
                continue;
            }

            let to_dwa = if let Some(&existing) = state_map.get(&*next_state) {
                existing
            } else {
                let key = next_state.clone();
                let new_id = dwa.add_state();
                state_map.insert(key.clone(), new_id);
                worklist.push_back(key);
                new_id
            };

            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
        }

        is_start = false;
    }

    dwa
}

#[derive(Clone, Debug, Default)]
struct DeterminizeBundleProfile {
    pop_state_ms: f64,
    alive_groups_ms: f64,
    effective_weights_ms: f64,
    final_weight_ms: f64,
    collect_labels_ms: f64,
    next_state_ms: f64,
    edge_weight_ms: f64,
    state_lookup_ms: f64,
    add_transition_ms: f64,
    states_visited: usize,
    labels_processed: usize,
    transitions_added: usize,
    worklist_peak: usize,
    cache_entries: usize,
}

fn determinize_bundle_groups_profiled(
    groups: &[(&Weight, UnweightedDfa)],
) -> (DWA, DeterminizeBundleProfile) {
    use crate::automata::weighted_u32::dwa::DWA;

    let mut profile = DeterminizeBundleProfile::default();

    let n = groups.len();
    if n == 0 {
        return (DWA::new(0, 0), profile);
    }

    const DEAD: u32 = u32::MAX;

    let union_all = Weight::union_all(groups.iter().map(|(w, _)| *w));
    let complement_all = union_all.complement();
    let normalized_weights: Vec<Weight> = groups
        .iter()
        .map(|(w, _)| (*w).union(&complement_all))
        .collect();
    let start_effective_weights: Vec<Weight> = groups
        .iter()
        .map(|(weight, _)| (*weight).clone())
        .collect();

    let mut alive_cache: FxHashMap<Vec<usize>, Vec<Weight>> = FxHashMap::default();

    let start_key: Vec<u32> = groups.iter().map(|(_, dfa)| dfa.start_state).collect();

    let mut dwa = DWA::new(0, 0);
    let mut state_map: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    state_map.insert(start_key.clone(), 0);
    worklist.push_back(start_key.clone());
    profile.worklist_peak = worklist.len();

    let mut is_start = true;
    let mut all_labels: BTreeSet<i32> = BTreeSet::new();
    let mut next_state: Vec<u32> = vec![DEAD; n];

    while let Some(product_state) = worklist.pop_front() {
        profile.states_visited += 1;
        let state_started_at = Instant::now();
        let dwa_state = state_map[&product_state];
        profile.pop_state_ms += elapsed_ms(state_started_at);

        let alive_started_at = Instant::now();
        let alive_groups: Vec<usize> = (0..n)
            .filter(|&i| product_state[i] != DEAD)
            .collect();
        profile.alive_groups_ms += elapsed_ms(alive_started_at);

        let effective_started_at = Instant::now();
        let effective_weights: &Vec<Weight> = if is_start {
            &start_effective_weights
        } else {
            alive_cache
                .entry(alive_groups.clone())
                .or_insert_with(|| {
                    compute_effective_group_weights(&alive_groups, &normalized_weights)
                })
        };
        profile.effective_weights_ms += elapsed_ms(effective_started_at);

        let final_started_at = Instant::now();
        let mut final_w = Weight::empty();
        for &i in &alive_groups {
            if groups[i].1.states[product_state[i] as usize].is_accepting {
                final_w = final_w.union(&effective_weights[i]);
            }
        }
        if !final_w.is_empty() {
            dwa.set_final_weight(dwa_state, final_w);
        }
        profile.final_weight_ms += elapsed_ms(final_started_at);

        let labels_started_at = Instant::now();
        all_labels.clear();
        for &i in &alive_groups {
            for &label in groups[i].1.states[product_state[i] as usize]
                .transitions
                .keys()
            {
                all_labels.insert(label);
            }
        }
        profile.collect_labels_ms += elapsed_ms(labels_started_at);
        profile.labels_processed += all_labels.len();

        for &label in &all_labels {
            let next_state_started_at = Instant::now();
            for i in 0..n {
                next_state[i] = if product_state[i] == DEAD {
                    DEAD
                } else if let Some(&target) = groups[i]
                    .1
                    .states[product_state[i] as usize]
                    .transitions
                    .get(&label)
                {
                    target
                } else {
                    DEAD
                };
            }
            profile.next_state_ms += elapsed_ms(next_state_started_at);

            let edge_weight_started_at = Instant::now();
            let mut edge_w = Weight::empty();
            for i in 0..n {
                if next_state[i] != DEAD {
                    edge_w = edge_w.union(&effective_weights[i]);
                }
            }
            if edge_w.is_empty() {
                profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);
                continue;
            }
            profile.edge_weight_ms += elapsed_ms(edge_weight_started_at);

            let lookup_started_at = Instant::now();
            let to_dwa = if let Some(&existing) = state_map.get(&*next_state) {
                existing
            } else {
                let key = next_state.clone();
                let new_id = dwa.add_state();
                state_map.insert(key.clone(), new_id);
                worklist.push_back(key);
                profile.worklist_peak = profile.worklist_peak.max(worklist.len());
                new_id
            };
            profile.state_lookup_ms += elapsed_ms(lookup_started_at);

            let add_transition_started_at = Instant::now();
            dwa.add_transition(dwa_state, label, to_dwa, edge_w);
            profile.add_transition_ms += elapsed_ms(add_transition_started_at);
            profile.transitions_added += 1;
        }

        is_start = false;
    }

    profile.cache_entries = alive_cache.len();

    (dwa, profile)
}

/// Union multiple unweighted DFAs into one DFA via NFA union + determinize + minimize.
fn union_unweighted_dfas<'a>(dfas: impl Iterator<Item = &'a UnweightedDfa>) -> UnweightedDfa {
    let mut nfa = UnweightedNfa::new_empty();
    let shared_start = nfa.add_state();
    nfa.start_states.push(shared_start);

    for dfa in dfas {
        if dfa.states.is_empty() {
            continue;
        }
        let offset = nfa.states.len() as u32;
        for _ in &dfa.states {
            nfa.add_state();
        }
        // Epsilon from shared start to this DFA's start.
        nfa.add_epsilon(shared_start, offset + dfa.start_state);
        for (state_id, state) in dfa.states.iter().enumerate() {
            let from = offset + state_id as u32;
            if state.is_accepting {
                nfa.set_accepting(from);
            }
            for (&label, &target) in &state.transitions {
                nfa.add_transition(from, label, offset + target);
            }
        }
    }

    let det = unweighted_determinize(&nfa);
    unweighted_minimize(&det)
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let states = dwa
        .states()
        .iter()
        .map(|state| NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| (label, vec![(*target, weight.clone())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();

    NWA::from_parts(
        states,
        vec![dwa.start_state()],
    )
}
