//! Template bundle assembly into a weighted NWA.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use rustc_hash::FxHashMap;

use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::automata::unweighted_u32::determinize::determinize as unweighted_determinize;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as unweighted_minimize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize_fast;
use crate::automata::weighted::nwa::{NWA, NWAState};
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::templates::compile_dfa::Templates;
use crate::ds::weight::Weight;

fn empty_bundle_nwa() -> NWA {
    let mut nwa = NWA::new(0, 0);
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);
    nwa
}

fn instantiate_weighted_nwa_from_skeleton(skeleton: &NWA, weight: &Weight) -> NWA {
    let mut bundle = skeleton.clone();
    for state in &mut bundle.states {
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
            minimize_fast(&bundle_dwa)
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
        .states
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

    NWA {
        states,
        start_states: vec![dwa.start_state],
    }
}
