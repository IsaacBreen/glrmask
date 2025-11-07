// src/precompute4/weighted_automata/determinization.rs
//
// New determinization strategy: per-bit-plane decomposition.
// The weight on edges is a bitset (SimpleBitset). The automaton semantics
// (intersection on path, union over choices) are bitwise operations. This means
// we can decompose the weighted automaton problem into N independent unweighted
// automaton problems, one for each bit in the bitset.
//
// The algorithm is as follows:
// 1. For each bit `i` in the `Weight` bitset, construct an NFA (`NFA_i`) where
//    an edge exists if and only if the `i`-th bit is set in the corresponding
//    weight in the original NWA.
// 2. Many of these `NFA_i` might be structurally identical. We hash them to find
//    the set of unique NFAs.
// 3. Each unique NFA is determinized to a DFA using standard subset construction.
//    This is an unweighted determinization, which is much simpler and faster
//    than weighted determinization.
// 4. The final DWA is constructed as a product of all the unique DFAs. A state
//    in the final DWA corresponds to a tuple of states, one from each unique DFA.
//    We only build the reachable states of this product automaton.
// 5. Transitions in the final DWA have weight `Weight::all()`. The logic is
//    encoded in the structure of the automaton.
// 6. The final weight of a state in the DWA is a bitset where bit `i` is set
//    if the corresponding component DFA state (for bit `i`) is an accepting state.
//
// This approach is effective because the structure of the NWA (few SCCs with
// mostly `Weight::all()` internally) leads to many identical or simple bit-plane
// NFAs, making the DFAs and their product manageable in size.

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::Weight;
use super::dwa::DWA;
use super::nwa::{NWAStates, NWA};
use crate::precompute4::weighted_automata::{StateID, NWAStateID};

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

impl NWA {
    pub fn determinize_to_dwa(&self) -> DWA {
        let mut nwa = self.clone();
        nwa.simplify();
        if nwa.states.len() == 0 {
            return DWA::new();
        }
        nwa.determinize_by_bits()
    }

    fn determinize_by_bits(&self) -> DWA {
        let num_bits = Weight::all().len();
        if num_bits == 0 {
            return DWA::new();
        }

        // 1. Create NFAs for each bit plane, identifying unique ones.
        let mut bit_to_dfa_idx = vec![0; num_bits];
        let mut unique_dfas = Vec::<DFA>::new();
        let mut nfa_to_dfa_idx = HashMap::<NFA, usize>::new();

        for i in 0..num_bits {
            let nfa = NFA::from_nwa(self, i);
            if let Some(&dfa_idx) = nfa_to_dfa_idx.get(&nfa) {
                bit_to_dfa_idx[i] = dfa_idx;
            } else {
                let dfa = nfa.determinize();
                let dfa_idx = unique_dfas.len();
                unique_dfas.push(dfa);
                nfa_to_dfa_idx.insert(nfa, dfa_idx);
                bit_to_dfa_idx[i] = dfa_idx;
            }
        }

        // 2. Product construction of unique DFAs.
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let mut product_map = HashMap::<Vec<StateID>, StateID>::new();
        let mut worklist = VecDeque::<StateID>::new();
        let mut dwa_id_to_product_state = Vec::<Vec<StateID>>::new();

        let get_or_create_dwa_state = |product_state: Vec<StateID>,
                                       dwa: &mut DWA,
                                       product_map: &mut HashMap<Vec<StateID>, StateID>,
                                       worklist: &mut VecDeque<StateID>,
                                       dwa_id_to_product_state: &mut Vec<Vec<StateID>>|
         -> StateID {
            *product_map.entry(product_state.clone()).or_insert_with(|| {
                let new_id = dwa.add_state();
                worklist.push_back(new_id);
                dwa_id_to_product_state.push(product_state);
                new_id
            })
        };

        let start_product_state: Vec<StateID> =
            unique_dfas.iter().map(|dfa| dfa.start_state).collect();
        dwa.body.start_state = get_or_create_dwa_state(
            start_product_state,
            &mut dwa,
            &mut product_map,
            &mut worklist,
            &mut dwa_id_to_product_state,
        );

        while let Some(dwa_id) = worklist.pop_front() {
            let product_state = &dwa_id_to_product_state[dwa_id];

            // Collect all exception labels from all component DFA states.
            let mut labels = BTreeSet::new();
            for (dfa_idx, &dfa_state_id) in product_state.iter().enumerate() {
                for &label in unique_dfas[dfa_idx].states[dfa_state_id].transitions.keys() {
                    labels.insert(label);
                }
            }

            // Default transition
            let default_target_prod: Vec<StateID> = product_state
                .iter()
                .enumerate()
                .map(|(dfa_idx, &dfa_state_id)| unique_dfas[dfa_idx].get_next_state(dfa_state_id, None))
                .collect();
            let default_dwa_target = get_or_create_dwa_state(
                default_target_prod.clone(),
                &mut dwa,
                &mut product_map,
                &mut worklist,
                &mut dwa_id_to_product_state,
            );
            dwa.set_default_transition(dwa_id, default_dwa_target, Weight::all()).unwrap();

            // Exception transitions
            for label in labels {
                let target_prod: Vec<StateID> = product_state
                    .iter()
                    .enumerate()
                    .map(|(dfa_idx, &dfa_state_id)| {
                        unique_dfas[dfa_idx].get_next_state(dfa_state_id, Some(label))
                    })
                    .collect();

                // Only add an exception if it's different from the default.
                if target_prod != default_target_prod {
                    let dwa_target = get_or_create_dwa_state(
                        target_prod,
                        &mut dwa,
                        &mut product_map,
                        &mut worklist,
                        &mut dwa_id_to_product_state,
                    );
                    dwa.add_transition(dwa_id, label, dwa_target, Weight::all()).unwrap();
                }
            }
        }

        // 3. Set final weights.
        for dwa_id in 0..dwa.states.len() {
            let product_state = &dwa_id_to_product_state[dwa_id];
            let mut final_weight = Weight::zeros();
            for bit in 0..num_bits {
                let dfa_idx = bit_to_dfa_idx[bit];
                let dfa_state_id = product_state[dfa_idx];
                if unique_dfas[dfa_idx].states[dfa_state_id].is_final {
                    final_weight.insert(bit);
                }
            }
            if !final_weight.is_empty() {
                dwa.set_final_weight(dwa_id, final_weight).unwrap();
            }
        }

        dwa
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct NFAState {
    transitions: BTreeMap<i16, Vec<NWAStateID>>,
    epsilon: Vec<NWAStateID>,
    default: Vec<NWAStateID>,
    is_final: bool,
}

#[derive(Debug, Default, PartialEq, Eq, Hash)]
struct NFA {
    states: Vec<NFAState>,
    start_state: NWAStateID,
}

impl NFA {
    fn from_nwa(nwa: &NWA, bit: usize) -> Self {
        let mut states = vec![NFAState::default(); nwa.states.len()];
        for (id, nwa_state) in nwa.states.0.iter().enumerate() {
            if let Some(w) = &nwa_state.final_weight {
                if w.contains(bit) {
                    states[id].is_final = true;
                }
            }
            for (on, (to, w)) in &nwa_state.transitions {
                if w.contains(bit) {
                    states[id].transitions.entry(*on).or_default().push(*to);
                }
            }
            for (to, w) in &nwa_state.epsilons {
                if w.contains(bit) {
                    states[id].epsilon.push(*to);
                }
            }
            if let Some((to, w)) = &nwa_state.default {
                if w.contains(bit) {
                    states[id].default.push(*to);
                }
            }
        }
        Self { states, start_state: nwa.body.start_state }
    }

    fn epsilon_closure(&self, initial_states: &BTreeSet<NWAStateID>) -> BTreeSet<NWAStateID> {
        let mut closure = initial_states.clone();
        let mut queue: VecDeque<_> = initial_states.iter().copied().collect();
        let mut visited = initial_states.clone();

        while let Some(u) = queue.pop_front() {
            for &v in &self.states[u].epsilon {
                if v < self.states.len() && !visited.contains(&v) {
                    closure.insert(v);
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
        }
        closure
    }

    fn determinize(&self) -> DFA {
        let mut dfa = DFA::default();
        let mut worklist = VecDeque::new();
        let mut subset_map = HashMap::new();

        let start_subset = self.epsilon_closure(&BTreeSet::from([self.start_state]));
        dfa.start_state = dfa.get_or_create_state(&start_subset, self, &mut subset_map, &mut worklist);

        while let Some(subset) = worklist.pop_front() {
            let from_dfa_id = *subset_map.get(&subset).unwrap();

            let mut labels = BTreeSet::new();
            for &nfa_id in &subset {
                if nfa_id < self.states.len() {
                    for &label in self.states[nfa_id].transitions.keys() {
                        labels.insert(label);
                    }
                }
            }

            // Default transition
            let mut default_next = BTreeSet::new();
            for &nfa_id in &subset {
                if nfa_id < self.states.len() {
                    default_next.extend(&self.states[nfa_id].default);
                }
            }
            let default_subset = self.epsilon_closure(&default_next);
            let to_dfa_id = dfa.get_or_create_state(&default_subset, self, &mut subset_map, &mut worklist);
            dfa.states[from_dfa_id].default = Some(to_dfa_id);

            // Exception transitions
            for label in labels {
                let mut next_states = BTreeSet::new();
                for &nfa_id in &subset {
                    if nfa_id < self.states.len() {
                        if let Some(targets) = self.states[nfa_id].transitions.get(&label) {
                            next_states.extend(targets);
                        } else {
                            next_states.extend(&self.states[nfa_id].default);
                        }
                    }
                }
                let next_subset = self.epsilon_closure(&next_states);
                let to_dfa_id = dfa.get_or_create_state(&next_subset, self, &mut subset_map, &mut worklist);
                dfa.states[from_dfa_id].transitions.insert(label, to_dfa_id);
            }
        }
        dfa
    }
}

#[derive(Debug, Default)]
struct DFAState {
    transitions: BTreeMap<i16, StateID>,
    default: Option<StateID>,
    is_final: bool,
}

#[derive(Debug, Default)]
struct DFA {
    states: Vec<DFAState>,
    start_state: StateID,
}

impl DFA {
    fn get_or_create_state(
        &mut self,
        subset: &BTreeSet<NWAStateID>,
        nfa: &NFA,
        subset_map: &mut HashMap<BTreeSet<NWAStateID>, StateID>,
        worklist: &mut VecDeque<BTreeSet<NWAStateID>>,
    ) -> StateID {
        if let Some(&id) = subset_map.get(subset) {
            return id;
        }
        let new_id = self.states.len();
        let is_final = subset.iter().any(|&id| id < nfa.states.len() && nfa.states[id].is_final);
        self.states.push(DFAState { is_final, ..Default::default() });
        subset_map.insert(subset.clone(), new_id);
        if !subset.is_empty() {
            worklist.push_back(subset.clone());
        }
        new_id
    }

    fn get_next_state(&self, from_state_id: StateID, on: Option<i16>) -> StateID {
        let state = &self.states[from_state_id];
        if let Some(on_char) = on {
            if let Some(&to) = state.transitions.get(&on_char) {
                return to;
            }
        }
        state.default.expect("DFA state should have a default transition")
    }
}