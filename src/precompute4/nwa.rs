// src/precompute4/nwa.rs

#![allow(dead_code)]

use crate::precompute4::weighted_automata::{DWA, DWABody, DWAState, DWAStates, StateID, Weight};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::{Deref, Index, IndexMut};

// --- NWA Definitions ---

#[derive(Clone, Debug, Default)]
pub struct NWAState {
    pub transitions: BTreeMap<i16, StateID>,
    pub trans_weights: BTreeMap<i16, Weight>,
    pub epsilon_transitions: Vec<(StateID, Weight)>,
    pub final_weight: Option<Weight>,
}

impl NWAState {
    /// Intersects all weights in this state with the given weight.
    pub fn apply_weight(&mut self, weight: &Weight) {
        if let Some(fw) = &mut self.final_weight {
            *fw &= weight;
            if fw.is_empty() {
                self.final_weight = None;
            }
        }
        for w in self.trans_weights.values_mut() {
            *w &= weight;
        }
        for (_, w) in &mut self.epsilon_transitions {
            *w &= weight;
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NWAStates(pub Vec<NWAState>);

impl Index<usize> for NWAStates {
    type Output = NWAState;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for NWAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl Deref for NWAStates {
    type Target = [NWAState];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl NWAStates {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(NWAState::default());
        id
    }
    pub fn add_existing_state(&mut self, state: NWAState) -> StateID {
        let id = self.0.len();
        self.0.push(state);
        id
    }

    pub fn copy_subgraph_from(
        &mut self,
        other_states: &NWAStates,
        start_id: StateID,
    ) -> (StateID, HashMap<StateID, StateID>) {
        let mut remap = HashMap::new();
        let mut q = VecDeque::new();

        if start_id >= other_states.len() {
            let new_start_id = self.add_state();
            return (new_start_id, remap);
        }

        let new_start_id = self.add_existing_state(other_states[start_id].clone());
        remap.insert(start_id, new_start_id);
        q.push_back((start_id, new_start_id));

        while let Some((old_id, new_id)) = q.pop_front() {
            let old_state_clone = other_states[old_id].clone();

            // Remap non-epsilon transitions
            self[new_id].transitions = old_state_clone
                .transitions
                .iter()
                .map(|(ch, &old_target)| {
                    let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                        let new_id = self.add_existing_state(other_states[old_target].clone());
                        q.push_back((old_target, new_id));
                        new_id
                    });
                    (*ch, new_target_id)
                })
                .collect();

            // Remap epsilon transitions
            self[new_id].epsilon_transitions = old_state_clone
                .epsilon_transitions
                .iter()
                .map(|(old_target, weight)| {
                    let new_target_id = *remap.entry(*old_target).or_insert_with(|| {
                        let new_id = self.add_existing_state(other_states[*old_target].clone());
                        q.push_back((*old_target, new_id));
                        new_id
                    });
                    (new_target_id, weight.clone())
                })
                .collect();
        }
        (new_start_id, remap)
    }

    pub fn apply_weight_to_subgraph(&mut self, start_id: StateID, weight: &Weight) {
        let mut visited = BTreeSet::new();
        let mut q = VecDeque::new();

        if start_id < self.len() {
            q.push_back(start_id);
            visited.insert(start_id);
        }

        while let Some(state_id) = q.pop_front() {
            self[state_id].apply_weight(weight);

            let state_clone = self[state_id].clone();
            for &target_id in state_clone.transitions.values() {
                if visited.insert(target_id) {
                    q.push_back(target_id);
                }
            }
            for (target_id, _) in state_clone.epsilon_transitions {
                if visited.insert(target_id) {
                    q.push_back(target_id);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NWABody {
    pub start_state: StateID,
}

#[derive(Clone, Debug, Default)]
pub struct NWA {
    pub states: NWAStates,
    pub body: NWABody,
}

impl NWA {
    pub fn new() -> Self {
        let mut states = NWAStates::default();
        let start = states.add_state();
        NWA { states, body: NWABody { start_state: start } }
    }

    pub fn from_dwa(dwa: &DWA) -> Self {
        let mut nwa_states = NWAStates::default();
        nwa_states.0.reserve(dwa.states.len());

        for d_state in &dwa.states.0 {
            let mut n_state = NWAState::default();
            n_state.final_weight = d_state.final_weight.clone();

            if let Some(def_target) = d_state.transitions.default {
                // This is a simplification. We are not expanding the default transition
                // into all possible character transitions. Determinization will handle this.
                // For now, we ignore default transitions in the conversion, as our
                // determinizer doesn't support them directly.
                // A more robust implementation would expand them.
            }

            for (&ch, &target) in &d_state.transitions.exceptions {
                n_state.transitions.insert(ch, target);
                if let Some(w) = d_state.trans_weights_exceptions.get(&ch) {
                    n_state.trans_weights.insert(ch, w.clone());
                }
            }
            nwa_states.add_existing_state(n_state);
        }

        NWA { states: nwa_states, body: NWABody { start_state: dwa.body.start_state } }
    }

    /// Component-based union. Operates on a shared NWAStates arena.
    pub fn union_components(states: &mut NWAStates, body1: &NWABody, body2: &NWABody) -> NWABody {
        let new_start = states.add_state();
        states[new_start].epsilon_transitions.push((body1.start_state, Weight::all()));
        states[new_start].epsilon_transitions.push((body2.start_state, Weight::all()));
        NWABody { start_state: new_start }
    }

    /// Component-based concatenation. Operates on a shared NWAStates arena.
    pub fn concatenate_components(
        states: &mut NWAStates,
        body1: &NWABody,
        body2: &NWABody,
    ) -> NWABody {
        // 1. Copy left subgraph to not modify shared states.
        let (new_start_a, remap_a) =
            states.copy_subgraph_from(&states.clone(), body1.start_state);

        // 2. Find final states in the copied left automaton.
        let final_a_states: Vec<(StateID, Weight)> = remap_a
            .values()
            .filter_map(|&new_id| states[new_id].final_weight.clone().map(|w| (new_id, w)))
            .collect();

        for (s_a_id, final_weight) in final_a_states {
            // Add epsilon transition from final state of A to start of B
            states[s_a_id].epsilon_transitions.push((body2.start_state, final_weight));

            // If B's start is final, A's final state needs to incorporate that.
            if let Some(b_start_fw) = &states[body2.start_state].final_weight {
                let new_final =
                    states[s_a_id].final_weight.as_ref().unwrap() & b_start_fw;
                if new_final.is_empty() {
                    states[s_a_id].final_weight = None;
                } else {
                    states[s_a_id].final_weight = Some(new_final);
                }
            } else {
                // The finality is "passed over" to B.
                states[s_a_id].final_weight = None;
            }
        }

        // If start of A was final, the new start state must also be connected to B's start.
        if let Some(start_a_final_weight) = states[body1.start_state].final_weight.clone() {
            states[new_start_a].epsilon_transitions.push((body2.start_state, start_a_final_weight));
        }

        NWABody { start_state: new_start_a }
    }

    /// Component-level variant: create a new start state gated by the weight.
    pub fn apply_weight_components(
        states: &mut NWAStates,
        body: &mut NWABody,
        weight: &Weight,
    ) -> StateID {
        let old_start = body.start_state;
        let new_start = states.add_state();
        states[new_start].epsilon_transitions.push((old_start, weight.clone()));
        body.start_state = new_start;
        new_start
    }

    pub fn determinize(&self) -> DWA {
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        if self.states.len() == 0 {
            dwa.add_state();
            return dwa;
        }

        let mut dwa_state_map: BTreeMap<BTreeSet<StateID>, StateID> = BTreeMap::new();
        let mut worklist: VecDeque<BTreeSet<StateID>> = VecDeque::new();

        let start_nwa_states = self.epsilon_closure_states(self.body.start_state);
        if start_nwa_states.is_empty() {
            dwa.add_state();
            return dwa;
        }

        let start_dwa_id = dwa.add_state();
        dwa.body.start_state = start_dwa_id;
        dwa_state_map.insert(start_nwa_states.clone(), start_dwa_id);
        worklist.push_back(start_nwa_states);

        while let Some(current_nwa_states) = worklist.pop_front() {
            let current_dwa_id = *dwa_state_map.get(&current_nwa_states).unwrap();

            let mut final_weight = Weight::zeros();
            for &nwa_state_id in &current_nwa_states {
                if let Some(fw) = &self.states[nwa_state_id].final_weight {
                    final_weight |= fw;
                }
            }
            if !final_weight.is_empty() {
                dwa.states[current_dwa_id].final_weight = Some(final_weight);
            }

            let mut transitions: BTreeMap<i16, (BTreeSet<StateID>, Weight)> = BTreeMap::new();
            for &nwa_state_id in &current_nwa_states {
                let nwa_state = &self.states[nwa_state_id];
                for (&symbol, &target_id) in &nwa_state.transitions {
                    let weight = nwa_state.trans_weights.get(&symbol).unwrap();
                    let entry = transitions.entry(symbol).or_default();
                    entry.0.insert(target_id);
                    entry.1 |= weight;
                }
            }

            for (symbol, (target_nwa_states, trans_weight)) in transitions {
                let next_nwa_states_closure = self.epsilon_closure_of_set(&target_nwa_states);

                if next_nwa_states_closure.is_empty() {
                    continue;
                }

                let next_dwa_id =
                    *dwa_state_map.entry(next_nwa_states_closure.clone()).or_insert_with(|| {
                        let new_id = dwa.add_state();
                        worklist.push_back(next_nwa_states_closure);
                        new_id
                    });

                dwa.add_transition(current_dwa_id, symbol, next_dwa_id, trans_weight)
                    .ok(); // Ignore error if transition exists
            }
        }

        dwa
    }

    fn epsilon_closure_states(&self, start_id: StateID) -> BTreeSet<StateID> {
        let mut closure = BTreeSet::new();
        let mut worklist = vec![start_id];
        closure.insert(start_id);

        while let Some(state_id) = worklist.pop() {
            for (target_id, _) in &self.states[state_id].epsilon_transitions {
                if closure.insert(*target_id) {
                    worklist.push(*target_id);
                }
            }
        }
        closure
    }

    fn epsilon_closure_of_set(&self, states: &BTreeSet<StateID>) -> BTreeSet<StateID> {
        let mut closure = states.clone();
        let mut worklist: Vec<StateID> = states.iter().copied().collect();

        while let Some(state_id) = worklist.pop() {
            if state_id >= self.states.len() { continue; }
            for (target_id, _) in &self.states[state_id].epsilon_transitions {
                if closure.insert(*target_id) {
                    worklist.push(*target_id);
                }
            }
        }
        closure
    }
}
