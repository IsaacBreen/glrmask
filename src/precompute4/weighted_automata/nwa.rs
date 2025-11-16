// src/precompute4/weighted_automata/nwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_i16_char, NWAStateID, Weight};
use super::dwa::{DWA, DWAState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::ops::{Index, IndexMut};
use crate::precompute4::weighted_automata::determinization_rustfst::determinize_nwa_to_dwa;
use crate::precompute4::weighted_automata::StateID;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NWABuildError {
    StateOutOfBounds { state: NWAStateID },
}

impl Display for NWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NWABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NWAState {
    pub final_weight: Option<Weight>,
    pub transitions: BTreeMap<i16, Vec<(NWAStateID, Weight)>>,
    pub epsilons: Vec<(NWAStateID, Weight)>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NWAStates(pub Vec<NWAState>);

impl NWAStates {
    pub fn len(&self) -> usize { self.0.len() }

    pub fn add_state(&mut self) -> NWAStateID {
        let id = self.0.len();
        self.0.push(NWAState::default());
        id
    }

    pub fn copy_state(&mut self, state_id: NWAStateID) -> NWAStateID {
        assert!(state_id < self.len(), "copy_state: state_id out of bounds");
        let new_id = self.add_state();
        self.0[new_id] = self.0[state_id].clone();
        new_id
    }

    pub fn add_epsilon(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) {
        assert!(from < self.len() && to < self.len(), "add_epsilon: state id out of bounds");
        self.0[from].epsilons.push((to, w));
    }

    pub fn add_transition(
        &mut self,
        from: NWAStateID,
        on: i16,
        to: NWAStateID,
        w: Weight,
    ) -> Result<(), NWABuildError> {
        if from >= self.len() {
            return Err(NWABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.len() {
            return Err(NWABuildError::StateOutOfBounds { state: to });
        }
        self.0[from].transitions.entry(on).or_default().push((to, w));
        Ok(())
    }

    pub fn copy_subgraph_from_and_return_body(&mut self, other: &NWAStates, body: NWABody) -> NWABody {
        let (new_start, _remap) = self.copy_subgraph_from(other, body.start_state);
        NWABody { start_state: new_start }
    }

    pub fn copy_subgraph_in_place_and_return_body(&mut self, body: NWABody) -> NWABody {
        let (new_start, _remap) = self.copy_subgraph_in_place(body.start_state);
        NWABody { start_state: new_start }
    }

    pub fn copy_subgraph_in_place(&mut self, start_id: NWAStateID) -> (NWAStateID, HashMap<NWAStateID, NWAStateID>) {
        let mut remap: HashMap<NWAStateID, NWAStateID> = HashMap::new();
        if start_id >= self.len() {
            let new_start = self.add_state();
            return (new_start, remap);
        }
        let new_start = self.add_state();
        self.0[new_start] = self.0[start_id].clone();
        remap.insert(start_id, new_start);

        let mut q = VecDeque::new();
        q.push_back((start_id, new_start));

        while let Some((old, new)) = q.pop_front() {
            let eps = self.0[old].epsilons.clone();
            self.0[new].epsilons.clear();
            for (to_old, w) in eps {
                let to_new = *remap.entry(to_old).or_insert_with(|| {
                    let n = self.add_state();
                    self.0[n] = self.0[to_old].clone();
                    q.push_back((to_old, n));
                    n
                });
                self.0[new].epsilons.push((to_new, w.clone()));
            }
            let trans = self.0[old].transitions.clone();
            self.0[new].transitions.clear();
            for (lbl, targets) in trans {
                for (to_old, w) in targets {
                    let to_new = *remap.entry(to_old).or_insert_with(|| {
                        let n = self.add_state();
                        self.0[n] = self.0[to_old].clone();
                        q.push_back((to_old, n));
                        n
                    });
                    self.0[new].transitions.entry(lbl).or_default().push((to_new, w.clone()));
                }
            }
        }

        (new_start, remap)
    }

    pub fn copy_subgraph_from(
        &mut self,
        other: &NWAStates,
        start_id: NWAStateID,
    ) -> (NWAStateID, HashMap<NWAStateID, NWAStateID>) {
        let mut remap: HashMap<NWAStateID, NWAStateID> = HashMap::new();
        if start_id >= other.len() {
            let new_start = self.add_state();
            return (new_start, remap);
        }
        let new_start = self.add_state();
        self.0[new_start] = other.0[start_id].clone();
        remap.insert(start_id, new_start);

        let mut q = VecDeque::new();
        q.push_back((start_id, new_start));

        while let Some((old, new)) = q.pop_front() {
            let eps = other.0[old].epsilons.clone();
            self.0[new].epsilons.clear();
            for (to_old, w) in eps {
                let to_new = *remap.entry(to_old).or_insert_with(|| {
                    let n = self.add_state();
                    self.0[n] = other.0[to_old].clone();
                    q.push_back((to_old, n));
                    n
                });
                self.0[new].epsilons.push((to_new, w.clone()));
            }
            let trans = other.0[old].transitions.clone();
            self.0[new].transitions.clear();
            for (lbl, targets) in trans {
                for (to_old, w) in targets {
                    let to_new = *remap.entry(to_old).or_insert_with(|| {
                        let n = self.add_state();
                        self.0[n] = other.0[to_old].clone();
                        q.push_back((to_old, n));
                        n
                    });
                    self.0[new].transitions.entry(lbl).or_default().push((to_new, w.clone()));
                }
            }
        }

        (new_start, remap)
    }
}

impl Index<NWAStateID> for NWAStates {
    type Output = NWAState;
    fn index(&self, index: NWAStateID) -> &Self::Output { &self.0[index] }
}

impl IndexMut<NWAStateID> for NWAStates {
    fn index_mut(&mut self, index: NWAStateID) -> &mut Self::Output { &mut self.0[index] }
}

impl Display for NWAStates {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWAStates ({} states):", self.0.len())?;
        for (id, state) in self.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", w)?;
            }
            for (on, targets) in &state.transitions {
                for (to, w) in targets {
                    let char_repr = format_i16_char(*on);
                    writeln!(f, "    {} -> {} (weight: {})", char_repr, to, w)?;
                }
            }
            for (to, w) in &state.epsilons {
                writeln!(f, "    ε -> {} (weight: {})", to, w)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct NWAStats {
    pub num_states: usize,
    pub num_final_states: usize,
    pub total_epsilon_transitions: usize,
    pub total_labeled_transitions: usize,
    pub avg_epsilon_per_state: f64,
    pub avg_labeled_per_state: f64,
}

impl Display for NWAStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "NWA Stats:")?;
        writeln!(f, "  - States: {}", self.num_states)?;
        writeln!(f, "  - Final States: {}", self.num_final_states)?;
        writeln!(f, "  - Epsilon Transitions: {}", self.total_epsilon_transitions)?;
        writeln!(f, "  - Labeled Transitions: {}", self.total_labeled_transitions)?;
        writeln!(f, "  - Avg Epsilon/State: {:.2}", self.avg_epsilon_per_state)?;
        writeln!(f, "  - Avg Labeled/State: {:.2}", self.avg_labeled_per_state)
    }
}

impl NWA {
    pub fn stats(&self) -> NWAStats {
        let num_states = self.states.len();
        if num_states == 0 {
            return NWAStats {
                num_states: 0,
                num_final_states: 0,
                total_epsilon_transitions: 0,
                total_labeled_transitions: 0,
                avg_epsilon_per_state: 0.0,
                avg_labeled_per_state: 0.0,
            };
        }

        let mut num_final_states = 0;
        let mut total_epsilon_transitions = 0;
        let mut total_labeled_transitions = 0;

        for state in &self.states.0 {
            if state.final_weight.is_some() {
                num_final_states += 1;
            }
            total_epsilon_transitions += state.epsilons.len();
            total_labeled_transitions += state.transitions.values().map(|v| v.len()).sum::<usize>();
        }

        let avg_epsilon_per_state = total_epsilon_transitions as f64 / num_states as f64;
        let avg_labeled_per_state = total_labeled_transitions as f64 / num_states as f64;

        NWAStats {
            num_states,
            num_final_states,
            total_epsilon_transitions,
            total_labeled_transitions,
            avg_epsilon_per_state,
            avg_labeled_per_state,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NWABody {
    pub start_state: NWAStateID,
}

impl Display for NWABody {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "NWABody (start: {})", self.start_state) }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NWA {
    pub states: NWAStates,
    pub body: NWABody,
}

impl NWA {
    pub fn new() -> Self {
        let mut states = NWAStates::default();
        let start = states.add_state();
        Self { states, body: NWABody { start_state: start } }
    }

    pub fn add_state(&mut self) -> StateID { self.states.add_state() }

    pub fn add_transition(
        &mut self,
        from: NWAStateID,
        on: i16,
        to: NWAStateID,
        w: Weight,
    ) -> Result<(), NWABuildError> {
        self.states.add_transition(from, on, to, w)
    }

    pub fn add_epsilon(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) { self.states.add_epsilon(from, to, w); }

    pub fn determinize_to_dwa(&self) -> DWA { self.determinize() }

    fn epsilon_closure(&self, subset: &BTreeMap<NWAStateID, Weight>) -> BTreeMap<NWAStateID, Weight> {
        let mut closure = subset.clone();
        let mut worklist: VecDeque<NWAStateID> = subset.keys().copied().collect();

        while let Some(u) = worklist.pop_front() {
            let u_weight = closure.get(&u).unwrap().clone();
            if u >= self.states.len() {
                continue;
            }
            for (v, eps_weight) in &self.states[u].epsilons {
                let v_new_weight = &u_weight & eps_weight;
                if !v_new_weight.is_empty() {
                    let v_current_weight = closure.entry(*v).or_insert_with(Weight::zeros);
                    let combined = &*v_current_weight | &v_new_weight;
                    if combined != *v_current_weight {
                        *v_current_weight = combined;
                        worklist.push_back(*v);
                    }
                }
            }
        }
        closure
    }

    pub fn determinize(&self) -> DWA {
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let mut subset_map: HashMap<BTreeMap<NWAStateID, Weight>, NWAStateID> = HashMap::new();
        let mut worklist: VecDeque<BTreeMap<NWAStateID, Weight>> = VecDeque::new();

        let mut start_subset = BTreeMap::new();
        if self.body.start_state < self.states.len() {
            start_subset.insert(self.body.start_state, Weight::all());
            let initial_subset = self.epsilon_closure(&start_subset);

            if !initial_subset.is_empty() {
                let start_id = dwa.add_state();
                dwa.body.start_state = start_id;
                subset_map.insert(initial_subset.clone(), start_id);
                worklist.push_back(initial_subset);
            } else {
                dwa.body.start_state = dwa.add_state();
            }
        } else {
            dwa.body.start_state = dwa.add_state();
        }

        while let Some(subset) = worklist.pop_front() {
            let from_dwa_id = *subset_map.get(&subset).unwrap();

            let mut final_weight = Weight::zeros();
            for (nwa_id, path_weight) in &subset {
                if let Some(fw) = &self.states[*nwa_id].final_weight {
                    final_weight |= &(path_weight & fw);
                }
            }
            if !final_weight.is_empty() {
                dwa.states[from_dwa_id].final_weight = Some(final_weight);
            }

            let mut transitions: BTreeMap<i16, BTreeMap<NWAStateID, Weight>> = BTreeMap::new();
            for (nwa_id, path_weight) in &subset {
                for (label, targets) in &self.states[*nwa_id].transitions {
                    for (target_nwa_id, trans_weight) in targets {
                        let next_path_weight = path_weight & trans_weight;
                        if !next_path_weight.is_empty() {
                            let entry = transitions.entry(*label).or_default();
                            *entry.entry(*target_nwa_id).or_insert_with(Weight::zeros) |= &next_path_weight;
                        }
                    }
                }
            }

            for (label, next_subset_pre_closure) in transitions {
                let next_subset = self.epsilon_closure(&next_subset_pre_closure);
                if next_subset.is_empty() {
                    continue;
                }
                let to_dwa_id = *subset_map.entry(next_subset.clone()).or_insert_with(|| {
                    let new_id = dwa.add_state();
                    worklist.push_back(next_subset);
                    new_id
                });
                dwa.add_transition(from_dwa_id, label, to_dwa_id, Weight::all()).unwrap();
            }
        }
        dwa
    }

    pub fn determinize_inplace(&mut self) {
        let dwa = self.determinize();
        *self = NWA::from_dwa(&dwa);
    }
}

impl Display for NWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", w)?;
            }
            for (on, targets) in &state.transitions {
                for (to, w) in targets {
                    let char_repr = format_i16_char(*on);
                    writeln!(f, "    {} -> {} (weight: {})", char_repr, to, w)?;
                }
            }
            for (to, w) in &state.epsilons {
                writeln!(f, "    ε -> {} (weight: {})", to, w)?;
            }
        }
        Ok(())
    }
}
