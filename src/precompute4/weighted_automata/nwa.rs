// src/precompute4/weighted_automata/nwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_i16_char, Label, NWAStateID, Weight, BENCHMARK_DEBUG};
use super::dwa::DWA;
use crate::precompute4::weighted_automata::{DWAState, StateID};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::ops::{Index, IndexMut};

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
    pub transitions: BTreeMap<Label, Vec<(NWAStateID, Weight)>>,
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

    pub fn add_existing_state(&mut self, state: NWAState) -> NWAStateID {
        let id = self.0.len(); self.0.push(state); id
    }

    pub fn add_epsilon(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) {
        if from < self.len() && to < self.len() {
            self.0[from].epsilons.push((to, w));
        }
    }

    pub fn add_transition(&mut self, from: NWAStateID, on: Label, to: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        if from >= self.len() { return Err(NWABuildError::StateOutOfBounds { state: from }); }
        if to >= self.len() { return Err(NWABuildError::StateOutOfBounds { state: to }); }
        self.0[from].transitions.entry(on).or_default().push((to, w));
        Ok(())
    }

    /// Appends all states from `other` into `self`, shifting their IDs.
    /// Returns the offset (ID of the first appended state).
    pub fn append(&mut self, other: &NWAStates) -> usize {
        let offset = self.len();
        self.0.reserve(other.len());
        for state in &other.0 {
            let mut new_state = state.clone();
            for targets in new_state.transitions.values_mut() {
                for (to, _) in targets { *to += offset; }
            }
            for (to, _) in &mut new_state.epsilons { *to += offset; }
            self.0.push(new_state);
        }
        offset
    }

    /// Concatenates `left` NWA onto `right_body` *in place*.
    /// `left` is appended to `self`. Then, for every final state in the appended `left`,
    /// epsilon transitions are added to all of `right_body`'s start states.
    /// Returns a new `NWABody` representing the starts of the concatenated structure (i.e., left's starts).
    pub fn concatenate_in_place(&mut self, left: &NWA, right_body: &NWABody) -> NWABody {
        let offset = self.append(&left.states);
        
        // Connect left's finals to right's starts
        for i in 0..left.states.len() {
            let abs_id = i + offset;
            if let Some(fw) = self.0[abs_id].final_weight.take() {
                if !fw.is_empty() {
                    for &r_start in &right_body.start_states {
                        self.add_epsilon(abs_id, r_start, fw.clone());
                    }
                }
            }
        }

        NWABody {
            start_states: left.body.start_states.iter().map(|s| s + offset).collect()
        }
    }

    pub fn union_in_place(&mut self, other: &NWA, existing_body: &NWABody) -> NWABody {
        let offset = self.append(&other.states);
        let mut new_starts = existing_body.start_states.clone();
        new_starts.extend(other.body.start_states.iter().map(|s| s + offset));
        NWABody {
            start_states: new_starts
        }
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
            if let Some(w) = &state.final_weight { writeln!(f, "    final_weight: {}", w)?; }
            for (on, targets) in &state.transitions {
                for (to, w) in targets {
                    writeln!(f, "    {} -> {} (weight: {})", format_i16_char(*on), to, w)?;
                }
            }
            for (to, w) in &state.epsilons { writeln!(f, "    ε -> {} (weight: {})", to, w)?; }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NWABody {
    pub start_states: Vec<NWAStateID>,
}

impl NWABody {
    pub fn union(a: &NWABody, b: &NWABody) -> NWABody {
        let mut s = a.start_states.clone();
        s.extend(&b.start_states);
        NWABody { start_states: s }
    }
}

impl Display for NWABody {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { 
        write!(f, "NWABody (starts: {:?})", self.start_states) 
    }
}

#[derive(Debug, Clone)]
pub struct NWAStats {
    pub num_states: usize,
    pub num_final_states: usize,
    pub total_epsilon: usize,
    pub total_labeled: usize,
}

impl Display for NWAStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "States: {}, Finals: {}, Eps: {}, Labeled: {}", 
            self.num_states, self.num_final_states, self.total_epsilon, self.total_labeled)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NWA {
    pub states: NWAStates,
    pub body: NWABody,
}

impl NWA {
    pub fn new_empty() -> Self {
        Self { states: NWAStates::default(), body: NWABody::default() }
    }
    pub fn new() -> Self {
        let mut nwa = Self::new_empty();
        let start = nwa.add_state();
        nwa.body.start_states.push(start);
        nwa
    }
    pub fn add_state(&mut self) -> NWAStateID { self.states.add_state() }
    pub fn add_epsilon(&mut self, u: NWAStateID, v: NWAStateID, w: Weight) { self.states.add_epsilon(u, v, w); }
    pub fn add_transition(&mut self, u: NWAStateID, l: Label, v: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        self.states.add_transition(u, l, v, w)
    }

    pub fn reverse(&self) -> NWA {
        let mut rev = NWA::new_empty();
        // Pre-allocate states
        for _ in 0..self.states.len() { rev.add_state(); }

        // Create a super-start for the reversed NWA that connects to all old final states
        let super_start = rev.add_state();
        rev.body.start_states = vec![super_start];

        for (u, state) in self.states.0.iter().enumerate() {
            // Reverse transitions: u -> v becomes v -> u
            for (lbl, targets) in &state.transitions {
                for (v, w) in targets { rev.add_transition(*v, *lbl, u, w.clone()).unwrap(); }
            }
            // Reverse epsilons
            for (v, w) in &state.epsilons { rev.add_epsilon(*v, u, w.clone()); }
            
            // Old finals become reachable from new super-start via epsilon
            if let Some(fw) = &state.final_weight {
                if !fw.is_empty() {
                    rev.add_epsilon(super_start, u, fw.clone());
                }
            }
        }

        // Old start states become final in the reversed NWA with Weight::all()
        for &s in &self.body.start_states {
            if s < rev.states.len() {
                let old = rev.states[s].final_weight.clone().unwrap_or_else(Weight::zeros);
                rev.states[s].final_weight = Some(old | Weight::all());
            }
        }
        
        rev
    }

    pub fn concatenate(left: &NWA, right: &NWA) -> NWA {
        let mut res = NWA::new_empty();
        let _ = res.states.append(&right.states); // Right is at offset 0
        // Construct a body for the right segment
        let right_body = right.body.clone(); // indices are 0-based, valid
        
        // Concatenate left into place (appends left states and links finals -> right starts)
        res.body = res.states.concatenate_in_place(left, &right_body);
        res
    }

    pub fn union(a: &NWA, b: &NWA) -> NWA {
        let mut res = NWA::new_empty();
        let off_a = res.states.append(&a.states);
        let off_b = res.states.append(&b.states);
        
        let mut starts = Vec::new();
        starts.extend(a.body.start_states.iter().map(|s| s + off_a));
        starts.extend(b.body.start_states.iter().map(|s| s + off_b));
        res.body.start_states = starts;
        res
    }

    pub fn union_assign(&mut self, other: &NWA) {
        let offset = self.states.append(&other.states);
        self.body.start_states.extend(other.body.start_states.iter().map(|s| s + offset));
    }

    pub fn concatenate_assign(&mut self, other: &NWA) {
        let offset = self.states.append(&other.states);
        let right_starts: Vec<_> = other.body.start_states.iter().map(|s| s + offset).collect();
        
        for i in 0..offset {
            if let Some(fw) = self.states[i].final_weight.take() {
                if !fw.is_empty() {
                    for &r_start in &right_starts {
                        self.add_epsilon(i, r_start, fw.clone());
                    }
                }
            }
        }
    }

    pub fn from_dwa(dwa: &DWA) -> Self {
        let mut nwa = NWA::new_empty();
        for _ in 0..dwa.states.len() { nwa.add_state(); }
        nwa.body.start_states = vec![dwa.body.start_state];

        for (i, st) in dwa.states.0.iter().enumerate() {
            nwa.states[i].final_weight = st.final_weight.clone();
            for (lbl, to) in &st.transitions {
                let w = st.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
                nwa.add_transition(i, *lbl, *to, w).unwrap();
            }
        }
        nwa
    }

    pub fn stats(&self) -> NWAStats {
        let mut st = NWAStats { num_states: self.states.len(), num_final_states: 0, total_epsilon: 0, total_labeled: 0 };
        for s in &self.states.0 {
            if s.final_weight.is_some() { st.num_final_states += 1; }
            st.total_epsilon += s.epsilons.len();
            st.total_labeled += s.transitions.values().map(|v| v.len()).sum::<usize>();
        }
        st
    }
}

impl Display for NWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NWA {}", self.body)?;
        write!(f, "{}", self.states)
    }
}
