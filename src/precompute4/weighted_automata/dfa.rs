// src/precompute4/weighted_automata/dfa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{I16Map, StateID};
use std::collections::{HashMap, VecDeque};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DFABuildError {
    TransitionAlreadyExists { from: StateID, on: i16 },
    DefaultTransitionAlreadyExists { from: StateID },
    StateOutOfBounds { state: StateID },
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFAState {
    pub transitions: I16Map<StateID>,
    pub is_final: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFAStates(pub Vec<DFAState>);

impl Index<usize> for DFAStates {
    type Output = DFAState;
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}
impl IndexMut<usize> for DFAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
}

impl DFAStates {
    pub fn len(&self) -> usize { self.0.len() }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(DFAState::default());
        id
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFABody {
    pub start_state: StateID,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFA {
    pub states: DFAStates,
    pub body: DFABody,
}

impl DFA {
    pub fn new() -> Self {
        let mut states = DFAStates::default();
        let start = states.add_state();
        DFA { states, body: DFABody { start_state: start } }
    }

    pub fn simplify(&mut self) {
        // For now, just prune unreachable. A full minimization is more complex.
        self.prune_unreachable();
    }

    fn prune_unreachable(&mut self) {
        let n = self.states.0.len();
        if n == 0 { return; }

        let mut reachable = vec![false; n];
        let mut q = VecDeque::new();
        if self.body.start_state < n {
            reachable[self.body.start_state] = true;
            q.push_back(self.body.start_state);
        }

        while let Some(u) = q.pop_front() {
            let state = &self.states[u];
            if let Some(v) = state.transitions.default {
                if v < n && !reachable[v] {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }
            for &v in state.transitions.exceptions.values() {
                if v < n && !reachable[v] {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }
        }

        let remap: HashMap<_, _> = reachable.iter().enumerate().filter(|(_, &r)| r).map(|(i, _)| i).enumerate().map(|(new, old)| (old, new)).collect();
        if remap.len() == n { return; }

        let mut new_states = DFAStates(vec![DFAState::default(); remap.len()]);
        for (old_idx, &is_reachable) in reachable.iter().enumerate() {
            if is_reachable {
                let new_idx = remap[&old_idx];
                let mut new_state = self.states[old_idx].clone();
                if let Some(target) = new_state.transitions.default.as_mut() { *target = remap[target]; }
                for target in new_state.transitions.exceptions.values_mut() { *target = remap[target]; }
                new_states.0[new_idx] = new_state;
            }
        }

        self.states = new_states;
        self.body.start_state = remap[&self.body.start_state];
    }
}
