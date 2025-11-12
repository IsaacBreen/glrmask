// src/precompute4/weighted_automata/nfa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_i16_char, NWAStateID as NFAStateID};
use super::dfa::{DFABody, DFAState, DFAStates, DFA};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NFABuildError {
    StateOutOfBounds { state: NFAStateID },
}

impl Display for NFABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NFABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NFADefaultTransition {
    pub target: NFAStateID,
    pub exceptions: BTreeSet<i16>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NFAState {
    pub is_final: bool,
    pub transitions: BTreeMap<i16, Vec<NFAStateID>>,
    pub epsilons: Vec<NFAStateID>,
    pub default: Vec<NFADefaultTransition>,
}

#[derive(Clone, Debug, Default)]
pub struct NFAStates(pub Vec<NFAState>);

impl NFAStates {
    pub fn len(&self) -> usize { self.0.len() }

    pub fn add_state(&mut self) -> NFAStateID {
        let id = self.0.len();
        self.0.push(NFAState::default());
        id
    }

    pub fn add_epsilon(&mut self, from: NFAStateID, to: NFAStateID) {
        assert!(from < self.len() && to < self.len(), "add_epsilon: state id out of bounds");
        self.0[from].epsilons.push(to);
    }

    pub fn add_transition(&mut self, from: NFAStateID, on: i16, to: NFAStateID) -> Result<(), NFABuildError> {
        if from >= self.len() { return Err(NFABuildError::StateOutOfBounds { state: from }); }
        if to >= self.len() { return Err(NFABuildError::StateOutOfBounds { state: to }); }
        self.0[from].transitions.entry(on).or_default().push(to);
        Ok(())
    }

    pub fn add_default_transition(&mut self, from: NFAStateID, to: NFAStateID, exceptions: BTreeSet<i16>) -> Result<(), NFABuildError> {
        if from >= self.len() { return Err(NFABuildError::StateOutOfBounds { state: from }); }
        if to >= self.len() { return Err(NFABuildError::StateOutOfBounds { state: to }); }
        self.0[from].default.push(NFADefaultTransition { target: to, exceptions });
        Ok(())
    }
}

impl Index<NFAStateID> for NFAStates {
    type Output = NFAState;
    fn index(&self, index: NFAStateID) -> &Self::Output { &self.0[index] }
}
impl IndexMut<NFAStateID> for NFAStates {
    fn index_mut(&mut self, index: NFAStateID) -> &mut Self::Output { &mut self.0[index] }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NFABody {
    pub start_state: NFAStateID,
}

#[derive(Clone, Debug, Default)]
pub struct NFA {
    pub states: NFAStates,
    pub body: NFABody,
}

impl NFA {
    pub fn new() -> Self {
        let mut states = NFAStates::default();
        let start = states.add_state();
        Self { states, body: NFABody { start_state: start } }
    }

    pub fn add_transition(&mut self, from: NFAStateID, on: i16, to: NFAStateID) -> Result<(), NFABuildError> {
        self.states.add_transition(from, on, to)
    }

    pub fn add_default_transition(&mut self, from: NFAStateID, to: NFAStateID, exceptions: BTreeSet<i16>) -> Result<(), NFABuildError> {
        self.states.add_default_transition(from, to, exceptions)
    }

    pub fn add_epsilon(&mut self, from: NFAStateID, to: NFAStateID) {
        self.states.add_epsilon(from, to);
    }

    pub fn determinize_to_dfa(&self) -> DFA {
        // Simplified subset construction for unweighted NFA
        let mut dfa_states = DFAStates::default();
        let mut subset_map: HashMap<BTreeSet<NFAStateID>, usize> = HashMap::new();
        let mut worklist = VecDeque::new();

        let mut alphabet = BTreeSet::new();
        for state in &self.states.0 {
            for &symbol in state.transitions.keys() { alphabet.insert(symbol); }
            for def in &state.default { for &symbol in &def.exceptions { alphabet.insert(symbol); } }
        }
        let alphabet: Vec<i16> = alphabet.into_iter().collect();

        let start_closure = self.epsilon_closure(&[self.body.start_state].iter().cloned().collect());
        let start_dfa_state = dfa_states.add_state();
        subset_map.insert(start_closure.clone(), start_dfa_state);
        worklist.push_back(start_closure.clone());

        dfa_states[start_dfa_state].is_final = self.is_final_subset(&start_closure);

        while let Some(current_subset) = worklist.pop_front() {
            let current_dfa_state = *subset_map.get(&current_subset).unwrap();

            let mut transitions = BTreeMap::new();
            for &symbol in &alphabet {
                let mut next_states = BTreeSet::new();
                for &nfa_state in &current_subset {
                    if let Some(targets) = self.states[nfa_state].transitions.get(&symbol) { next_states.extend(targets); }
                    else { for def in &self.states[nfa_state].default { if !def.exceptions.contains(&symbol) { next_states.insert(def.target); } } }
                }
                if !next_states.is_empty() { transitions.insert(symbol, next_states); }
            }

            let mut default_next_states = BTreeSet::new();
            for &nfa_state in &current_subset { for def in &self.states[nfa_state].default { if def.exceptions.is_empty() { default_next_states.insert(def.target); } } }

            for (symbol, next_states) in transitions {
                let next_closure = self.epsilon_closure(&next_states);
                if next_closure.is_empty() { continue; }
                let next_dfa_state = *subset_map.entry(next_closure.clone()).or_insert_with(|| { let new_state = dfa_states.add_state(); dfa_states[new_state].is_final = self.is_final_subset(&next_closure); worklist.push_back(next_closure); new_state });
                dfa_states[current_dfa_state].transitions.exceptions.insert(symbol, next_dfa_state);
            }

            if !default_next_states.is_empty() {
                let default_closure = self.epsilon_closure(&default_next_states);
                if !default_closure.is_empty() {
                    let default_dfa_state = *subset_map.entry(default_closure.clone()).or_insert_with(|| { let new_state = dfa_states.add_state(); dfa_states[new_state].is_final = self.is_final_subset(&default_closure); worklist.push_back(default_closure); new_state });
                    dfa_states[current_dfa_state].transitions.default = Some(default_dfa_state);
                }
            }
        }

        DFA { states: dfa_states, body: DFABody { start_state: start_dfa_state } }
    }

    fn epsilon_closure(&self, states: &BTreeSet<NFAStateID>) -> BTreeSet<NFAStateID> {
        let mut closure = states.clone();
        let mut worklist: Vec<_> = states.iter().cloned().collect();
        while let Some(state) = worklist.pop() {
            for &epsilon_target in &self.states[state].epsilons {
                if closure.insert(epsilon_target) { worklist.push(epsilon_target); }
            }
        }
        closure
    }

    fn is_final_subset(&self, subset: &BTreeSet<NFAStateID>) -> bool {
        subset.iter().any(|&s| self.states[s].is_final)
    }
}
