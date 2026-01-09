//! Non-deterministic Finite Automaton (NFA) - unweighted.
//!
//! This is a simpler structure than NWA that doesn't carry weights on transitions.
//! Internally wraps rustfst for determinize operations (to be replaced later).

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::ops::{Index, IndexMut};
use rustfst::prelude::{CoreFst, ExpandedFst, MutableFst, StateId, Tr, VectorFst, EPS_LABEL, Trs};
use rustfst::semirings::TropicalWeight;
use rustfst::algorithms::determinize::{determinize_with_config, DeterminizeConfig, DeterminizeType};
use rustfst::algorithms::rm_epsilon::rm_epsilon;
use rustfst::Semiring;

use super::dfa::{DFA, DFAState, StateID};
use crate::dwa_i32::common::Label;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NFABuildError {
    StateOutOfBounds { state: StateID },
}

impl Display for NFABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NFABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

/// A single NFA state with transitions (non-deterministic) and epsilon transitions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NFAState {
    /// Whether this state is final (accepting)
    pub is_final: bool,
    /// Transitions: label -> list of destination states
    pub transitions: BTreeMap<Label, Vec<StateID>>,
    /// Epsilon transitions: list of destination states
    pub epsilons: Vec<StateID>,
}

/// Collection of NFA states.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NFAStates(pub Vec<NFAState>);

impl Index<StateID> for NFAStates {
    type Output = NFAState;
    fn index(&self, index: StateID) -> &Self::Output { &self.0[index] }
}

impl IndexMut<StateID> for NFAStates {
    fn index_mut(&mut self, index: StateID) -> &mut Self::Output { &mut self.0[index] }
}

impl std::ops::Deref for NFAStates {
    type Target = [NFAState];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl NFAStates {
    pub fn len(&self) -> usize { self.0.len() }
    
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    
    pub fn num_transitions(&self) -> usize {
        self.0.iter()
            .map(|s| s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len())
            .sum()
    }
    
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(NFAState::default());
        id
    }
    
    pub fn add_existing_state(&mut self, state: NFAState) -> StateID {
        let id = self.0.len();
        self.0.push(state);
        id
    }
    
    pub fn add_epsilon(&mut self, from: StateID, to: StateID) {
        if from < self.len() && to < self.len() {
            self.0[from].epsilons.push(to);
        }
    }
    
    pub fn add_transition(&mut self, from: StateID, label: Label, to: StateID) -> Result<(), NFABuildError> {
        if from >= self.len() {
            return Err(NFABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.len() {
            return Err(NFABuildError::StateOutOfBounds { state: to });
        }
        self.0[from].transitions.entry(label).or_default().push(to);
        Ok(())
    }
    
    /// Appends all states from `other` into `self`, shifting their IDs.
    /// Returns the offset (ID of the first appended state).
    pub fn append(&mut self, other: &NFAStates) -> usize {
        let offset = self.len();
        self.0.reserve(other.len());
        for state in &other.0 {
            let mut new_state = state.clone();
            for targets in new_state.transitions.values_mut() {
                for to in targets { *to += offset; }
            }
            for to in &mut new_state.epsilons { *to += offset; }
            self.0.push(new_state);
        }
        offset
    }
}

/// NFA body containing start state info.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NFABody {
    pub start_states: Vec<StateID>,
}

impl NFABody {
    pub fn union(a: &NFABody, b: &NFABody) -> NFABody {
        let mut s = a.start_states.clone();
        s.extend(&b.start_states);
        NFABody { start_states: s }
    }
}

/// Non-deterministic Finite Automaton (unweighted).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NFA {
    pub states: NFAStates,
    pub body: NFABody,
}

impl NFA {
    /// Create an empty NFA with no states.
    pub fn new_empty() -> Self {
        Self { states: NFAStates::default(), body: NFABody::default() }
    }
    
    /// Create a new NFA with a single start state.
    pub fn new() -> Self {
        let mut nfa = Self::new_empty();
        let start = nfa.add_state();
        nfa.body.start_states.push(start);
        nfa
    }
    
    /// Add a state and return its ID.
    pub fn add_state(&mut self) -> StateID {
        self.states.add_state()
    }
    
    /// Add an epsilon transition.
    pub fn add_epsilon(&mut self, from: StateID, to: StateID) {
        self.states.add_epsilon(from, to);
    }
    
    /// Add a labeled transition.
    pub fn add_transition(&mut self, from: StateID, label: Label, to: StateID) -> Result<(), NFABuildError> {
        self.states.add_transition(from, label, to)
    }
    
    /// Set a state as final (accepting).
    pub fn set_final(&mut self, state: StateID) {
        if state < self.states.len() {
            self.states[state].is_final = true;
        }
    }
    
    /// Check if a state is final.
    pub fn is_final(&self, state: StateID) -> bool {
        state < self.states.len() && self.states[state].is_final
    }
    
    /// Union this NFA with another, modifying self in place.
    pub fn union_assign(this: &mut NFA, other: &NFA) {
        if other.states.is_empty() {
            return;
        }
        let offset = this.states.append(&other.states);
        for &start in &other.body.start_states {
            this.body.start_states.push(start + offset);
        }
    }
    
    /// Determinize this NFA into a DFA using rustfst.
    pub fn determinize(&self) -> DFA {
        if self.states.is_empty() {
            return DFA::new_empty();
        }
        
        // Convert to rustfst VectorFst
        let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
        
        // Add states
        let state_map: Vec<StateId> = (0..self.states.len())
            .map(|_| fst.add_state())
            .collect();
        
        // Handle multiple start states: create a super-start with epsilon to all starts
        let super_start = if self.body.start_states.len() == 1 {
            state_map[self.body.start_states[0]]
        } else {
            let super_s = fst.add_state();
            for &start in &self.body.start_states {
                fst.add_tr(super_s, Tr::new(EPS_LABEL, EPS_LABEL, TropicalWeight::one(), state_map[start])).unwrap();
            }
            super_s
        };
        fst.set_start(super_start).unwrap();
        
        // Add transitions
        for (src_id, state) in self.states.0.iter().enumerate() {
            let fst_src = state_map[src_id];
            
            // Labeled transitions
            for (&label, dsts) in &state.transitions {
                let fst_label = label_to_fst_label(label);
                for &dst_id in dsts {
                    fst.add_tr(fst_src, Tr::new(fst_label, fst_label, TropicalWeight::one(), state_map[dst_id])).unwrap();
                }
            }
            
            // Epsilon transitions
            for &dst_id in &state.epsilons {
                fst.add_tr(fst_src, Tr::new(EPS_LABEL, EPS_LABEL, TropicalWeight::one(), state_map[dst_id])).unwrap();
            }
            
            // Final state
            if state.is_final {
                fst.set_final(fst_src, TropicalWeight::one()).unwrap();
            }
        }
        
        // Remove epsilons first
        rm_epsilon(&mut fst).unwrap();
        
        // Determinize
        let config = DeterminizeConfig::default().with_det_type(DeterminizeType::DeterminizeFunctional);
        let det_fst: VectorFst<TropicalWeight> = determinize_with_config(&fst, config).unwrap();
        
        // Convert back to DFA
        let mut dfa = DFA::new_empty();
        let new_state_count = det_fst.num_states();
        
        // Create new states
        let new_state_map: Vec<StateID> = (0..new_state_count)
            .map(|_| dfa.states.add_state())
            .collect();
        
        // Set start state
        if let Some(start) = det_fst.start() {
            dfa.body.start_state = new_state_map[start as usize];
        }
        
        // Copy transitions and final states
        for fst_state_id in 0..new_state_count {
            let our_state_id = new_state_map[fst_state_id];
            
            // Transitions
            for tr in det_fst.get_trs(fst_state_id as StateId).unwrap().trs() {
                if tr.ilabel != EPS_LABEL {
                    let label = fst_label_to_label(tr.ilabel);
                    let dst = new_state_map[tr.nextstate as usize];
                    dfa.states[our_state_id].transitions.insert(label, dst);
                }
            }
            
            // Final state
            if det_fst.final_weight(fst_state_id as StateId).unwrap().is_some() {
                dfa.states[our_state_id].is_final = true;
            }
        }
        
        dfa
    }
    
    /// Determinize and minimize in one step.
    pub fn determinize_and_minimize(&self) -> DFA {
        let mut dfa = self.determinize();
        dfa.minimize();
        dfa
    }
    
    /// Get statistics about this NFA.
    pub fn stats(&self) -> String {
        let num_final = self.states.0.iter().filter(|s| s.is_final).count();
        let num_eps = self.states.0.iter().map(|s| s.epsilons.len()).sum::<usize>();
        format!("{} states, {} transitions ({} eps), {} final, {} starts", 
            self.states.len(), 
            self.states.num_transitions(),
            num_eps,
            num_final,
            self.body.start_states.len())
    }
}

impl Display for NFA {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "NFA ({} states, starts={:?})", self.states.len(), self.body.start_states)?;
        for (id, state) in self.states.0.iter().enumerate() {
            let final_marker = if state.is_final { " [FINAL]" } else { "" };
            writeln!(f, "  State {}{}", id, final_marker)?;
            for (&label, dsts) in &state.transitions {
                for dst in dsts {
                    writeln!(f, "    {:?} -> {}", label, dst)?;
                }
            }
            for dst in &state.epsilons {
                writeln!(f, "    ε -> {}", dst)?;
            }
        }
        Ok(())
    }
}

// Helper functions for label conversion
#[inline]
fn label_to_fst_label(label: Label) -> u32 {
    let result = (label as isize - Label::MIN as isize + 1) as u32;
    debug_assert_ne!(result, 0);
    result
}

#[inline]
fn fst_label_to_label(label: u32) -> Label {
    debug_assert_ne!(label, 0);
    (label as isize + Label::MIN as isize - 1) as Label
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nfa_basic() {
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        
        nfa.add_transition(nfa.body.start_states[0], 1, s1).unwrap();
        nfa.add_transition(s1, 2, s2).unwrap();
        nfa.set_final(s2);
        
        assert_eq!(nfa.states.len(), 3);
        assert!(nfa.is_final(s2));
    }
    
    #[test]
    fn test_nfa_determinize() {
        // Create an NFA with non-determinism
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        let s3 = nfa.add_state();
        
        let start = nfa.body.start_states[0];
        nfa.add_transition(start, 1, s1).unwrap();
        nfa.add_transition(start, 1, s2).unwrap(); // Non-determinism: same label, different targets
        nfa.add_transition(s1, 2, s3).unwrap();
        nfa.add_transition(s2, 3, s3).unwrap();
        nfa.set_final(s3);
        
        let dfa = nfa.determinize();
        
        // DFA should be deterministic
        for state in &dfa.states.0 {
            // Each label should have at most one transition
            assert!(state.transitions.values().all(|_| true));
        }
        
        // Should still have a final state
        assert!(dfa.states.0.iter().any(|s| s.is_final));
    }
    
    #[test]
    fn test_nfa_with_epsilons() {
        let mut nfa = NFA::new();
        let s1 = nfa.add_state();
        let s2 = nfa.add_state();
        
        let start = nfa.body.start_states[0];
        nfa.add_epsilon(start, s1);
        nfa.add_transition(s1, 1, s2).unwrap();
        nfa.set_final(s2);
        
        let dfa = nfa.determinize();
        
        // Epsilon should be eliminated
        for state in &dfa.states.0 {
            // No epsilon transitions in DFA
            assert!(state.transitions.keys().all(|_| true));
        }
        
        assert!(dfa.states.0.iter().any(|s| s.is_final));
    }
    
    #[test]
    fn test_nfa_union() {
        let mut nfa1 = NFA::new();
        let s1 = nfa1.add_state();
        nfa1.add_transition(nfa1.body.start_states[0], 1, s1).unwrap();
        nfa1.set_final(s1);
        
        let mut nfa2 = NFA::new();
        let s2 = nfa2.add_state();
        nfa2.add_transition(nfa2.body.start_states[0], 2, s2).unwrap();
        nfa2.set_final(s2);
        
        NFA::union_assign(&mut nfa1, &nfa2);
        
        assert_eq!(nfa1.body.start_states.len(), 2);
        assert_eq!(nfa1.states.len(), 4);
    }
}
