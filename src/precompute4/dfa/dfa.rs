//! Deterministic Finite Automaton (DFA) - unweighted.
//!
//! This is a simpler structure than DWA that doesn't carry weights on transitions.
//! Internally wraps rustfst for minimize operations (to be replaced later).

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::ops::{Index, IndexMut};
use rustfst::prelude::{CoreFst, ExpandedFst, MutableFst, StateId, Tr, VectorFst, EPS_LABEL, Trs};
use rustfst::semirings::TropicalWeight;
use rustfst::algorithms::minimize_with_config;
use rustfst::prelude::MinimizeConfig;
use rustfst::Semiring;

use crate::precompute4::weighted_automata::{DWA, DWAState, Weight};
use crate::precompute4::weighted_automata::common::Label;

/// State ID type
pub type StateID = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DFABuildError {
    TransitionAlreadyExists { from: StateID, on: Label },
    StateOutOfBounds { state: StateID },
}

impl Display for DFABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A single DFA state with transitions and optional final flag.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFAState {
    /// Transitions: label -> destination state
    pub transitions: BTreeMap<Label, StateID>,
    /// Whether this state is final (accepting)
    pub is_final: bool,
}

impl DFAState {
    pub fn get_transition(&self, ch: Label) -> Option<StateID> {
        self.transitions.get(&ch).copied()
    }
}

/// Collection of DFA states.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFAStates(pub Vec<DFAState>);

impl Index<usize> for DFAStates {
    type Output = DFAState;
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}

impl IndexMut<usize> for DFAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
}

impl std::ops::Deref for DFAStates {
    type Target = [DFAState];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DFAStates {
    pub fn len(&self) -> usize { self.0.len() }
    
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    
    pub fn num_transitions(&self) -> usize { 
        self.0.iter().map(|s| s.transitions.len()).sum() 
    }
    
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(DFAState::default());
        id
    }
    
    pub fn add_existing_state(&mut self, state: DFAState) -> StateID {
        let id = self.0.len();
        self.0.push(state);
        id
    }
}

/// DFA body containing start state info.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFABody {
    pub start_state: StateID,
}

/// Deterministic Finite Automaton (unweighted).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DFA {
    pub states: DFAStates,
    pub body: DFABody,
}

impl DFA {
    /// Create a new DFA with a single start state.
    pub fn new() -> Self {
        let mut states = DFAStates::default();
        let start = states.add_state();
        DFA { states, body: DFABody { start_state: start } }
    }
    
    /// Create an empty DFA with no states.
    pub fn new_empty() -> Self {
        DFA { states: DFAStates::default(), body: DFABody { start_state: 0 } }
    }
    
    /// Add a state and return its ID.
    pub fn add_state(&mut self) -> StateID {
        self.states.add_state()
    }
    
    /// Add a transition from one state to another on a label.
    pub fn add_transition(&mut self, from: StateID, label: Label, to: StateID) -> Result<(), DFABuildError> {
        if from >= self.states.len() {
            return Err(DFABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.states.len() {
            return Err(DFABuildError::StateOutOfBounds { state: to });
        }
        if self.states[from].transitions.contains_key(&label) {
            return Err(DFABuildError::TransitionAlreadyExists { from, on: label });
        }
        self.states[from].transitions.insert(label, to);
        Ok(())
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
    
    /// Minimize the DFA using rustfst.
    pub fn minimize(&mut self) {
        if self.states.len() <= 1 {
            return;
        }
        
        // Convert to rustfst VectorFst
        let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
        
        // Add states
        let state_map: Vec<StateId> = (0..self.states.len())
            .map(|_| fst.add_state())
            .collect();
        
        // Set start state
        fst.set_start(state_map[self.body.start_state]).unwrap();
        
        // Add transitions
        for (src_id, state) in self.states.0.iter().enumerate() {
            for (&label, &dst_id) in &state.transitions {
                let fst_label = label_to_fst_label(label);
                fst.add_tr(
                    state_map[src_id],
                    Tr::new(fst_label, fst_label, TropicalWeight::one(), state_map[dst_id])
                ).unwrap();
            }
            if state.is_final {
                fst.set_final(state_map[src_id], TropicalWeight::one()).unwrap();
            }
        }
        
        // Minimize
        let config = MinimizeConfig::default();
        minimize_with_config(&mut fst, config).unwrap();
        
        // Convert back
        self.states = DFAStates::default();
        let new_state_count = fst.num_states();
        
        // Create new states
        let new_state_map: Vec<StateID> = (0..new_state_count)
            .map(|_| self.states.add_state())
            .collect();
        
        // Set start state
        if let Some(start) = fst.start() {
            self.body.start_state = new_state_map[start as usize];
        }
        
        // Copy transitions and final states
        for fst_state_id in 0..new_state_count {
            let our_state_id = new_state_map[fst_state_id];
            
            // Transitions
            for tr in fst.get_trs(fst_state_id as StateId).unwrap().trs() {
                let label = fst_label_to_label(tr.ilabel);
                let dst = new_state_map[tr.nextstate as usize];
                self.states[our_state_id].transitions.insert(label, dst);
            }
            
            // Final state
            if let Some(_) = fst.final_weight(fst_state_id as StateId).unwrap() {
                self.states[our_state_id].is_final = true;
            }
        }
    }
    
    /// Convert this DFA to a DWA with Weight::all() on all transitions and finals.
    pub fn to_dwa(&self) -> DWA {
        if self.states.is_empty() {
            // Return a minimal DWA with just a start state
            return DWA::new();
        }
        
        let mut dwa = DWA::new_empty();
        
        // Add states
        let state_map: Vec<crate::precompute4::weighted_automata::StateID> = (0..self.states.len())
            .map(|_| dwa.states.add_state())
            .collect();
        
        // Set start state
        dwa.body.start_state = state_map[self.body.start_state];
        
        // Copy transitions with Weight::all()
        for (src_id, state) in self.states.0.iter().enumerate() {
            let dwa_src = state_map[src_id];
            
            for (&label, &dst_id) in &state.transitions {
                let dwa_dst = state_map[dst_id];
                dwa.states[dwa_src].transitions.insert(label, dwa_dst);
                dwa.states[dwa_src].trans_weights.insert(label, Weight::all());
            }
            
            if state.is_final {
                dwa.states[dwa_src].final_weight = Some(Weight::all());
            }
        }
        
        dwa
    }
    
    /// Get statistics about this DFA.
    pub fn stats(&self) -> String {
        let num_final = self.states.0.iter().filter(|s| s.is_final).count();
        format!("{} states, {} transitions, {} final", 
            self.states.len(), 
            self.states.num_transitions(),
            num_final)
    }
}

impl Display for DFA {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "DFA ({} states, start={})", self.states.len(), self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            let final_marker = if state.is_final { " [FINAL]" } else { "" };
            writeln!(f, "  State {}{}", id, final_marker)?;
            for (&label, &dst) in &state.transitions {
                writeln!(f, "    {:?} -> {}", label, dst)?;
            }
        }
        Ok(())
    }
}

// Helper functions for label conversion (same as weighted_automata)
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
    fn test_dfa_basic() {
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        let s2 = dfa.add_state();
        
        dfa.add_transition(dfa.body.start_state, 1, s1).unwrap();
        dfa.add_transition(s1, 2, s2).unwrap();
        dfa.set_final(s2);
        
        assert_eq!(dfa.states.len(), 3);
        assert!(dfa.is_final(s2));
        assert!(!dfa.is_final(s1));
    }
    
    #[test]
    fn test_dfa_to_dwa() {
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        dfa.add_transition(dfa.body.start_state, 1, s1).unwrap();
        dfa.set_final(s1);
        
        let dwa = dfa.to_dwa();
        assert_eq!(dwa.states.len(), 2);
        assert!(dwa.states[1].final_weight.is_some());
        assert_eq!(dwa.states[0].transitions.get(&1), Some(&1));
    }
    
    #[test]
    fn test_dfa_minimize() {
        // Create a DFA with redundant states that can be minimized
        let mut dfa = DFA::new();
        let s1 = dfa.add_state();
        let s2 = dfa.add_state();
        let s3 = dfa.add_state();
        
        // Both s2 and s3 are equivalent final states with no outgoing transitions
        dfa.add_transition(dfa.body.start_state, 1, s1).unwrap();
        dfa.add_transition(s1, 2, s2).unwrap();
        dfa.add_transition(s1, 3, s3).unwrap();
        dfa.set_final(s2);
        dfa.set_final(s3);
        
        let original_states = dfa.states.len();
        dfa.minimize();
        
        // s2 and s3 should be merged
        assert!(dfa.states.len() < original_states);
    }
}
