//! Nondeterministic Weighted Automaton (NWA).
//!
//! The NWA is the intermediate representation produced by the compiler
//! (one NWA per grammar nonterminal, or a combined super-NWA) before
//! determinization into a [`CompDwa`](super::dwa::CompDwa).
//!
//! Transition labels are `i32` (grammar symbol IDs).  Weights are
//! [`Weight`](super::weight::Weight) sets representing which
//! (token, TSID) positions survive a transition.

use std::collections::BTreeMap;

use super::weight::Weight;

/// Grammar-symbol label.
pub type Label = i32;

/// A single NWA state.
#[derive(Debug, Clone)]
pub struct NwaState {
    /// Optional final (accepting) weight.  `Some(w)` means the state is
    /// accepting and the set of surviving positions is `w`.
    pub final_weight: Option<Weight>,
    /// Label-keyed transitions: label → list of (target, weight).
    pub transitions: BTreeMap<Label, Vec<(u32, Weight)>>,
    /// ε-transitions: (target, weight).
    pub epsilons: Vec<(u32, Weight)>,
}

impl Default for NwaState {
    fn default() -> Self {
        Self {
            final_weight: None,
            transitions: BTreeMap::new(),
            epsilons: Vec::new(),
        }
    }
}

/// A Nondeterministic Weighted Automaton.
#[derive(Debug, Clone)]
pub struct Nwa {
    /// All states.
    pub states: Vec<NwaState>,
    /// Start states (subset construction begins from the ε-closure of these).
    pub start_states: Vec<u32>,
    /// Number of TSIDs (for constructing `Weight::all()`).
    pub num_tsids: u32,
    /// Maximum token ID (for `Weight::all()` / complement universe).
    pub max_token: u32,
}

impl Nwa {
    /// Create an empty NWA.
    pub fn new(num_tsids: u32, max_token: u32) -> Self {
        Self {
            states: Vec::new(),
            start_states: Vec::new(),
            num_tsids,
            max_token,
        }
    }

    /// Add a new state and return its ID.
    pub fn add_state(&mut self) -> u32 {
        let id = self.states.len() as u32;
        self.states.push(NwaState::default());
        id
    }

    /// Number of states.
    pub fn num_states(&self) -> u32 {
        self.states.len() as u32
    }

    /// Set the final weight for a state (makes it accepting).
    pub fn set_final_weight(&mut self, state: u32, weight: Weight) {
        self.states[state as usize].final_weight = Some(weight);
    }

    /// Add a labelled transition.
    pub fn add_transition(&mut self, from: u32, label: Label, to: u32, weight: Weight) {
        self.states[from as usize]
            .transitions
            .entry(label)
            .or_default()
            .push((to, weight));
    }

    /// Add an ε-transition.
    pub fn add_epsilon(&mut self, from: u32, to: u32, weight: Weight) {
        self.states[from as usize].epsilons.push((to, weight));
    }

    /// Total number of transitions (labelled + ε).
    pub fn num_transitions(&self) -> usize {
        self.states
            .iter()
            .map(|s| {
                s.transitions.values().map(|v| v.len()).sum::<usize>() + s.epsilons.len()
            })
            .sum()
    }

    /// Maximum position in the weight space.
    pub fn max_position(&self) -> u32 {
        self.max_token
            .saturating_mul(self.num_tsids.max(1))
            .saturating_add(self.num_tsids.max(1) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ds::RangeSet;

    #[test]
    fn test_nwa_basic() {
        let mut nwa = Nwa::new(2, 10);
        let s0 = nwa.add_state();
        let s1 = nwa.add_state();
        let s2 = nwa.add_state();

        let w = Weight::from_uniform_tsid_set(0, 10, &RangeSet::from_range(0, 1), 2);
        nwa.add_transition(s0, 0, s1, w.clone());
        nwa.add_epsilon(s1, s2, w.clone());
        nwa.set_final_weight(s2, w);

        assert_eq!(nwa.num_states(), 3);
        assert_eq!(nwa.num_transitions(), 2);
        assert!(nwa.states[s2 as usize].final_weight.is_some());
    }
}
