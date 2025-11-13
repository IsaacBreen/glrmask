// src/precompute4/weighted_automata/dwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_pos_code, StateID, Weight, DEFAULT_TRANSITION_SYMBOL};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{self, Display, Formatter};
use std::ops::{Deref, Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DWABuildError {
    TransitionAlreadyExists { from: StateID, on: i16 },
    StateOutOfBounds { state: StateID },
}

impl Display for DWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DWABuildError::TransitionAlreadyExists { from, on } => {
                write!(f, "Transition from state {} on code {} already exists", from, on)
            }
            DWABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAState {
    pub transitions: BTreeMap<i16, StateID>,
    pub final_weight: Option<Weight>,
    pub trans_weights: BTreeMap<i16, Weight>,
    /// Optional state-entry weight (intersected upon entering the state).
    pub state_weight: Option<Weight>,
}

impl DWAState {
    pub fn get_transition(&self, ch: i16) -> Option<(StateID, &Weight)> {
        // First, try to find an explicit transition for the character.
        if let Some(to) = self.transitions.get(&ch) {
            if let Some(w) = self.trans_weights.get(&ch) {
                return Some((*to, w));
            }
        }
        // If not found, fall back to the default transition.
        if let Some(to) = self.transitions.get(&DEFAULT_TRANSITION_SYMBOL) {
            if let Some(w) = self.trans_weights.get(&DEFAULT_TRANSITION_SYMBOL) {
                return Some((*to, w));
            }
        }
        None
    }

    pub fn get_weight(&self, ch: i16) -> Option<&Weight> {
        self.trans_weights.get(&ch).or_else(|| self.trans_weights.get(&DEFAULT_TRANSITION_SYMBOL))
    }

    /// Intersects all weights in this state with the given weight.
    pub fn apply_weight(&mut self, weight: &Weight) {
        if let Some(sw) = &mut self.state_weight {
            *sw &= weight;
            if sw.is_empty() {
                self.state_weight = None;
            }
        }

        if let Some(fw) = &mut self.final_weight {
            *fw &= weight;
            if fw.is_empty() {
                self.final_weight = None;
            }
        }

        for w in self.trans_weights.values_mut() {
            *w &= weight;
        }
    }

    /// Subtracts a weight from all weights in this state.
    pub fn exclude_weight(&mut self, weight: &Weight) {
        if let Some(sw) = &mut self.state_weight {
            *sw -= weight;
            if sw.is_empty() {
                self.state_weight = None;
            }
        }

        if let Some(fw) = &mut self.final_weight {
            *fw -= weight;
            if fw.is_empty() {
                self.final_weight = None;
            }
        }

        for w in self.trans_weights.values_mut() {
            *w -= weight;
        }
    }

    /// Iterator over all outgoing edges:
    /// - Edges appear as (label, target, weight)
    #[inline]
    pub fn iter_edges(&self) -> impl Iterator<Item = (i16, StateID, &Weight)> {
        self.transitions
            .iter()
            .filter_map(move |(ch, to)| self.trans_weights.get(ch).map(|w| (*ch, *to, w)))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAStates(pub Vec<DWAState>);

impl Index<usize> for DWAStates {
    type Output = DWAState;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl IndexMut<usize> for DWAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl Deref for DWAStates {
    type Target = [DWAState];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DWAStates {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len();
        self.0.push(DWAState::default());
        id
    }

    /// Adds a pre-existing DWAState to the collection and returns its new ID.
    pub fn add_existing_state(&mut self, state: DWAState) -> StateID {
        let id = self.0.len();
        self.0.push(state);
        id
    }

    pub fn copy_state(&mut self, state_id: StateID) -> StateID {
        assert!(state_id < self.len(), "state_id out of bounds");
        let state = self[state_id].clone();
        self.add_existing_state(state)
    }

    pub fn apply_weight(&mut self, state_id: StateID, weight: &Weight) {
        assert!(state_id < self.len(), "state_id out of bounds");
        self[state_id].apply_weight(weight);
    }

    pub fn copy_subgraph(&mut self, start_id: StateID) -> (StateID, HashMap<StateID, StateID>) {
        let mut remap = HashMap::new();
        let mut q = VecDeque::new();

        if start_id >= self.len() {
            let new_start_id = self.add_state();
            return (new_start_id, remap);
        }

        let new_start_id = self.add_existing_state(self[start_id].clone());
        remap.insert(start_id, new_start_id);
        q.push_back((start_id, new_start_id));

        while let Some((old_id, new_id)) = q.pop_front() {
            let old_state_clone = self[old_id].clone();

            // Remap transitions
            let mut new_transitions = BTreeMap::new();
            for (ch, &old_target) in &old_state_clone.transitions {
                let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                    let new_id = self.add_existing_state(self[old_target].clone());
                    q.push_back((old_target, new_id));
                    new_id
                });
                new_transitions.insert(*ch, new_target_id);
            }
            self[new_id].transitions = new_transitions;
        }
        (new_start_id, remap)
    }

    pub fn copy_subgraph_from(&mut self, other_states: &DWAStates, start_id: StateID) -> (StateID, HashMap<StateID, StateID>) {
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

            self[new_id].transitions = old_state_clone.transitions.iter().map(|(ch, &old_target)| {
                let new_target_id = *remap.entry(old_target).or_insert_with(|| {
                    let new_id = self.add_existing_state(other_states[old_target].clone());
                    q.push_back((old_target, new_id));
                    new_id
                });
                (*ch, new_target_id)
            }).collect();
            self[new_id].trans_weights = old_state_clone.trans_weights;
        }
        (new_start_id, remap)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWABody {
    pub start_state: StateID,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWA {
    pub states: DWAStates,
    pub body: DWABody,
}

impl DWA {
    pub fn new() -> Self {
        let mut states = DWAStates::default();
        let start = states.add_state();
        DWA { states, body: DWABody { start_state: start } }
    }

    pub fn add_state(&mut self) -> StateID {
        self.states.add_state()
    }

    pub fn set_state_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state });
        }
        self.states[state].state_weight = Some(weight);
        Ok(())
    }

    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state });
        }
        self.states[state].final_weight = Some(weight);
        Ok(())
    }

    pub fn add_transition(
        &mut self,
        from: StateID,
        on: i16,
        to: StateID,
        weight: Weight,
    ) -> Result<(), DWABuildError> {
        if from >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.states.len() {
            return Err(DWABuildError::StateOutOfBounds { state: to });
        }
        let from_state = &mut self.states[from];
        if from_state.transitions.contains_key(&on) {
            return Err(DWABuildError::TransitionAlreadyExists { from, on });
        }
        from_state.transitions.insert(on, to);
        from_state.trans_weights.insert(on, weight);
        Ok(())
    }

    pub fn set_default_transition(
        &mut self,
        from: StateID,
        to: StateID,
        weight: Weight,
    ) -> Result<(), DWABuildError> {
        self.add_transition(from, DEFAULT_TRANSITION_SYMBOL, to, weight)
    }
}

#[derive(Debug, Clone)]
pub struct DWAStats {
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_final_states: usize,
    pub num_default_transitions: usize,
    pub avg_exceptions_per_state: f64,
}

impl Display for DWAStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "DWA Stats:")?;
        writeln!(f, "  - States: {}", self.num_states)?;
        writeln!(f, "  - Transitions: {}", self.num_transitions)?;
        writeln!(f, "  - Final States: {}", self.num_final_states)?;
        writeln!(f, "  - States with Default: {}", self.num_default_transitions)?;
        writeln!(f, "  - Avg Exceptions/State: {:.2}", self.avg_exceptions_per_state)
    }
}

impl DWA {
    pub fn stats(&self) -> DWAStats {
        let num_states = self.states.len();
        if num_states == 0 {
            return DWAStats {
                num_states: 0,
                num_transitions: 0,
                num_final_states: 0,
                num_default_transitions: 0,
                avg_exceptions_per_state: 0.0,
            };
        }

        let mut num_exceptions = 0;
        let mut num_final_states = 0;
        let mut num_default_transitions = 0;

        for state in &self.states.0 {
            num_exceptions += state.transitions.len();
            if state.final_weight.is_some() {
                num_final_states += 1;
            }
            if state.transitions.contains_key(&DEFAULT_TRANSITION_SYMBOL) {
                num_default_transitions += 1;
            }
        }

        let num_transitions = num_exceptions;
        let num_exceptions_only = num_exceptions - num_default_transitions;
        let avg_exceptions_per_state = num_exceptions_only as f64 / num_states as f64;

        DWAStats {
            num_states,
            num_transitions,
            num_final_states,
            num_default_transitions,
            avg_exceptions_per_state,
        }
    }
}

impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(sw) = &state.state_weight {
                writeln!(f, "    state_weight: {}", sw)?;
            }
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", w)?;
            }
            for (on, to) in &state.transitions {
                if *on == DEFAULT_TRANSITION_SYMBOL {
                    if let Some(w) = state.trans_weights.get(on) {
                        writeln!(f, "    * -> {} (trans_weight: {})", to, w)?;
                    } else {
                        writeln!(f, "    * -> {}", to)?;
                    }
                } else {
                    let char_repr = if *on >= 0 {
                        format_pos_code(*on)
                    } else {
                        let decoded_id = on.wrapping_sub(i16::MIN);
                        format!("neg({})", decoded_id)
                    };
                    if let Some(w) = state.trans_weights.get(on) {
                        writeln!(f, "    {} -> {} (trans_weight: {})", char_repr, to, w)?;
                    } else {
                        writeln!(f, "    {} -> {}", char_repr, to)?;
                    }
                }
            }
        }
        Ok(())
    }
}
