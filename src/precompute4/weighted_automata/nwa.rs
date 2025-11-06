// src/precompute4/weighted_automata/nwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_i16_char, NWAStateID, Weight};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NWABuildError {
    DefaultTransitionAlreadyExists { from: NWAStateID },
    StateOutOfBounds { state: NWAStateID },
}

impl Display for NWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NWABuildError::DefaultTransitionAlreadyExists { from } => {
                write!(f, "Default transition from state {} already exists", from)
            }
            NWABuildError::StateOutOfBounds { state } => write!(f, "State {} is out of bounds", state),
        }
    }
}

/// - Non-epsilon transitions: unique target per input symbol.
/// - Each transition carries a weight (Weight), which is intersected along the path; final states
///   carry a final weight that is intersected at acceptance; multiple alternative paths union their weights.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWAState {
    pub final_weight: Option<Weight>,
    /// Non-epsilon transitions: unique target per input symbol.
    /// Map: label -> (target, weight)
    pub transitions: BTreeMap<i16, (NWAStateID, Weight)>,
    /// Epsilon transitions: list of (target, weight).
    pub epsilons: Vec<(NWAStateID, Weight)>,
    /// Default transition: used when a labeled transition for a symbol is absent.
    /// Semantics: for any symbol not present as a labeled transition, this edge applies.
    pub default: Option<(NWAStateID, Weight)>,
}

impl NWAState {
    pub fn get_transition(&self, on: i16) -> Option<&(NWAStateID, Weight)> {
        self.transitions.get(&on).or_else(|| {
            if let Some(result) = &self.default {
                Some(result)
            } else {
                None
            }
        })
    }
}

#[derive(Clone, Debug, Default)]
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

    /// Add a labeled transition; if an existing transition on the same label exists,
    /// we merge weights if the target is the same, otherwise we assert (as per restricted NWA).
    pub fn add_transition(&mut self, from: NWAStateID, on: i16, to: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        if from >= self.len() {
            return Err(NWABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.len() {
            return Err(NWABuildError::StateOutOfBounds { state: to });
        }
        if let Some((old_to, old_w)) = self.0[from].transitions.get_mut(&on) {
            assert_eq!(*old_to, to, "NWA restricted: only one target per (state, symbol)");
            *old_w |= &w;
        } else {
            self.0[from].transitions.insert(on, (to, w));
        }
        Ok(())
    }

    pub fn add_default_transition(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        if from >= self.len() {
            return Err(NWABuildError::StateOutOfBounds { state: from });
        }
        if to >= self.len() {
            return Err(NWABuildError::StateOutOfBounds { state: to });
        }
        if self.0[from].default.is_some() {
            return Err(NWABuildError::DefaultTransitionAlreadyExists { from });
        }
        self.0[from].default = Some((to, w));
        Ok(())
    }

    /// Deep-copy a subgraph starting at 'start_id' from another NWAStates arena into self.
    /// Returns (new_start_id, remap_old_to_new)
    pub fn copy_subgraph_from(&mut self, other: &NWAStates, start_id: NWAStateID) -> (NWAStateID, HashMap<NWAStateID, NWAStateID>) {
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
            // Epsilon edges
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
            // Labeled edges
            let trans = other.0[old].transitions.clone();
            self.0[new].transitions.clear();
            for (lbl, (to_old, w)) in trans {
                let to_new = *remap.entry(to_old).or_insert_with(|| {
                    let n = self.add_state();
                    self.0[n] = other.0[to_old].clone();
                    q.push_back((to_old, n));
                    n
                });
                self.0[new].transitions.insert(lbl, (to_new, w.clone()));
            }
            // Default edge
            let def_old = other.0[old].default.clone();
            self.0[new].default = None;
            if let Some((to_old, w)) = def_old {
                let to_new = *remap.entry(to_old).or_insert_with(|| {
                    let n = self.add_state();
                    self.0[n] = other.0[to_old].clone();
                    q.push_back((to_old, n));
                    n
                });
                self.0[new].default = Some((to_new, w));
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
            if let Some((to, w)) = &state.default {
                writeln!(f, "    * -> {} (weight: {})", to, w)?;
            }
            for (on, (to, w)) in &state.transitions {
                let char_repr = format_i16_char(*on);
                writeln!(f, "    {} -> {} (weight: {})", char_repr, to, w)?;
            }
            for (to, w) in &state.epsilons {
                writeln!(f, "    ε -> {} (weight: {})", to, w)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NWABody {
    pub start_state: NWAStateID,
}

impl Display for NWABody {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "NWABody (start: {})", self.start_state)
    }
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
        Self { states, body: NWABody { start_state: start } }
    }

    pub fn add_transition(&mut self, from: NWAStateID, on: i16, to: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        self.states.add_transition(from, on, to, w)
    }

    pub fn add_default_transition(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) -> Result<(), NWABuildError> {
        self.states.add_default_transition(from, to, w)
    }

    pub fn add_epsilon(&mut self, from: NWAStateID, to: NWAStateID, w: Weight) {
        self.states.add_epsilon(from, to, w);
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
            if let Some((to, w)) = &state.default {
                writeln!(f, "    * -> {} (weight: {})", to, w)?;
            }
            for (on, (to, w)) in &state.transitions {
                let char_repr = format_i16_char(*on);
                writeln!(f, "    {} -> {} (weight: {})", char_repr, to, w)?;
            }
            for (to, w) in &state.epsilons {
                writeln!(f, "    ε -> {} (weight: {})", to, w)?;
            }
        }
        Ok(())
    }
}
