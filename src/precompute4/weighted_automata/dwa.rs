// src/precompute4/weighted_automata/dwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{format_pos_code, Label, StateID, Weight};
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::ops::{Deref, Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DWABuildError {
    TransitionAlreadyExists { from: StateID, on: Label },
    StateOutOfBounds { state: StateID },
}

impl Display for DWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAState {
    pub transitions: BTreeMap<Label, StateID>,
    pub final_weight: Option<Weight>,
    pub trans_weights: BTreeMap<Label, Weight>,
    pub state_weight: Option<Weight>,
}

impl DWAState {
    pub fn get_transition(&self, ch: Label) -> Option<(StateID, &Weight)> {
        self.transitions.get(&ch).and_then(|to| self.trans_weights.get(&ch).map(|w| (*to, w)))
    }
    
    pub fn apply_weight(&mut self, weight: &Weight) {
        if let Some(sw) = &mut self.state_weight { *sw &= weight; if sw.is_empty() { self.state_weight = None; } }
        if let Some(fw) = &mut self.final_weight { *fw &= weight; if fw.is_empty() { self.final_weight = None; } }
        for w in self.trans_weights.values_mut() { *w &= weight; }
    }

    pub fn clip_weights(&mut self, max: usize) {
        if let Some(sw) = &mut self.state_weight { sw.clip_max(max); if sw.is_empty() { self.state_weight = None; } }
        if let Some(fw) = &mut self.final_weight { fw.clip_max(max); if fw.is_empty() { self.final_weight = None; } }
        for w in self.trans_weights.values_mut() { w.clip_max(max); }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAStates(pub Vec<DWAState>);

impl Index<usize> for DWAStates {
    type Output = DWAState;
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}
impl IndexMut<usize> for DWAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
}
impl Deref for DWAStates {
    type Target = [DWAState];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DWAStates {
    pub fn len(&self) -> usize { self.0.len() }
    pub fn num_transitions(&self) -> usize { self.0.iter().map(|s| s.transitions.len()).sum() }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len(); self.0.push(DWAState::default()); id
    }
    pub fn add_existing_state(&mut self, state: DWAState) -> StateID {
        let id = self.0.len(); self.0.push(state); id
    }
    pub fn copy_state(&mut self, state_id: StateID) -> StateID {
        let state = self[state_id].clone(); self.add_existing_state(state)
    }
    pub fn apply_weight_to_state(&mut self, state_id: StateID, weight: &Weight) {
        self[state_id].apply_weight(weight);
    }
    pub fn apply_weight_to_all_states(&mut self, weight: &Weight) {
        for state in self.0.iter_mut() { state.apply_weight(weight); }
    }
    pub fn clip_weights(&mut self, max: usize) {
        for state in self.0.iter_mut() { state.clip_weights(max); }
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
    pub fn add_state(&mut self) -> StateID { self.states.add_state() }
    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() { return Err(DWABuildError::StateOutOfBounds { state }); }
        self.states[state].final_weight = Some(weight); Ok(())
    }
    pub fn add_transition(&mut self, from: StateID, on: Label, to: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if from >= self.states.len() { return Err(DWABuildError::StateOutOfBounds { state: from }); }
        if to >= self.states.len() { return Err(DWABuildError::StateOutOfBounds { state: to }); }
        if self.states[from].transitions.contains_key(&on) { return Err(DWABuildError::TransitionAlreadyExists { from, on }); }
        self.states[from].transitions.insert(on, to);
        self.states[from].trans_weights.insert(on, weight); Ok(())
    }
    
    pub fn eval_word_weight(&self, word: &[Label]) -> Weight {
        if self.states.0.is_empty() { return Weight::zeros(); }
        let mut s = self.body.start_state;
        let mut acc = Weight::all();

        if s < self.states.len() {
             if let Some(sw) = &self.states[s].state_weight { acc &= sw; if acc.is_empty() { return Weight::zeros(); } }
        } else { return Weight::zeros(); }

        for &ch in word {
            if s >= self.states.len() { return Weight::zeros(); }
            if let Some((t, w)) = self.states[s].get_transition(ch) {
                acc &= w; if acc.is_empty() { return Weight::zeros(); }
                s = t;
                if let Some(sw) = &self.states[s].state_weight { acc &= sw; if acc.is_empty() { return Weight::zeros(); } }
            } else { return Weight::zeros(); }
        }
        if s >= self.states.len() { return Weight::zeros(); }
        match &self.states[s].final_weight {
            Some(fw) => { let res = &acc & fw; if res.is_empty() { Weight::zeros() } else { res } }
            None => Weight::zeros(),
        }
    }

    pub fn apply_weight_inplace(&mut self, weight: &Weight) {
        if self.body.start_state < self.states.len() {
            let s = &mut self.states[self.body.start_state];
            if let Some(sw) = &mut s.state_weight { *sw &= weight; } else { s.state_weight = Some(weight.clone()); }
            s.apply_weight(weight);
        }
    }

    pub fn stats(&self) -> String {
        format!("States: {}, Transitions: {}", self.states.len(), self.states.iter().map(|s| s.transitions.len()).sum::<usize>())
    }

    pub fn optimize_for_visualization(&mut self) {
        let n = self.states.len();
        // 1. Forward reachability
        let mut reachable = vec![Weight::zeros(); n];
        if self.body.start_state < n {
            reachable[self.body.start_state] = Weight::all();
        }

        let mut changed = true;
        while changed {
            changed = false;
            for s in 0..n {
                let r_s = reachable[s].clone();
                if r_s.is_empty() { continue; }
                
                for (lbl, &target) in &self.states[s].transitions {
                    if let Some(w) = self.states[s].trans_weights.get(lbl) {
                        if target < n {
                            let flow = &r_s & w;
                            if !flow.is_subset_of(&reachable[target]) {
                                reachable[target] |= &flow;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        
        // 2. Backward reachability
        let mut useful = vec![Weight::zeros(); n];
        changed = true;
        while changed {
            changed = false;
            for s in 0..n {
                let mut u_s = useful[s].clone();
                
                if let Some(fw) = &self.states[s].final_weight {
                    u_s |= fw;
                }
                
                for (lbl, &target) in &self.states[s].transitions {
                    if let Some(w) = self.states[s].trans_weights.get(lbl) {
                        if target < n {
                            let contribution = w & &useful[target];
                            u_s |= &contribution;
                        }
                    }
                }
                
                if !u_s.is_subset_of(&useful[s]) {
                    useful[s] = u_s;
                    changed = true;
                }
            }
        }
        
        // 3. Apply weights
        for s in 0..n {
            if let Some(sw) = &mut self.states[s].state_weight {
                *sw &= &reachable[s];
                *sw &= &useful[s];
                if sw.is_empty() { self.states[s].state_weight = None; }
            }
            
            if let Some(fw) = &mut self.states[s].final_weight {
                *fw &= &reachable[s];
                if fw.is_empty() { self.states[s].final_weight = None; }
            }
            
            let targets: Vec<(Label, StateID)> = self.states[s].transitions.iter().map(|(&l, &t)| (l, t)).collect();
            for (lbl, target) in targets {
                if target < n {
                    if let Some(w) = self.states[s].trans_weights.get_mut(&lbl) {
                        *w &= &reachable[s];
                        *w &= &useful[target];
                    }
                }
            }

            let mut dead_keys = Vec::new();
            for (lbl, w) in &self.states[s].trans_weights {
                if w.is_empty() { dead_keys.push(*lbl); }
            }
            for k in dead_keys {
                self.states[s].trans_weights.remove(&k);
                self.states[s].transitions.remove(&k);
            }
        }
    }
}

impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(sw) = &state.state_weight { writeln!(f, "    state_weight: {}", sw)?; }
            if let Some(w) = &state.final_weight { writeln!(f, "    final_weight: {}", w)?; }
            for (on, to) in &state.transitions {
                let w = state.trans_weights.get(on).cloned().unwrap_or_else(Weight::all);
                if w.is_all_fast() {
                    writeln!(f, "    {} -> {}", format_pos_code(*on), to)?;
                } else {
                    writeln!(f, "    {} -> {} (weight: {})", format_pos_code(*on), to, w)?;
                }
            }
        }
        Ok(())
    }
}
