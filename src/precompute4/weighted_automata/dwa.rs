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
        if n == 0 {
            return;
        }

        // Snapshot effective state weights: `None` means "all tokens".
        let mut state_weights: Vec<Weight> = Vec::with_capacity(n);
        for i in 0..n {
            let w = self.states[i]
                .state_weight
                .clone()
                .unwrap_or_else(Weight::all);
            state_weights.push(w);
        }

        let start = self.body.start_state;
        if start >= n {
            return;
        }

        // 1. Forward tokens: for each state s, tokens that can reach s from the start
        // while satisfying all transition and state weights along the way.
        let mut forward: Vec<Weight> = vec![Weight::zeros(); n];
        forward[start] = state_weights[start].clone();

        let mut changed = true;
        while changed {
            changed = false;
            for u in 0..n {
                let fu = forward[u].clone();
                if fu.is_empty() {
                    continue;
                }

                let state = &self.states[u];
                for (lbl, &v) in &state.transitions {
                    if v >= n {
                        continue;
                    }
                    let w = state
                        .trans_weights
                        .get(lbl)
                        .cloned()
                        .unwrap_or_else(Weight::all);
                    let mut flow = fu.clone();
                    flow &= &w;
                    flow &= &state_weights[v];
                    if !flow.is_subset_of(&forward[v]) {
                        forward[v] |= &flow;
                        changed = true;
                    }
                }
            }
        }

        // 2. Backward tokens: for each state s, tokens that can go from s to some
        // final state while satisfying all transition, state, and final weights.
        let mut backward: Vec<Weight> = vec![Weight::zeros(); n];
        for s in 0..n {
            if let Some(fw) = &self.states[s].final_weight {
                backward[s] |= fw;
            }
        }

        changed = true;
        while changed {
            changed = false;
            for u in (0..n).rev() {
                let mut bu_new = backward[u].clone();
                let state = &self.states[u];
                for (lbl, &v) in &state.transitions {
                    if v >= n {
                        continue;
                    }
                    let w = state
                        .trans_weights
                        .get(lbl)
                        .cloned()
                        .unwrap_or_else(Weight::all);
                    let contribution = &w & &state_weights[v] & &backward[v];
                    if !contribution.is_subset_of(&bu_new) {
                        bu_new |= &contribution;
                    }
                }
                if !bu_new.is_subset_of(&backward[u]) {
                    backward[u] |= &bu_new;
                    changed = true;
                }
            }
        }

        // 3. Apply trimming to states and transitions.
        for s in 0..n {
            let live_state_tokens = &forward[s] & &backward[s];

            // State weights: keep only tokens that appear on some accepted path
            // that passes through this state.
            if live_state_tokens.is_empty() {
                self.states[s].state_weight = None;
            } else {
                let mut new_sw = live_state_tokens.clone();
                new_sw &= &state_weights[s];
                if new_sw.is_all_fast() {
                    self.states[s].state_weight = None;
                } else {
                    self.states[s].state_weight = Some(new_sw);
                }
            }

            // Final weights: tokens must be reachable from the start.
            if let Some(fw) = &mut self.states[s].final_weight {
                *fw &= &forward[s];
                if fw.is_empty() {
                    self.states[s].final_weight = None;
                }
            }

            // Transitions: w_new = w & forward[u] & state_weights[v] & backward[v].
            let labels: Vec<Label> = self.states[s].transitions.keys().copied().collect();
            for lbl in labels {
                let to = match self.states[s].transitions.get(&lbl) {
                    Some(&t) => t,
                    None => continue,
                };
                if to >= n {
                    self.states[s].transitions.remove(&lbl);
                    self.states[s].trans_weights.remove(&lbl);
                    continue;
                }

                let old_w = self.states[s]
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let mut new_w = old_w;
                new_w &= &forward[s];
                new_w &= &state_weights[to];
                new_w &= &backward[to];

                if new_w.is_empty() {
                    self.states[s].transitions.remove(&lbl);
                    self.states[s].trans_weights.remove(&lbl);
                } else if let Some(w_mut) = self.states[s].trans_weights.get_mut(&lbl) {
                    *w_mut = new_w;
                } else {
                    self.states[s].trans_weights.insert(lbl, new_w);
                }
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
