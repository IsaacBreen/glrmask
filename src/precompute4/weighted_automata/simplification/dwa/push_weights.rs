//! Push weights from state_weight into transitions and finals.

use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::DWA;

impl DWA {
    pub fn push_weights_into_transitions_and_finals(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        let start = self.body.start_state;
        if start >= n {
            return false;
        }

        let mut changed = false;
        let mut preds: Vec<Vec<(StateID, Label)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    preds[v].push((u, label));
                }
            }
        }

        for v in 0..n {
            if v == start {
                continue;
            }
            if let Some(sw) = self.states[v].state_weight.take() {
                if sw.is_empty() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else if sw != Weight::all() {
                    changed = true;
                    for (u, label) in &preds[v] {
                        if let Some(w) = self.states[*u].trans_weights.get_mut(label) {
                            *w &= &sw;
                        }
                    }
                } else {
                    changed = true;
                }
            }
        }

        if let Some(sw0) = self.states[start].state_weight.take() {
            if !sw0.is_empty() && sw0 != Weight::all() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else if sw0.is_empty() {
                changed = true;
                for st in &mut self.states.0 {
                    if let Some(ref mut fw) = st.final_weight {
                        *fw &= &sw0;
                    }
                }
            } else {
                changed = true;
            }
        }

        changed
    }
}
