//! Push weights toward initial state.

use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::DWA;
use std::collections::VecDeque;

impl DWA {
    pub fn push_weights_to_initial(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }

        // 1. Compute backward distance (accumulated weight to final)
        let mut d = vec![Weight::zeros(); n];
        let mut q = VecDeque::new();
        let mut in_queue = vec![false; n];

        // Initialize with final weights
        for i in 0..n {
            if let Some(fw) = &self.states[i].final_weight {
                if !fw.is_empty() {
                    d[i] = fw.clone();
                    q.push_back(i);
                    in_queue[i] = true;
                }
            }
        }

        // Build reverse graph for propagation
        let mut preds: Vec<Vec<(StateID, Label, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for (&label, &v) in &st.transitions {
                if v < n {
                    let w = st.trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                    preds[v].push((u, label, w));
                }
            }
        }

        while let Some(v) = q.pop_front() {
            in_queue[v] = false;
            let d_v = d[v].clone();
            if d_v.is_empty() { continue; }

            for (u, _label, w) in &preds[v] {
                // d[u] += w * d[v]
                let new_d = w & &d_v;
                if !new_d.is_subset_of(&d[*u]) {
                    d[*u] |= &new_d;
                    if !in_queue[*u] {
                        q.push_back(*u);
                        in_queue[*u] = true;
                    }
                }
            }
        }

        // 2. Reweight
        let mut changed = false;
        let start_node = self.body.start_state;
        for (u, st) in self.states.0.iter_mut().enumerate() {
            let d_u = &d[u];
            let inv_d_u = if u == start_node { Weight::zeros() } else { d_u.complement() };

            // Transitions
            for (&label, &v) in &st.transitions {
                if v < n {
                    let d_v = &d[v];
                    if let Some(w) = st.trans_weights.get_mut(&label) {
                        let new_w = (&*w & d_v) | &inv_d_u;
                        if *w != new_w {
                            *w = new_w;
                            changed = true;
                        }
                    }
                }
            }
            
            // Final weights
            if let Some(fw) = &mut st.final_weight {
                let new_fw = &*fw | &inv_d_u;
                if *fw != new_fw {
                    *fw = new_fw;
                    changed = true;
                }
            }
        }
        changed
    }
}
