//! Push weights toward initial state in NWA.

use crate::dwa_i32::common::{Label, NWAStateID, Weight};
use crate::dwa_i32::nwa::NWA;
use std::collections::{HashSet, VecDeque};

impl NWA {
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

        // Build reverse graph
        let mut preds: Vec<Vec<(NWAStateID, Option<Label>, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for &(v, ref w) in &st.epsilons {
                if v < n {
                    preds[v].push((u, None, w.clone()));
                }
            }
            for (&lbl, targets) in &st.transitions {
                for &(v, ref w) in targets {
                    if v < n {
                        preds[v].push((u, Some(lbl), w.clone()));
                    }
                }
            }
        }

        while let Some(v) = q.pop_front() {
            in_queue[v] = false;
            let d_v = d[v].clone();
            if d_v.is_empty() { continue; }

            for (u, _, w) in &preds[v] {
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
        let starts: HashSet<NWAStateID> = self.body.start_states.iter().cloned().collect();

        for (u, st) in self.states.0.iter_mut().enumerate() {
            let d_u = &d[u];
            let inv_d_u = if starts.contains(&u) { Weight::zeros() } else { d_u.complement() };

            // Epsilons
            for (v, w) in &mut st.epsilons {
                if *v < n {
                    let d_v = &d[*v];
                    let new_w = (&*w & d_v) | inv_d_u.clone();
                    if *w != new_w {
                        *w = new_w;
                        changed = true;
                    }
                }
            }
            // Labeled
            for targets in st.transitions.values_mut() {
                for (v, w) in targets {
                    if *v < n {
                        let d_v = &d[*v];
                        let new_w = (&*w & d_v) | inv_d_u.clone();
                        if *w != new_w {
                            *w = new_w;
                            changed = true;
                        }
                    }
                }
            }
            // Final weights
            if let Some(fw) = &mut st.final_weight {
                let new_fw = fw.clone() | inv_d_u.clone();
                if *fw != new_fw {
                    *fw = new_fw;
                    changed = true;
                }
            }
        }
        changed
    }
}
