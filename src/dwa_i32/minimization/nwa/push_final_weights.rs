//! Push final weights along epsilons in NWA.

use crate::dwa_i32::common::{NWAStateID, Weight};
use crate::dwa_i32::nwa::NWA;
use std::collections::VecDeque;

impl NWA {
    pub fn push_final_weights_along_epsilons(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pushing final weights along epsilons...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Build reverse epsilon adjacency
        let mut rev_eps: Vec<Vec<(NWAStateID, Weight)>> = vec![Vec::new(); n];
        for (u, st) in self.states.0.iter().enumerate() {
            for &(v, ref w) in &st.epsilons {
                if v < n && !w.is_empty() {
                    rev_eps[v].push((u, w.clone()));
                }
            }
        }

        let mut final_weights: Vec<Weight> = Vec::with_capacity(n);
        let mut queue: VecDeque<NWAStateID> = VecDeque::new();
        for i in 0..n {
            let w = self.states.0[i].final_weight.clone().unwrap_or_else(Weight::zeros);
            if !w.is_empty() {
                queue.push_back(i);
            }
            final_weights.push(w);
        }

        let mut changed = false;

        while let Some(v) = queue.pop_front() {
            let w_v = final_weights[v].clone();
            if w_v.is_empty() {
                continue;
            }

            for &(u, ref w_uv) in &rev_eps[v] {
                let candidate = &w_v & w_uv;
                if candidate.is_empty() {
                    continue;
                }
                let new_w = &final_weights[u] | &candidate;
                if new_w != final_weights[u] {
                    final_weights[u] = new_w;
                    queue.push_back(u);
                }
            }
        }

        for i in 0..n {
            let new_w = &final_weights[i];
            let new_final = if new_w.is_empty() { None } else { Some(new_w.clone()) };
            if self.states.0[i].final_weight != new_final {
                self.states.0[i].final_weight = new_final;
                changed = true;
            }
        }

        changed
    }
}
