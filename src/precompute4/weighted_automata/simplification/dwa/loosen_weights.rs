//! Loosen weights for minimization.
//!
//! For each state, computes "don't care" weights - tokens that can't reach this state.
//! These are added to transition and final weights, making states more similar.

use crate::precompute4::weighted_automata::common::{StateID, Weight};
use crate::precompute4::weighted_automata::dwa::DWA;
use std::collections::VecDeque;

impl DWA {
    /// Loosen weights before minimization to enable more state merging.
    ///
    /// Algorithm:
    /// 1. Compute Pre(q) = tokens that can reach state q from start
    /// 2. Loosen finals: w*_final(q) = w_final(q) | !Pre(q)
    /// 3. Loosen transitions: w*_trans(p,a) = w_trans(p,a) | !Pre(p)
    pub fn loosen_weights_for_minimize(&mut self) -> bool {
        if std::env::var("DISABLE_WEIGHT_LOOSENING").map(|v| v == "1").unwrap_or(false) {
            return false;
        }

        let n = self.states.len();
        if n == 0 {
            return false;
        }
        
        let start = self.body.start_state;
        if start >= n {
            return false;
        }
        
        // === Phase 1: Compute Pre(q) using Topological Sort (Kahn's Algorithm) ===
        let mut in_degree = vec![0; n];
        for st in self.states.0.iter() {
            for &v in st.transitions.values() {
                if v < n {
                    in_degree[v] += 1;
                }
            }
        }
        
        let mut pre: Vec<Weight> = vec![Weight::zeros(); n];
        pre[start] = Weight::all();
        
        let mut queue: VecDeque<StateID> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }
        
        let mut visited_count = 0;
        
        while let Some(u) = queue.pop_front() {
            visited_count += 1;
            
            let u_is_reachable = !pre[u].is_empty();
            let pre_u = if u_is_reachable { pre[u].clone() } else { Weight::zeros() };
            
            for (&label, &v) in &self.states[u].transitions {
                if v >= n {
                    continue;
                }
                
                if u_is_reachable {
                    let w = self.states[u].trans_weights.get(&label).cloned().unwrap_or_else(Weight::all);
                    if !w.is_empty() {
                        let flow = &pre_u & &w;
                        if !flow.is_empty() {
                             pre[v] |= &flow;
                        }
                    }
                }
                
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        
        // If cycle detected, abort loosening
        if visited_count < n {
            return false;
        }
        
        // === Phase 2: Loosen weights ===
        let mut changed = false;
        
        // 1. Loosen final weights based on Pre(q)
        for q in 0..n {
            if let Some(ref mut fw) = self.states[q].final_weight {
                let dont_care = pre[q].complement();
                if !dont_care.is_empty() && !dont_care.is_subset_of(fw) {
                    let new_fw = &*fw | &dont_care;
                    if new_fw != *fw {
                        *fw = new_fw;
                        changed = true;
                    }
                }
            }
        }
        
        // 2. Loosen transition weights based on Pre(p)
        for p in 0..n {
            let dont_care = pre[p].complement();
            if dont_care.is_empty() {
                continue;
            }
            
            let labels: Vec<i32> = self.states[p].transitions.keys().copied().collect();
            for label in labels {
                if let Some(w) = self.states[p].trans_weights.get_mut(&label) {
                    if !dont_care.is_subset_of(w) {
                        let new_w = &*w | &dont_care;
                        if new_w != *w {
                            *w = new_w;
                            changed = true;
                        }
                    }
                }
            }
        }
        
        changed
    }
}
