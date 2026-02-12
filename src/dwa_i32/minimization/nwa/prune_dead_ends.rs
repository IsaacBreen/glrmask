//! Prune dead-end states from NWA.

use crate::dwa_i32::common::{Label, NWAStateID};
use crate::dwa_i32::nwa::{NWAStates, NWA};
use std::collections::{BTreeMap, VecDeque};

impl NWA {
    pub fn prune_dead_ends(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pruning dead ends...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Phase 1: Build reverse graph using flat CSR format.
        // Two passes over edges: first count, then fill. 
        // Avoids 1.75M Vec allocations from vec![vec![]; n].
        let mut rev_count = vec![0u32; n];
        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    rev_count[t] += 1;
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        rev_count[t] += 1;
                    }
                }
            }
        }

        // Compute prefix sums for CSR offsets
        let mut rev_offset = vec![0u32; n + 1];
        for i in 0..n {
            rev_offset[i + 1] = rev_offset[i] + rev_count[i];
        }
        let total_rev_edges = rev_offset[n] as usize;
        
        // Fill flat reverse edge array
        let mut rev_edges: Vec<NWAStateID> = vec![0; total_rev_edges];
        let mut write_pos = vec![0u32; n];
        
        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    let pos = rev_offset[t] as usize + write_pos[t] as usize;
                    rev_edges[pos] = p;
                    write_pos[t] += 1;
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        let pos = rev_offset[t] as usize + write_pos[t] as usize;
                        rev_edges[pos] = p;
                        write_pos[t] += 1;
                    }
                }
            }
        }
        drop(rev_count);
        drop(write_pos);

        // Phase 2: BFS backward from final states
        let mut live = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();

        for s in 0..n {
            if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                if !live[s] {
                    live[s] = true;
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            let start = rev_offset[v] as usize;
            let end = rev_offset[v + 1] as usize;
            for &p in &rev_edges[start..end] {
                if !live[p] {
                    live[p] = true;
                    q.push_back(p);
                }
            }
        }

        drop(rev_edges);
        drop(rev_offset);

        if live.iter().all(|&b| b) {
            return false;
        }
        
        let any_start_live = self.body.start_states.iter().any(|&s| s < n && live[s]);
        if !any_start_live {
             if n == 0 { return false; }
             self.states = NWAStates::default();
             self.body.start_states.clear();
             return true;
        }

        // Phase 3: Remap states
        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        // Move states instead of cloning — avoids expensive BTreeMap/Vec clones.
        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = std::mem::take(&mut self.states.0[i]);
            }
        }

        // Remap transition targets in-place (no new BTreeMap allocation).
        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty() && live[*v]);
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            for targets in st.transitions.values_mut() {
                targets.retain(|(v, w)| *v < n && !w.is_empty() && live[*v]);
                for (v, _) in targets.iter_mut() {
                    *v = map[*v];
                }
            }
            st.transitions.retain(|_, targets| !targets.is_empty());
        }

        let mut new_start_states = Vec::new();
        for &s in &self.body.start_states {
            if s < n && live[s] {
                new_start_states.push(map[s]);
            }
        }
        self.body.start_states = new_start_states;
        self.states = new_states;
        true
    }
}
