//! Prune dead-end states from DWA.

use crate::precompute4::weighted_automata::common::StateID;
use crate::precompute4::weighted_automata::dwa::{DWAStates, DWA};
use std::collections::VecDeque;

impl DWA {
    pub fn prune_dead_ends_cyclic(&mut self) -> bool {
        crate::debug!(7, "[DWA] Pruning dead ends...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        let mut rev: Vec<Vec<StateID>> = vec![vec![]; n];

        for p in 0..n {
            for &t in self.states[p].transitions.values() {
                if t < n {
                    rev[t].push(p);
                }
            }
        }

        for s in 0..n {
            if self.states[s].final_weight.as_ref().map_or(false, |w| !w.is_empty()) {
                if !live[s] {
                    live[s] = true;
                    q.push_back(s);
                }
            }
        }

        while let Some(v) = q.pop_front() {
            for &p in &rev[v] {
                if !live[p] {
                    live[p] = true;
                    q.push_back(p);
                }
            }
        }

        if live.iter().all(|&b| b) {
            return false;
        }

        let start = self.body.start_state;
        if start >= n || !live[start] {
            if n == 0 {
                return false;
            }
            self.states = DWAStates::default();
            self.body.start_state = 0;
            return true;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.transitions.retain(|_, &mut v| v < n && live[v]);
            for v in st.transitions.values_mut() {
                *v = map[*v];
            }
            st.trans_weights.retain(|lbl, _| st.transitions.contains_key(lbl));
        }

        self.body.start_state = map[start];
        self.states = new_states;
        true
    }
}
