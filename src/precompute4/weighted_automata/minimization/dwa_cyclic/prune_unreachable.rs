//! Prune unreachable states from DWA.

use crate::precompute4::weighted_automata::common::StateID;
use crate::precompute4::weighted_automata::dwa::{DWAStates, DWA};
use std::collections::VecDeque;

impl DWA {
    pub fn prune_unreachable_cyclic(&mut self) -> bool {
        crate::debug!(7, "[DWA] Pruning unreachable states...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let start = self.body.start_state;
        if start >= n {
            let changed = n > 0;
            if changed {
                self.states = DWAStates::default();
                self.body.start_state = 0;
            }
            return changed;
        }

        let mut reachable = vec![false; n];
        let mut q: VecDeque<StateID> = VecDeque::new();
        reachable[start] = true;
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            for (&_lbl, &v) in &self.states[u].transitions {
                if v < n && !reachable[v] {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }
        }

        if reachable.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = DWAStates::default();

        for i in 0..n {
            if reachable[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.transitions.retain(|_, &mut v| v < n && reachable[v]);
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
