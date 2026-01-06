//! Prune dead-end states from NWA.

use crate::precompute4::weighted_automata::common::{Label, NWAStateID};
use crate::precompute4::weighted_automata::nwa::{NWAStates, NWA};
use std::collections::{BTreeMap, VecDeque};

impl NWA {
    pub fn prune_dead_ends(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pruning dead ends...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        let mut live = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();
        let mut rev: Vec<Vec<NWAStateID>> = vec![vec![]; n];

        for p in 0..n {
            let st = &self.states[p];
            for &(t, ref w) in &st.epsilons {
                if t < n && !w.is_empty() {
                    rev[t].push(p);
                }
            }
            for (_, targets) in &st.transitions {
                for &(t, ref w) in targets {
                    if t < n && !w.is_empty() {
                        rev[t].push(p);
                    }
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
        
        let any_start_live = self.body.start_states.iter().any(|&s| s < n && live[s]);
        if !any_start_live {
             if n == 0 { return false; }
             self.states = NWAStates::default();
             self.body.start_states.clear();
             return true;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        for i in 0..n {
            if live[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = self.states[i].clone();
            }
        }

        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && !w.is_empty() && live[*v]);
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            let mut new_transitions: BTreeMap<Label, Vec<(NWAStateID, crate::precompute4::weighted_automata::Weight)>> = BTreeMap::new();
            for (&lbl, targets) in &st.transitions {
                let mut new_targets = Vec::new();
                for &(v, ref w) in targets {
                    if v < n && !w.is_empty() && live[v] {
                        new_targets.push((map[v], w.clone()));
                    }
                }
                if !new_targets.is_empty() {
                    new_transitions.insert(lbl, new_targets);
                }
            }
            st.transitions = new_transitions;
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
