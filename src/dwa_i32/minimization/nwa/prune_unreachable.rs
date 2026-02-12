//! Prune unreachable states from NWA.

use crate::dwa_i32::common::{Label, NWAStateID};
use crate::dwa_i32::nwa::{NWAStates, NWA};
use std::collections::{BTreeMap, VecDeque};

impl NWA {
    pub fn prune_unreachable(&mut self) -> bool {
        crate::debug!(7, "[NWA] Pruning unreachable states...");
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        if self.body.start_states.is_empty() {
            let changed = n > 0;
            if changed {
                self.states = NWAStates::default();
                self.body.start_states.clear();
            }
            return changed;
        }

        let mut reachable = vec![false; n];
        let mut q: VecDeque<NWAStateID> = VecDeque::new();

        for &start in &self.body.start_states {
            if start < n && !reachable[start] {
                reachable[start] = true;
                q.push_back(start);
            }
        }

        while let Some(u) = q.pop_front() {
            let st = &self.states[u];

            for &(v, ref w) in &st.epsilons {
                if v < n && !reachable[v] && !w.is_empty() {
                    reachable[v] = true;
                    q.push_back(v);
                }
            }

            for (_, targets) in &st.transitions {
                for &(v, ref w) in targets {
                    if v < n && !reachable[v] && !w.is_empty() {
                        reachable[v] = true;
                        q.push_back(v);
                    }
                }
            }
        }

        if reachable.iter().all(|&b| b) {
            return false;
        }

        let mut map = vec![usize::MAX; n];
        let mut new_states = NWAStates::default();

        // Move states instead of cloning — avoids expensive BTreeMap/Vec clones.
        for i in 0..n {
            if reachable[i] {
                let new_id = new_states.add_state();
                map[i] = new_id;
                new_states[new_id] = std::mem::take(&mut self.states.0[i]);
            }
        }

        // Remap transition targets in-place (no new BTreeMap allocation).
        for st in &mut new_states.0 {
            st.epsilons.retain(|(v, w)| *v < n && reachable[*v] && !w.is_empty());
            for (v, _) in &mut st.epsilons {
                *v = map[*v];
            }

            for targets in st.transitions.values_mut() {
                targets.retain(|(v, w)| *v < n && reachable[*v] && !w.is_empty());
                for (v, _) in targets.iter_mut() {
                    *v = map[*v];
                }
            }
            st.transitions.retain(|_, targets| !targets.is_empty());
        }

        let mut new_start_states = Vec::new();
        for &s in &self.body.start_states {
            if s < n && reachable[s] {
                new_start_states.push(map[s]);
            }
        }
        self.body.start_states = new_start_states;
        // Drop the old (mostly empty after take) states on a background thread
        // to avoid blocking on 1.7M NWAState destructors.
        let old_states = std::mem::replace(&mut self.states, new_states);
        std::thread::spawn(move || drop(old_states));
        true
    }
}
