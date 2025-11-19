use super::common::{StateID, Weight};
use super::dwa::DWA;
use std::collections::{HashMap, VecDeque};

impl DWA {
    /// Unrolls cycles in the DWA by expanding states into (state, accumulated_weight) pairs.
    /// This relies on the property that cycles are finite because weights (bitsets)
    /// strictly decrease (via intersection) until they become empty.
    pub fn unroll_cycles(&self) -> DWA {
        let mut new_dwa = DWA::new();
        new_dwa.states.0.clear(); // Remove default start state

        let start_node = self.body.start_state;
        if start_node >= self.states.len() {
            return new_dwa;
        }

        let mut start_weight = Weight::all();
        if let Some(sw) = &self.states[start_node].state_weight {
            start_weight &= sw;
        }

        if start_weight.is_empty() {
            return new_dwa;
        }

        // Map (original_state, accumulated_weight) -> new_state_id
        let mut visited: HashMap<(StateID, Weight), StateID> = HashMap::new();
        let mut queue: VecDeque<(StateID, Weight)> = VecDeque::new();

        let new_start = new_dwa.add_state();
        new_dwa.body.start_state = new_start;

        // Copy properties for start state
        new_dwa.states[new_start].state_weight = self.states[start_node].state_weight.clone();
        new_dwa.states[new_start].final_weight = self.states[start_node].final_weight.clone();

        visited.insert((start_node, start_weight.clone()), new_start);
        queue.push_back((start_node, start_weight));

        while let Some((u, w_u)) = queue.pop_front() {
            let new_u = *visited.get(&(u, w_u.clone())).unwrap();
            let u_state = &self.states[u];

            for (&label, &v) in &u_state.transitions {
                if v >= self.states.len() {
                    continue;
                }

                let trans_w = u_state.trans_weights.get(&label).unwrap();
                let mut next_w = w_u.clone();
                next_w &= trans_w;

                let v_state = &self.states[v];
                if let Some(sw) = &v_state.state_weight {
                    next_w &= sw;
                }

                if !next_w.is_empty() {
                    let next_key = (v, next_w.clone());

                    let new_v = if let Some(&id) = visited.get(&next_key) {
                        id
                    } else {
                        let id = new_dwa.add_state();
                        new_dwa.states[id].state_weight = v_state.state_weight.clone();
                        new_dwa.states[id].final_weight = v_state.final_weight.clone();
                        visited.insert(next_key.clone(), id);
                        queue.push_back(next_key);
                        id
                    };

                    let _ = new_dwa.add_transition(new_u, label, new_v, trans_w.clone());
                }
            }
        }

        new_dwa
    }
}
