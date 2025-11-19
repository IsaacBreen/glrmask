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

        // Optimization: Use a vector of maps instead of a single map.
        // visited[original_state_id] -> HashMap<Weight, new_state_id>
        let mut visited: Vec<HashMap<Weight, StateID>> = vec![HashMap::new(); self.states.len()];
        
        // Queue stores (new_state_id, original_state_id, accumulated_weight)
        let mut queue: VecDeque<(StateID, StateID, Weight)> = VecDeque::new();

        let new_start = new_dwa.add_state();
        new_dwa.body.start_state = new_start;

        // Copy properties for start state
        {
            let start_state_ref = &self.states[start_node];
            let new_state_ref = &mut new_dwa.states[new_start];
            new_state_ref.state_weight = start_state_ref.state_weight.clone();
            new_state_ref.final_weight = start_state_ref.final_weight.clone();
        }

        visited[start_node].insert(start_weight.clone(), new_start);
        queue.push_back((new_start, start_node, start_weight));

        while let Some((new_u, u, w_u)) = queue.pop_front() {
            let u_state = &self.states[u];

            for (&label, &v) in &u_state.transitions {
                if v >= self.states.len() {
                    continue;
                }

                let trans_w = u_state.trans_weights.get(&label).unwrap();
                let mut next_w = &w_u & trans_w;
                if next_w.is_empty() {
                    continue;
                }

                let v_state = &self.states[v];
                if let Some(sw) = &v_state.state_weight {
                    next_w &= sw;
                    if next_w.is_empty() {
                        continue;
                    }
                }

                let v_visited = &mut visited[v];
                let new_v = if let Some(&id) = v_visited.get(&next_w) {
                    id
                } else {
                    let id = new_dwa.add_state();
                    let new_st = &mut new_dwa.states[id];
                    new_st.state_weight = v_state.state_weight.clone();
                    new_st.final_weight = v_state.final_weight.clone();
                    
                    v_visited.insert(next_w.clone(), id);
                    queue.push_back((id, v, next_w));
                    id
                };

                let src_st = &mut new_dwa.states[new_u];
                src_st.transitions.insert(label, new_v);
                src_st.trans_weights.insert(label, trans_w.clone());
            }
        }

        new_dwa
    }
}
