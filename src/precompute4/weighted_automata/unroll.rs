use super::common::{StateID, Weight};
use super::dwa::DWA;
use std::collections::{BTreeMap, HashMap, VecDeque};

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

        // visited[original_state_id] -> HashMap<Weight, new_state_id>
        // Use Option to lazy initialize HashMaps to save memory/time for sparse traversals
        let mut visited: Vec<Option<HashMap<Weight, StateID>>> = vec![None; self.states.len()];
        
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

        visited[start_node] = Some(HashMap::from([(start_weight.clone(), new_start)]));
        queue.push_back((new_start, start_node, start_weight));

        while let Some((new_u, u, w_u)) = queue.pop_front() {
            let u_state = &self.states[u];

            // Collect transitions to bulk insert later, avoiding repeated re-borrows of new_dwa
            // Pre-allocate vectors for bulk BTreeMap construction
            let capacity = u_state.transitions.len();
            let mut new_transitions_vec = Vec::with_capacity(capacity);
            let mut new_trans_weights_vec = Vec::with_capacity(capacity);

            // Iterate transitions and weights in lockstep to avoid O(log N) lookups
            for ((&label, &v), (_, trans_w)) in u_state.transitions.iter().zip(u_state.trans_weights.iter()) {
                if v >= self.states.len() {
                    continue;
                }

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

                let v_visited = visited[v].get_or_insert_with(HashMap::new);

                let new_v = if let Some(&id) = v_visited.get(&next_w) {
                    id
                } else {
                    let id = new_dwa.add_state();
                    // Initialize new state
                    let new_st = &mut new_dwa.states[id];
                    new_st.state_weight = v_state.state_weight.clone();
                    new_st.final_weight = v_state.final_weight.clone();
                    
                    v_visited.insert(next_w.clone(), id);
                    queue.push_back((id, v, next_w));
                    id
                };

                new_transitions_vec.push((label, new_v));
                new_trans_weights_vec.push((label, trans_w.clone()));
            }

            let src_st = &mut new_dwa.states[new_u];
            src_st.transitions = new_transitions_vec.into_iter().collect();
            src_st.trans_weights = new_trans_weights_vec.into_iter().collect();
        }

        new_dwa
    }
}
