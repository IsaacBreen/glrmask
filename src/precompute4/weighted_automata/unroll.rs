use super::common::{StateID, Weight};
use super::dwa::DWA;
use std::collections::{HashMap, VecDeque};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

impl DWA {
    pub fn unroll_cycles(&self) -> DWA {
        let mut new_dwa = DWA::new();
        new_dwa.states.0.clear();
        let start_node = self.body.start_state;
        if start_node >= self.states.len() { return new_dwa; }

        let mut start_weight = Weight::all();
        if let Some(sw) = &self.states[start_node].state_weight { start_weight &= sw; }
        if start_weight.is_empty() { return new_dwa; }

        let mut visited: Vec<Option<HashMap<Weight, StateID>>> = vec![None; self.states.len()];
        let mut queue: VecDeque<(StateID, StateID, Weight)> = VecDeque::new();

        let new_start = new_dwa.add_state();
        new_dwa.body.start_state = new_start;
        new_dwa.states[new_start].state_weight = self.states[start_node].state_weight.clone();
        new_dwa.states[new_start].final_weight = self.states[start_node].final_weight.clone();

        visited[start_node] = Some(HashMap::from([(start_weight.clone(), new_start)]));
        queue.push_back((new_start, start_node, start_weight.clone()));

        let m = MultiProgress::new();
        let pb = m.add(ProgressBar::new_spinner());
        pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} [{elapsed_precise}] Unrolled states: {pos}").unwrap());

        let mut processed_count: u64 = 0;
        while let Some((new_u, _, w_u)) = queue.pop_front() {
            processed_count += 1;
            pb.set_position(processed_count);
            // Note: accessing `u` from queue tuple was ignored above using `_` by mistake in previous versions.
            // Actually we need `u` (original_state_id).
            // Re-fetching from queue properly:
        }
        
        // Correct logic:
        queue.clear(); 
        visited = vec![None; self.states.len()];
        new_dwa = DWA::new();
        new_dwa.states.0.clear();
        
        let new_start = new_dwa.add_state();
        new_dwa.body.start_state = new_start;
        new_dwa.states[new_start].state_weight = self.states[start_node].state_weight.clone();
        new_dwa.states[new_start].final_weight = self.states[start_node].final_weight.clone();

        visited[start_node] = Some(HashMap::from([(start_weight.clone(), new_start)]));
        queue.push_back((new_start, start_node, start_weight));
        
        while let Some((new_u, u, w_u)) = queue.pop_front() {
            processed_count += 1;
            pb.set_position(processed_count);
            let u_state = &self.states[u];
            
            for (&label, &v) in &u_state.transitions {
                if v >= self.states.len() { continue; }
                let trans_w = u_state.trans_weights.get(&label).unwrap(); // Must exist if transition exists
                let mut next_w = &w_u & trans_w;
                if next_w.is_empty() { continue; }
                
                let v_state = &self.states[v];
                if let Some(sw) = &v_state.state_weight { next_w &= sw; if next_w.is_empty() { continue; } }

                let v_visited = visited[v].get_or_insert_with(HashMap::new);
                let new_v = if let Some(&id) = v_visited.get(&next_w) { id } else {
                    let id = new_dwa.add_state();
                    new_dwa.states[id].state_weight = v_state.state_weight.clone();
                    new_dwa.states[id].final_weight = v_state.final_weight.clone();
                    v_visited.insert(next_w.clone(), id);
                    queue.push_back((id, v, next_w.clone()));
                    id
                };
                new_dwa.add_transition(new_u, label, new_v, trans_w.clone()).unwrap();
            }
        }
        pb.finish_with_message(format!("Done ({} states)", new_dwa.states.len()));
        new_dwa
    }
}
