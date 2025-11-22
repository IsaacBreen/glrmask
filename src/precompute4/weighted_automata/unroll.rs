use super::common::{StateID, Weight};
use super::dwa::DWA;
use std::collections::{HashMap, VecDeque};
use indicatif::{ProgressBar, ProgressStyle};

impl DWA {
    pub fn unroll_cycles(&self) -> DWA {
        let start = self.body.start_state;
        if start >= self.states.len() { return DWA::new(); }
        let mut dwa = DWA::new(); dwa.states.0.clear();
        let new_start = dwa.add_state(); dwa.body.start_state = new_start;

        let mut q = VecDeque::from([(new_start, start, Weight::all())]);
        // visited[orig_state] -> {weight: new_state}
        let mut visited = vec![HashMap::new(); self.states.len()];
        visited[start].insert(Weight::all(), new_start);

        let pb = ProgressBar::new_spinner();
        pb.set_style(ProgressStyle::default_spinner().template("{spinner} Unrolling {pos} states").unwrap());

        while let Some((nu, u, w)) = q.pop_front() {
            pb.inc(1);
            let st = &self.states[u];
            dwa.states[nu].final_weight = st.final_weight.clone(); // Final weight is just copied, effectively
            // Real intersection for transitions
            for (&l, &v) in &st.transitions {
                if v >= self.states.len() { continue; }
                let edge_w = &st.trans_weights[&l] & &w;
                if edge_w.is_empty() { continue; }
                
                let nv = *visited[v].entry(edge_w.clone()).or_insert_with(|| {
                    let id = dwa.add_state();
                    q.push_back((id, v, edge_w.clone()));
                    id
                });
                let _ = dwa.add_transition(nu, l, nv, st.trans_weights[&l].clone());
            }
        }
        pb.finish_with_message(format!("Done: {} states", dwa.states.len()));
        dwa
    }
}
