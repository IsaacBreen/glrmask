// src/precompute4/weighted_automata/unroll.rs

use super::common::{StateID, Weight};
use super::dwa::DWA;
use std::collections::{HashMap, VecDeque};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

impl DWA {
    pub fn unroll_cycles(&self) -> DWA {
        let start = self.body.start_state;
        if start >= self.states.len() { return DWA::new(); }

        let mut new_dwa = DWA::new();
        new_dwa.states.0.clear();
        let new_start = new_dwa.add_state();
        new_dwa.body.start_state = new_start;

        let mut sw = Weight::all();
        if let Some(w) = &self.states[start].state_weight { sw &= w; }
        if sw.is_empty() { return new_dwa; }

        new_dwa.states[new_start].final_weight = self.states[start].final_weight.clone();
        new_dwa.states[new_start].state_weight = self.states[start].state_weight.clone();

        let mut visited = vec![None; self.states.len()];
        visited[start] = Some(HashMap::from([(sw.clone(), new_start)]));
        let mut q = VecDeque::from([(new_start, start, sw)]);

        let mp = MultiProgress::new();
        let pb = mp.add(ProgressBar::new_spinner());
        pb.set_style(ProgressStyle::default_spinner().template("{spinner:.green} Unrolling: {pos}").unwrap());

        let mut count = 0;
        while let Some((nu, u, wu)) = q.pop_front() {
            count += 1; pb.set_position(count);
            
            let st = &self.states[u];
            let mut trans = Vec::new();
            let mut weights = Vec::new();

            for (&l, &v) in &st.transitions {
                if v >= self.states.len() { continue; }
                let tw = &st.trans_weights[&l];
                let mut next_w = &wu & tw;
                if next_w.is_empty() { continue; }
                if let Some(w) = &self.states[v].state_weight { next_w &= w; if next_w.is_empty() { continue; } }

                let v_vis = visited[v].get_or_insert_with(HashMap::new);
                let nv = if let Some(&id) = v_vis.get(&next_w) { id } else {
                    let id = new_dwa.add_state();
                    new_dwa.states[id].final_weight = self.states[v].final_weight.clone();
                    new_dwa.states[id].state_weight = self.states[v].state_weight.clone();
                    v_vis.insert(next_w.clone(), id);
                    q.push_back((id, v, next_w));
                    id
                };
                trans.push((l, nv));
                weights.push((l, tw.clone()));
            }
            new_dwa.states[nu].transitions = trans.into_iter().collect();
            new_dwa.states[nu].trans_weights = weights.into_iter().collect();
        }
        pb.finish_with_message(format!("Done: {} states", new_dwa.states.len()));
        new_dwa
    }
}
