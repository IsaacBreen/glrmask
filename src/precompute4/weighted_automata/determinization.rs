#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use chrono::Local;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::common::{DETERMINIZE_DEBUG, Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::{NWA, NWAStates};
use crate::precompute4::test_weighted_automata;

// Invariants: strictly sorted by NWAStateID, no duplicate IDs, no empty Weights.
type WeightedSubset = Vec<(NWAStateID, Weight)>;

fn is_zero(w: &Weight) -> bool { w.is_empty() }

struct Determinizer<'a> {
    nwa: &'a NWA,
    
    // Map from canonical closure (Sorted Vec) to DWA State ID
    seen: HashMap<WeightedSubset, usize>,
    queue: VecDeque<usize>,
    // Store the closure for each DWA state
    closures: Vec<WeightedSubset>,
    
    dwa: DWA,

    // Reusable buffers to avoid allocations
    trans_buffer: Vec<(Label, NWAStateID, Weight)>,
    local_weight_map: Vec<Option<Weight>>,
    dirty_indices: Vec<usize>,
    bfs_queue: VecDeque<NWAStateID>,
    reach_buffer: WeightedSubset,
}

impl<'a> Determinizer<'a> {
    fn new(nwa: &'a NWA) -> Self {
        let mut dwa = DWA::new();
        dwa.states.0.clear();
        dwa.body.start_state = 0;
        
        let num_states = nwa.states.len();

        Determinizer {
            nwa,
            seen: HashMap::new(),
            queue: VecDeque::new(),
            closures: Vec::new(),
            dwa,
            trans_buffer: Vec::with_capacity(1024),
            local_weight_map: vec![None; num_states],
            dirty_indices: Vec::with_capacity(num_states),
            bfs_queue: VecDeque::with_capacity(64),
            reach_buffer: Vec::with_capacity(64),
        }
    }

    fn register_closure(&mut self, closure: WeightedSubset) -> usize {
        if let Some(&id) = self.seen.get(&closure) {
            return id;
        }

        let id = self.dwa.add_state();

        // Compute final weight for this new DWA state
        let mut finalw = Weight::zeros();
        for (sid, cw) in &closure {
            if let Some(fw) = &self.nwa.states[*sid].final_weight {
                let cand = cw & fw;
                if !cand.is_empty() {
                    finalw |= &cand;
                }
            }
        }
        if !finalw.is_empty() {
            let _ = self.dwa.set_final_weight(id, finalw);
        }

        self.seen.insert(closure.clone(), id);
        self.closures.push(closure);
        self.queue.push_back(id);
        id
    }

    /// Computes the epsilon closure of the given (state, weight) pairs.
    /// Result is stored in self.reach_buffer (sorted).
    fn compute_closure_into_buffer(&mut self, seed: impl Iterator<Item = (NWAStateID, Weight)>) {
        // 1. Initialize BFS with seed
        for (u, w) in seed {
            if w.is_empty() { continue; }
            if u >= self.local_weight_map.len() { continue; }

            match &mut self.local_weight_map[u] {
                Some(existing) => {
                    if !w.is_subset_of(existing) {
                        *existing |= &w;
                        self.bfs_queue.push_back(u);
                    }
                },
                None => {
                    self.local_weight_map[u] = Some(w);
                    self.dirty_indices.push(u);
                    self.bfs_queue.push_back(u);
                }
            }
        }

        // 2. Run BFS
        while let Some(u) = self.bfs_queue.pop_front() {
            // Clone weight to avoid borrowing issues
            let w_u = match &self.local_weight_map[u] {
                Some(w) => w.clone(),
                None => continue,
            };

            if u >= self.nwa.states.len() { continue; }

            for (v, w_eps) in &self.nwa.states[u].epsilons {
                if *v >= self.local_weight_map.len() { continue; }

                let w_new = &w_u & w_eps;
                if w_new.is_empty() { continue; }

                match &mut self.local_weight_map[*v] {
                    Some(existing) => {
                        if !w_new.is_subset_of(existing) {
                            *existing |= &w_new;
                            self.bfs_queue.push_back(*v);
                        }
                    },
                    None => {
                        self.local_weight_map[*v] = Some(w_new);
                        self.dirty_indices.push(*v);
                        self.bfs_queue.push_back(*v);
                    }
                }
            }
        }

        // 3. Collect results
        self.reach_buffer.clear();
        for &idx in &self.dirty_indices {
            if let Some(w) = self.local_weight_map[idx].take() {
                self.reach_buffer.push((idx, w));
            }
        }
        
        // 4. Cleanup
        // local_weight_map entries were taken (set to None) above.
        // Just clear the dirty list.
        self.dirty_indices.clear();

        // 5. Sort
        self.reach_buffer.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }

    fn expand_state(&mut self, sid: usize) {
        let closure_idx = sid;
        // We can't hold a reference to closures[sid] while mutating self.
        // But closures are append-only and we only need to read *this* one.
        // To be safe with Rust borrow checker, clone it or index carefully.
        // Cloning the closure (Vec) is relatively cheap compared to the work.
        let closure = self.closures[closure_idx].clone();

        if closure.is_empty() {
            return;
        }

        // 1. Collect all immediate transitions
        self.trans_buffer.clear();
        for (u, w_u) in &closure {
            if *u >= self.nwa.states.len() { continue; }
            let st = &self.nwa.states[*u];
            
            for (lbl, targets) in &st.transitions {
                for (v, w_trans) in targets {
                    let w_comb = w_u & w_trans;
                    if !w_comb.is_empty() {
                        self.trans_buffer.push((*lbl, *v, w_comb));
                    }
                }
            }
        }

        if self.trans_buffer.is_empty() {
            return;
        }

        // 2. Sort by Label to group
        self.trans_buffer.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        // 3. Iterate groups
        let mut i = 0;
        while i < self.trans_buffer.len() {
            let lbl = self.trans_buffer[i].0;
            let mut j = i;
            
            let mut edge_weight = Weight::zeros();
            
            // Identify range for this label
            while j < self.trans_buffer.len() && self.trans_buffer[j].0 == lbl {
                edge_weight |= &self.trans_buffer[j].2;
                j += 1;
            }

            // Compute closure for the targets in this range
            // We pass an iterator of (v, w)
            let targets_iter = self.trans_buffer[i..j].iter().map(|(_, v, w)| (*v, w.clone()));
            self.compute_closure_into_buffer(targets_iter);

            // Register the result (reach_buffer has the canonical subset)
            // Clone reach_buffer because register_closure needs to store it
            let dest_subset = self.reach_buffer.clone();
            let dest_id = self.register_closure(dest_subset);

            // Add transition
            let _ = self.dwa.add_transition(sid, lbl, dest_id, edge_weight);

            i = j;
        }
    }
}

fn try_build_singleton_loop_union(nwa: &NWA) -> Option<DWA> {
    if nwa.states.0.is_empty() || nwa.body.start_states.len() != 1 {
        return None;
    }

    let start = nwa.body.start_states[0];
    if start >= nwa.states.len() { return None; }

    if !nwa.states[start].transitions.is_empty() {
        return None;
    }

    // Manual minimal closure for the heuristic
    let mut start_closure = Vec::new();
    // Just BFS locally
    let mut q = VecDeque::new();
    let mut visited = HashMap::new();
    
    visited.insert(start, Weight::all());
    q.push_back(start);
    
    while let Some(u) = q.pop_front() {
        let w_u = visited[&u].clone();
        if u < nwa.states.len() {
            for (v, w_eps) in &nwa.states[u].epsilons {
                let w_new = &w_u & w_eps;
                if !w_new.is_empty() {
                    let entry = visited.entry(*v).or_insert_with(Weight::zeros);
                    if !w_new.is_subset_of(entry) {
                        *entry |= &w_new;
                        q.push_back(*v);
                    }
                }
            }
        }
    }
    for (u, w) in visited {
        start_closure.push((u, w));
    }

    let mut comps: Vec<(NWAStateID, Weight)> = Vec::new();
    for (sid, cw) in start_closure.iter() {
        if *sid == start || is_zero(cw) {
            continue;
        }
        let st = &nwa.states[*sid];

        if !st.epsilons.is_empty() {
            return None;
        }
        for (_lbl, vec_targets) in st.transitions.iter() {
            for (to, _) in vec_targets {
                if *to != *sid {
                    return None;
                }
            }
        }

        if let Some(fw) = &st.final_weight {
            let base = cw & fw;
            if !base.is_empty() {
                comps.push((*sid, base));
            }
        }
    }

    if comps.is_empty() {
        return None;
    }

    for i in 0..comps.len() {
        for j in (i + 1)..comps.len() {
            if !(comps[i].1.clone() & comps[j].1.clone()).is_empty() {
                return None;
            }
        }
    }

    let mut label_to_weight: BTreeMap<Label, Weight> = BTreeMap::new();
    for (sid, base) in &comps {
        let st = &nwa.states[*sid];
        for (lbl, vec_targets) in st.transitions.iter() {
            let mut w_union = Weight::zeros();
            for (_to, w) in vec_targets {
                w_union = w_union | w.clone();
            }
            if !w_union.is_empty() {
                let contrib = base.clone() & w_union;
                if !contrib.is_empty() {
                    let prev = label_to_weight.get(lbl).cloned().unwrap_or_else(Weight::zeros);
                    label_to_weight.insert(*lbl, prev | contrib);
                }
            }
        }
    }

    let mut final_union = Weight::zeros();
    for (_sid, base) in &comps {
        final_union = final_union | base.clone();
    }

    let mut dwa = DWA::new();
    let s0 = dwa.body.start_state;
    if !final_union.is_empty() {
        let _ = dwa.set_final_weight(s0, final_union);
    }
    for (lbl, w) in label_to_weight {
        if !w.is_empty() {
            let _ = dwa.add_transition(s0, lbl, s0, w);
        }
    }

    Some(dwa)
}

impl NWA {
    pub fn determinize_to_dwa2(&self) -> DWA {
        if let Some(dwa) = try_build_singleton_loop_union(self) {
            return dwa;
        }

        const STATE_LIMIT: usize = 250_000; 
        
        if self.states.0.is_empty() {
            return DWA::new();
        }

        crate::debug!(5, "Determinization: Starting...");
        
        let show_pbar = self.states.len() > 10000;
        let mp = if show_pbar { Some(MultiProgress::new()) } else { None };
        let main_pb = mp.as_ref().map(|mp_instance| {
            let pb = mp_instance.add(ProgressBar::new(1));
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] States: \
                         [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );
            pb.set_message("Determinizing NWA");
            pb
        });

        let mut det = Determinizer::new(self);

        // Construct initial start subset
        // Start states have implicit weight ALL.
        let initial_iter = self.body.start_states.iter().map(|&s| (s, Weight::all()));
        det.compute_closure_into_buffer(initial_iter);
        let start_subset = det.reach_buffer.clone();
        
        let start_id = det.register_closure(start_subset);
        det.dwa.body.start_state = start_id;

        let mut processed_count = 0;
        while let Some(sid) = det.queue.pop_front() {
            if det.seen.len() > STATE_LIMIT {
                let timestamp = Local::now().format("%Y%m%d-%H%M%S");
                let filename = format!("nwa_dump_{}.json", timestamp);
                crate::debug!(5, "Determinization state limit ({}) exceeded. Dumping NWA to {} and panicking.", STATE_LIMIT, filename);
                let f = std::fs::File::create(&filename).expect("Unable to create dump file");
                serde_json::to_writer(f, self).expect("Unable to write NWA to file");
                panic!("Determinization aborted after reaching {} states.", STATE_LIMIT);
            }

            if let Some(pb) = &main_pb {
                if processed_count % 100 == 0 {
                    let total_states = det.seen.len();
                    pb.set_length(total_states as u64);
                    pb.set_position(processed_count as u64);
                    pb.set_message(format!("Expanding state {}/{}", processed_count + 1, total_states));
                }
            }

            det.expand_state(sid);
            processed_count += 1;
        }
        if let Some(pb) = main_pb {
            pb.finish_with_message("Determinization complete");
        }

        if DETERMINIZE_DEBUG {
            let rustfst_dwa = self.determinize_to_dwa_with_rustfst();
            crate::debug!(5, "[DETERMINIZE_DEBUG] Comparing custom determinization with rustfst...");
            test_weighted_automata::stochastic_equivalence_test(det.dwa.clone(), rustfst_dwa);
        }

        det.dwa
    }

    // Main entry point
    pub fn determinize(&self) -> DWA {
        self.determinize_to_dwa2()
    }
    
    // Backward compatibility / alternative implementation
    // (Currently pointing to the main implementation, can be removed if desired)
    pub fn _determinize(&self) -> DWA {
        self.determinize_to_dwa2()
    }
}
