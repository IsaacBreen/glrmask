#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

use super::common::{Label, NWAStateID, Weight};
use super::dwa::DWA;
use super::nwa::NWA;
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

// A canonical subset is a sorted list of (StateID, Weight).
// We ensure it is sorted by StateID and contains no empty weights.
type Subset = Vec<(NWAStateID, Weight)>;

struct DeterminizerContext {
    // Dense map for epsilon closure: index -> accumulated weight
    // We use a dense vector because NWA state IDs are dense 0..N
    dense_weights: Vec<Option<Weight>>,
    // Indices in dense_weights that are currently Some(...) to allow fast clearing
    touched: Vec<NWAStateID>,

    // Worklist for epsilon closure fixpoint
    queue: VecDeque<NWAStateID>,
    // Membership check for queue to avoid duplicates
    in_queue: Vec<bool>,
}

impl DeterminizerContext {
    fn new(num_nwa_states: usize) -> Self {
        Self {
            dense_weights: vec![None; num_nwa_states],
            touched: Vec::with_capacity(num_nwa_states),
            queue: VecDeque::with_capacity(128),
            in_queue: vec![false; num_nwa_states],
        }
    }

    fn reset_closure(&mut self) {
        for &idx in &self.touched {
            self.dense_weights[idx] = None;
            self.in_queue[idx] = false;
        }
        self.touched.clear();
        self.queue.clear();
    }

    /// Computes epsilon closure in-place.
    /// Input: `nodes` (list of state-weight pairs).
    /// Output: Returns a canonical Subset (sorted by ID, no empty weights).
    fn compute_closure(&mut self, nwa: &NWA, nodes: &[(NWAStateID, Weight)]) -> Subset {
        self.reset_closure();

        // Initial population
        for &(u, ref w) in nodes {
            if u >= nwa.states.len() { continue; }
            if w.is_empty() { continue; }

            match &mut self.dense_weights[u] {
                Some(existing) => {
                    let new_w = existing.clone() | w;
                    if new_w != *existing {
                        *existing = new_w;
                        if !self.in_queue[u] {
                            self.queue.push_back(u);
                            self.in_queue[u] = true;
                        }
                    }
                }
                None => {
                    self.dense_weights[u] = Some(w.clone());
                    self.touched.push(u);
                    if !self.in_queue[u] {
                        self.queue.push_back(u);
                        self.in_queue[u] = true;
                    }
                }
            }
        }

        // Fixpoint iteration
        while let Some(u) = self.queue.pop_front() {
            self.in_queue[u] = false;

            // Retrieve weight. We must clone because we can't borrow self while iterating nwa states
            // (conceptually distinct, but Rust borrowing rules).
            // SimpleBitset clone is cheap (Arc).
            let w_u = self.dense_weights[u].as_ref().unwrap().clone();

            for &(v, ref w_eps) in &nwa.states[u].epsilons {
                if v >= nwa.states.len() { continue; }

                let w_trans = &w_u & w_eps;
                if w_trans.is_empty() { continue; }

                match &mut self.dense_weights[v] {
                    Some(existing) => {
                        let combined = existing.clone() | &w_trans;
                        if combined != *existing {
                            *existing = combined;
                            if !self.in_queue[v] {
                                self.queue.push_back(v);
                                self.in_queue[v] = true;
                            }
                        }
                    }
                    None => {
                        self.dense_weights[v] = Some(w_trans);
                        self.touched.push(v);
                        if !self.in_queue[v] {
                            self.queue.push_back(v);
                            self.in_queue[v] = true;
                        }
                    }
                }
            }
        }

        // Collect result
        // self.touched contains all indices that have a value.
        // Sort to ensure canonical representation (Subset must be sorted by ID).
        self.touched.sort_unstable();

        let mut result = Vec::with_capacity(self.touched.len());
        for &u in &self.touched {
            if let Some(w) = &self.dense_weights[u] {
                // Should always be Some if in touched
                if !w.is_empty() {
                    result.push((u, w.clone()));
                }
            }
        }
        result
    }
}

impl NWA {
    pub fn determinize(&self) -> DWA {
        let mut dwa = DWA::new();
        dwa.states.0.clear(); // Clear default start state created by new()

        let n_nwa = self.states.len();
        let mut ctx = DeterminizerContext::new(n_nwa);
        let mut trans_buffer: Vec<(Label, NWAStateID, Weight)> = Vec::with_capacity(1024);
        let mut next_state_buffer: Vec<(NWAStateID, Weight)> = Vec::with_capacity(128);

        // Initial subset: Start states with Weight::all()
        next_state_buffer.clear();
        for &s in &self.body.start_states {
            next_state_buffer.push((s, Weight::all()));
        }
        let start_subset = ctx.compute_closure(self, &next_state_buffer);

        // Map from canonical Subset to DWA StateID
        // HashMap is significantly faster than BTreeMap for large keys if we hash once
        let mut subset_to_id: HashMap<Subset, usize> = HashMap::new();
        let mut worklist: VecDeque<Subset> = VecDeque::new();

        // Create start state
        let start_id = dwa.add_state();
        dwa.body.start_state = start_id;

        if !start_subset.is_empty() {
            subset_to_id.insert(start_subset.clone(), start_id);
            worklist.push_back(start_subset);
        }

        while let Some(subset) = worklist.pop_front() {
            // subset is canonical (sorted by ID)
            let from_id = *subset_to_id.get(&subset).unwrap();

            // 1. Compute final weight for this subset
            let mut final_weight_acc = Weight::zeros();
            for (u, w) in &subset {
                if let Some(fw) = &self.states[*u].final_weight {
                    let contrib = w & fw;
                    if !contrib.is_empty() {
                        final_weight_acc |= &contrib;
                    }
                }
            }
            if !final_weight_acc.is_empty() {
                dwa.states[from_id].final_weight = Some(final_weight_acc);
            }

            // 2. Collect all outgoing transitions into a flat buffer
            trans_buffer.clear();
            for (u, w_u) in &subset {
                for (&label, targets) in &self.states[*u].transitions {
                    for &(v, ref w_edge) in targets {
                        let w_trans = w_u & w_edge;
                        if !w_trans.is_empty() {
                            trans_buffer.push((label, v, w_trans));
                        }
                    }
                }
            }

            if trans_buffer.is_empty() {
                continue;
            }

            // 3. Group by label
            // Sorting allows us to identify groups and merge duplicate targets linearly
            trans_buffer.sort_unstable_by(|a, b| {
                let lc = a.0.cmp(&b.0);
                if lc != Ordering::Equal { return lc; }
                a.1.cmp(&b.1)
            });

            let mut i = 0;
            while i < trans_buffer.len() {
                let label = trans_buffer[i].0;

                next_state_buffer.clear();

                // Process all transitions for this label
                // Since we sorted by (label, v), duplicates of v are adjacent.
                let mut j = i;
                while j < trans_buffer.len() && trans_buffer[j].0 == label {
                    let (_, v, ref w) = trans_buffer[j];

                    // Merge weights for same target v
                    if let Some(last) = next_state_buffer.last_mut() {
                        if last.0 == v {
                            last.1 |= w;
                            j += 1;
                            continue;
                        }
                    }

                    next_state_buffer.push((v, w.clone()));
                    j += 1;
                }

                // Compute epsilon closure for the target set
                let next_subset = ctx.compute_closure(self, &next_state_buffer);

                if !next_subset.is_empty() {
                    let to_id = if let Some(&id) = subset_to_id.get(&next_subset) {
                        id
                    } else {
                        let new_id = dwa.add_state();
                        subset_to_id.insert(next_subset.clone(), new_id);
                        worklist.push_back(next_subset);
                        new_id
                    };

                    // Determinized transitions typically carry no weight (weight is pushed to states/finals)
                    // or carry Weight::all() if the DWA model expects a weight present.
                    dwa.add_transition(from_id, label, to_id, Weight::all()).unwrap();
                }

                i = j;
            }
        }

        dwa
    }
}