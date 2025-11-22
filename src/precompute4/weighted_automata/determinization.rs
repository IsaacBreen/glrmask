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
    // Dense map for epsilon closure
    // dense_weights[u] stores the accumulated weight for state u
    dense_weights: Vec<Weight>,
    // seen_generation[u] tracks if dense_weights[u] is valid for the current pass
    seen_generation: Vec<u32>,
    // The current generation ID
    current_generation: u32,

    // Indices in dense_weights that are currently valid (for iteration)
    touched: Vec<NWAStateID>,

    // Worklist for epsilon closure fixpoint
    queue: VecDeque<NWAStateID>,
    // Membership check for queue to avoid duplicates (using generation logic)
    in_queue_generation: Vec<u32>,
}

impl DeterminizerContext {
    fn new(num_nwa_states: usize) -> Self {
        Self {
            dense_weights: vec![Weight::zeros(); num_nwa_states], // Placeholder values
            seen_generation: vec![0; num_nwa_states],
            current_generation: 0,
            touched: Vec::with_capacity(num_nwa_states),
            queue: VecDeque::with_capacity(128),
            in_queue_generation: vec![0; num_nwa_states],
        }
    }

    /// Prepares the context for a new closure computation.
    /// This is O(1) due to the generation counter.
    fn start_pass(&mut self) {
        self.current_generation = self.current_generation.wrapping_add(1);
        // Handle wrap-around very conservatively (unlikely to happen in one run, but safe)
        if self.current_generation == 0 {
            self.seen_generation.fill(0);
            self.in_queue_generation.fill(0);
            self.current_generation = 1;
        }
        self.touched.clear();
        self.queue.clear();
    }

    #[inline(always)]
    fn set_weight(&mut self, u: NWAStateID, w: Weight) {
        if self.seen_generation[u] != self.current_generation {
            self.dense_weights[u] = w;
            self.seen_generation[u] = self.current_generation;
            self.touched.push(u);
        } else {
            self.dense_weights[u] = w;
        }
    }

    #[inline(always)]
    fn update_weight(&mut self, u: NWAStateID, w_add: &Weight) -> bool {
        if self.seen_generation[u] != self.current_generation {
            self.dense_weights[u] = w_add.clone();
            self.seen_generation[u] = self.current_generation;
            self.touched.push(u);
            true
        } else {
            let old = &self.dense_weights[u];
            // Optimization: Check subset before OR to avoid allocation if not needed.
            // Also, simple bitset operations are behind a mutex cache, so minimize them.
            if !w_add.is_subset_of(old) {
                self.dense_weights[u] = old | w_add;
                true
            } else {
                false
            }
        }
    }

    #[inline(always)]
    fn enqueue_if_new(&mut self, u: NWAStateID) {
        if self.in_queue_generation[u] != self.current_generation {
            self.queue.push_back(u);
            self.in_queue_generation[u] = self.current_generation;
        }
    }

    /// Computes epsilon closure in-place.
    /// Input: `nodes` (list of state-weight pairs).
    /// Output: Returns a canonical Subset (sorted by ID, no empty weights).
    fn compute_closure(&mut self, nwa: &NWA, nodes: &[(NWAStateID, Weight)]) -> Subset {
        self.start_pass();

        // Initial population
        for &(u, ref w) in nodes {
            if u >= nwa.states.len() { continue; }
            if w.is_empty() { continue; }

            if self.update_weight(u, w) {
                self.enqueue_if_new(u);
            }
        }

        // Fixpoint iteration
        while let Some(u) = self.queue.pop_front() {
            // Remove from queue set so it can be re-added if updated again (though for simple closure logic, usually once is enough if topological, but cyclic epsilons exist)
            self.in_queue_generation[u] = 0;

            // Retrieve weight. Clone is cheap (Arc).
            let w_u = self.dense_weights[u].clone();

            for &(v, ref w_eps) in &nwa.states[u].epsilons {
                if v >= nwa.states.len() { continue; }

                let w_trans = &w_u & w_eps;
                if w_trans.is_empty() { continue; }

                if self.update_weight(v, &w_trans) {
                    self.enqueue_if_new(v);
                }
            }
        }

        // Collect result
        // self.touched contains all indices that have a value in this generation.
        // Sort to ensure canonical representation (Subset must be sorted by ID).
        self.touched.sort_unstable();

        let mut result = Vec::with_capacity(self.touched.len());
        for &u in &self.touched {
            // We know it's valid because it's in touched
            let w = &self.dense_weights[u];
            if !w.is_empty() {
                result.push((u, w.clone()));
            }
        }
        result
    }
}

impl NWA {
    pub fn determinize(&self) -> DWA {
        let mut dwa = DWA::new();
        dwa.states.0.clear();

        let n_nwa = self.states.len();
        let mut ctx = DeterminizerContext::new(n_nwa);

        // Buffer for collecting transitions.
        // (Label, Target, Weight).
        // We will sort this ONLY by Label and Target, ignoring Weight to save time.
        let mut trans_buffer: Vec<(Label, NWAStateID, Weight)> = Vec::with_capacity(4096);

        // Buffer for constructing the input for the next closure call
        let mut next_state_buffer: Vec<(NWAStateID, Weight)> = Vec::with_capacity(128);

        // Initial subset
        next_state_buffer.clear();
        for &s in &self.body.start_states {
            next_state_buffer.push((s, Weight::all()));
        }
        let start_subset = ctx.compute_closure(self, &next_state_buffer);

        // Map from canonical Subset to DWA StateID
        // Use HashMap. hashing Vec uses the content.
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
            let from_id = *subset_to_id.get(&subset).unwrap();

            // 1. Compute final weight
            let mut final_weight_acc = Weight::zeros();
            let mut has_final = false;
            for (u, w) in &subset {
                if let Some(fw) = &self.states[*u].final_weight {
                    let contrib = w & fw;
                    if !contrib.is_empty() {
                        if !has_final {
                            final_weight_acc = contrib;
                            has_final = true;
                        } else {
                            final_weight_acc |= &contrib;
                        }
                    }
                }
            }
            if has_final {
                dwa.states[from_id].final_weight = Some(final_weight_acc);
            }

            // 2. Collect transitions
            trans_buffer.clear();
            for (u, w_u) in &subset {
                // Optimization: avoid iterating if transition map is empty
                if self.states[*u].transitions.is_empty() { continue; }

                for (&label, targets) in &self.states[*u].transitions {
                    for &(v, ref w_edge) in targets {
                        // Minimize bitset ops
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

            // 3. Group by (Label, Target)
            // CRITICAL OPTIMIZATION: Do NOT compare Weights in the sort.
            // Comparing RangeSetBlaze is O(N). Comparing integers is O(1).
            trans_buffer.sort_unstable_by(|a, b| {
                let lc = a.0.cmp(&b.0);
                if lc != Ordering::Equal { return lc; }
                a.1.cmp(&b.1)
            });

            let mut i = 0;
            while i < trans_buffer.len() {
                let label = trans_buffer[i].0;

                next_state_buffer.clear();

                let mut j = i;
                while j < trans_buffer.len() {
                    let (l, v, ref w) = trans_buffer[j];
                    if l != label { break; }

                    // Merge duplicate targets
                    // Since we sorted by (Label, Target), duplicates are adjacent
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
                    // Check existence
                    if let Some(&id) = subset_to_id.get(&next_subset) {
                         dwa.add_transition(from_id, label, id, Weight::all()).unwrap();
                    } else {
                        let new_id = dwa.add_state();
                        // Insert clone of subset (Vec) as key
                        subset_to_id.insert(next_subset.clone(), new_id);
                        worklist.push_back(next_subset);
                         dwa.add_transition(from_id, label, new_id, Weight::all()).unwrap();
                    };
                }

                i = j;
            }
        }

        dwa
    }
}