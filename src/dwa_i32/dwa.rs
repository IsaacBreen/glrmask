// src/precompute4/weighted_automata/dwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

pub(crate) use super::common::{format_pos_code, Label, StateID, Weight, weight_all};
use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};
use std::ops::{Deref, Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DWABuildError {
    TransitionAlreadyExists { from: StateID, on: Label },
    StateOutOfBounds { state: StateID },
}

impl Display for DWABuildError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAState {
    pub transitions: BTreeMap<Label, StateID>,
    pub final_weight: Option<Weight>,
    pub trans_weights: BTreeMap<Label, Weight>,
}

impl DWAState {
    pub fn get_transition(&self, ch: Label) -> Option<(StateID, &Weight)> {
        self.transitions.get(&ch).and_then(|to| self.trans_weights.get(&ch).map(|w| (*to, w)))
    }
    
    pub fn apply_weight(&mut self, weight: &Weight) {
        if let Some(fw) = &mut self.final_weight { *fw &= weight; if fw.is_empty() { self.final_weight = None; } }
        for w in self.trans_weights.values_mut() { *w &= weight; }
    }

    pub fn clip_weights(&mut self, max: usize) {
        if let Some(fw) = &mut self.final_weight { fw.clip_max(max); if fw.is_empty() { self.final_weight = None; } }
        for w in self.trans_weights.values_mut() { w.clip_max(max); }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWAStates(pub Vec<DWAState>);

impl Index<usize> for DWAStates {
    type Output = DWAState;
    fn index(&self, index: usize) -> &Self::Output { &self.0[index] }
}
impl IndexMut<usize> for DWAStates {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.0[index] }
}
impl Deref for DWAStates {
    type Target = [DWAState];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl DWAStates {
    pub fn len(&self) -> usize { self.0.len() }
    pub fn num_transitions(&self) -> usize { self.0.iter().map(|s| s.transitions.len()).sum() }
    pub fn add_state(&mut self) -> StateID {
        let id = self.0.len(); self.0.push(DWAState::default()); id
    }
    pub fn add_existing_state(&mut self, state: DWAState) -> StateID {
        let id = self.0.len(); self.0.push(state); id
    }
    pub fn copy_state(&mut self, state_id: StateID) -> StateID {
        let state = self[state_id].clone(); self.add_existing_state(state)
    }
    pub fn apply_weight_to_state(&mut self, state_id: StateID, weight: &Weight) {
        self[state_id].apply_weight(weight);
    }
    pub fn apply_weight_to_all_states(&mut self, weight: &Weight) {
        for state in self.0.iter_mut() { state.apply_weight(weight); }
    }
    pub fn clip_weights(&mut self, max: usize) {
        for state in self.0.iter_mut() { state.clip_weights(max); }
    }
    
    /// Find the actual maximum value present in any weight across all states.
    /// Returns None if there are no weights or all weights are empty/ALL.
    pub fn find_actual_max(&self) -> Option<usize> {
        let mut max_val: Option<usize> = None;
        for state in &self.0 {
            if let Some(fw) = &state.final_weight {
                if !fw.is_all_fast() && !fw.is_empty() {
                    if let Some(m) = fw.max_item() {
                        if m != usize::MAX {  // Skip if extends to MAX
                            max_val = Some(max_val.map_or(m, |cur| cur.max(m)));
                        }
                    }
                }
            }
            for w in state.trans_weights.values() {
                if !w.is_all_fast() && !w.is_empty() {
                    if let Some(m) = w.max_item() {
                        if m != usize::MAX {  // Skip if extends to MAX
                            max_val = Some(max_val.map_or(m, |cur| cur.max(m)));
                        }
                    }
                }
            }
        }
        max_val
    }
}

use super::heavy_weight::WeightDimensions;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DWABody {
    pub start_state: StateID,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DWA {
    pub states: DWAStates,
    pub body: DWABody,
    /// Weight space dimensions (num_tokens × num_tsids).
    /// WeightDimensions::TEST if not set.
    pub dims: WeightDimensions,
}

impl Default for DWA {
    fn default() -> Self {
        Self { states: DWAStates::default(), body: DWABody::default(), dims: WeightDimensions::TEST }
    }
}

impl DWA {
    pub fn new() -> Self {
        let mut states = DWAStates::default();
        let start = states.add_state();
        DWA { states, body: DWABody { start_state: start }, dims: WeightDimensions::TEST }
    }
    pub fn new_with_dims(dims: WeightDimensions) -> Self {
        let mut states = DWAStates::default();
        let start = states.add_state();
        DWA { states, body: DWABody { start_state: start }, dims }
    }
    pub fn new_empty() -> Self {
        DWA { states: DWAStates::default(), body: DWABody::default(), dims: WeightDimensions::TEST }
    }
    pub fn new_empty_with_dims(dims: WeightDimensions) -> Self {
        DWA { states: DWAStates::default(), body: DWABody::default(), dims }
    }
    pub fn add_state(&mut self) -> StateID { self.states.add_state() }
    
    /// Get the weight dimensions.
    pub fn dimensions(&self) -> WeightDimensions { self.dims }
    
    /// Set the weight dimensions.
    pub fn set_dimensions(&mut self, dims: WeightDimensions) { self.dims = dims; }
    
    pub fn set_final_weight(&mut self, state: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if state >= self.states.len() { return Err(DWABuildError::StateOutOfBounds { state }); }
        self.states[state].final_weight = Some(weight); Ok(())
    }
    pub fn add_transition(&mut self, from: StateID, on: Label, to: StateID, weight: Weight) -> Result<(), DWABuildError> {
        if from >= self.states.len() { return Err(DWABuildError::StateOutOfBounds { state: from }); }
        if to >= self.states.len() { return Err(DWABuildError::StateOutOfBounds { state: to }); }
        if self.states[from].transitions.contains_key(&on) { return Err(DWABuildError::TransitionAlreadyExists { from, on }); }
        self.states[from].transitions.insert(on, to);
        self.states[from].trans_weights.insert(on, weight); Ok(())
    }
    
    pub fn eval_word_weight(&self, word: &[Label]) -> Weight {
        if self.states.0.is_empty() { return Weight::zeros(); }
        let mut s = self.body.start_state;
        let mut acc = weight_all();

        if s < self.states.len() {
        } else { return Weight::zeros(); }

        for &ch in word {
            if s >= self.states.len() { return Weight::zeros(); }
            if let Some((t, w)) = self.states[s].get_transition(ch) {
                acc &= w; if acc.is_empty() { return Weight::zeros(); }
                s = t;
            } else { return Weight::zeros(); }
        }
        if s >= self.states.len() { return Weight::zeros(); }
        match &self.states[s].final_weight {
            Some(fw) => { let res = &acc & fw; if res.is_empty() { Weight::zeros() } else { res } }
            None => Weight::zeros(),
        }
    }

    pub fn apply_weight_inplace(&mut self, weight: &Weight) {
        if self.body.start_state < self.states.len() {
            let s = &mut self.states[self.body.start_state];
            s.apply_weight(weight);
        }
    }
    
    /// Trim weights by clipping them to [0, actual_max] where actual_max is
    /// the maximum value that appears in any non-ALL weight across the DWA.
    /// 
    /// This removes unnecessary range extensions that go up to usize::MAX
    /// when no actual weight values exist beyond the true maximum.
    /// 
    /// Returns true if any weights were modified, false otherwise.
    pub fn trim_weights(&mut self) -> bool {
        // Find the actual maximum value across all weights
        let actual_max = match self.states.find_actual_max() {
            Some(max) => max,
            None => return false, // No non-ALL, non-empty weights to trim
        };
        
        // Count ranges before trimming
        let ranges_before = self.num_ranges();
        
        // Clip all weights to [0, actual_max]
        self.states.clip_weights(actual_max);
        
        // Count ranges after trimming
        let ranges_after = self.num_ranges();
        
        let changed = ranges_before != ranges_after;
        if changed {
            crate::debug!(5, "trim_weights: clipped to max={}, ranges {} -> {} ({:.1}% reduction)",
                actual_max, ranges_before, ranges_after,
                100.0 * (1.0 - ranges_after as f64 / ranges_before as f64));
        }
        
        changed
    }
    
    /// Trim weights to a specific domain maximum.
    /// 
    /// This is useful when you know the domain_max externally (e.g., from vocab size).
    /// Clips all weights to [0, domain_max].
    /// 
    /// Returns true if any weights were modified, false otherwise.
    pub fn trim_weights_to_domain(&mut self, domain_max: usize) -> bool {
        let ranges_before = self.num_ranges();
        self.states.clip_weights(domain_max);
        let ranges_after = self.num_ranges();
        
        let changed = ranges_before != ranges_after;
        if changed {
            crate::debug!(5, "trim_weights_to_domain: clipped to max={}, ranges {} -> {} ({:.1}% reduction)",
                domain_max, ranges_before, ranges_after,
                100.0 * (1.0 - ranges_after as f64 / ranges_before as f64));
        }
        
        changed
    }

    pub fn stats(&self) -> String {
        format!("States: {}, Transitions: {}", self.states.len(), self.states.iter().map(|s| s.transitions.len()).sum::<usize>())
    }

    /// Computes the average path length across all paths from start to final states.
    /// 
    /// Returns None if the DWA is cyclic (infinite paths) or has no paths.
    /// 
    /// For an acyclic DWA, this computes:
    /// - total_paths: number of distinct paths from start to any final state  
    /// - total_length: sum of lengths of all paths
    /// - average = total_length / total_paths
    /// 
    /// Uses O(n + m) time via topological order + dynamic programming.
    pub fn average_path_length(&self) -> Option<f64> {
        let n = self.states.len();
        if n == 0 { return None; }
        if self.is_cyclic() { return None; }
        
        let start = self.body.start_state;
        if start >= n { return None; }
        
        // Build reverse adjacency for topological sort
        let mut in_degree = vec![0usize; n];
        for u in 0..n {
            for &v in self.states[u].transitions.values() {
                if v < n { in_degree[v] += 1; }
            }
        }
        
        // Topological sort (Kahn's algorithm)
        let mut topo = Vec::with_capacity(n);
        let mut queue = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 { queue.push_back(i); }
        }
        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &v in self.states[u].transitions.values() {
                if v < n {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 { queue.push_back(v); }
                }
            }
        }
        if topo.len() != n { return None; } // Shouldn't happen if is_cyclic() returned false
        
        // Forward pass: count paths and sum of lengths reaching each state
        // paths_to[u] = number of paths from start to u
        // length_sum_to[u] = sum of path lengths from start to u (summed over all paths)
        // Use u128 to avoid overflow (path counts can be astronomical)
        let mut paths_to = vec![0u128; n];
        let mut length_sum_to = vec![0u128; n];
        
        paths_to[start] = 1;
        length_sum_to[start] = 0;
        
        for &u in &topo {
            if paths_to[u] == 0 { continue; } // Not reachable from start
            
            for &v in self.states[u].transitions.values() {
                if v < n {
                    // Each path to u contributes one path to v with length+1
                    paths_to[v] += paths_to[u];
                    // Sum of lengths to v: sum of (length_to_u + 1) for each path
                    // = length_sum_to[u] + paths_to[u] (one more step per path)
                    length_sum_to[v] += length_sum_to[u] + paths_to[u];
                }
            }
        }
        
        // Sum up paths and lengths to final states
        let mut total_paths: u128 = 0;
        let mut total_length: u128 = 0;
        
        for u in 0..n {
            if self.states[u].final_weight.is_some() && paths_to[u] > 0 {
                total_paths += paths_to[u];
                total_length += length_sum_to[u];
            }
        }
        
        if total_paths == 0 { return None; }
        
        Some(total_length as f64 / total_paths as f64)
    }

    /// Debug version of average_path_length that returns intermediate values.
    /// Note: Returns u64 values, clamped from u128 for display purposes.
    /// The avg returned is the true f64 average computed from u128 values.
    pub fn average_path_length_debug(&self) -> Option<(u64, u64, f64)> {
        let n = self.states.len();
        if n == 0 { return None; }
        if self.is_cyclic() { return None; }
        
        let start = self.body.start_state;
        if start >= n { return None; }
        
        let mut in_degree = vec![0usize; n];
        for u in 0..n {
            for &v in self.states[u].transitions.values() {
                if v < n { in_degree[v] += 1; }
            }
        }
        
        let mut topo = Vec::with_capacity(n);
        let mut queue = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 { queue.push_back(i); }
        }
        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &v in self.states[u].transitions.values() {
                if v < n {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 { queue.push_back(v); }
                }
            }
        }
        if topo.len() != n { return None; }
        
        // Use u128 to avoid overflow (path counts can be astronomical)
        let mut paths_to = vec![0u128; n];
        let mut length_sum_to = vec![0u128; n];
        
        paths_to[start] = 1;
        length_sum_to[start] = 0;
        
        for &u in &topo {
            if paths_to[u] == 0 { continue; }
            
            for &v in self.states[u].transitions.values() {
                if v < n {
                    paths_to[v] += paths_to[u];
                    length_sum_to[v] += length_sum_to[u] + paths_to[u];
                }
            }
        }
        
        let mut total_paths: u128 = 0;
        let mut total_length: u128 = 0;
        
        for u in 0..n {
            if self.states[u].final_weight.is_some() && paths_to[u] > 0 {
                total_paths += paths_to[u];
                total_length += length_sum_to[u];
            }
        }
        
        if total_paths == 0 { return None; }
        
        // Clamp to u64 for return value (for display), but avg uses full u128 precision
        let total_paths_u64 = if total_paths > u64::MAX as u128 { u64::MAX } else { total_paths as u64 };
        let total_length_u64 = if total_length > u64::MAX as u128 { u64::MAX } else { total_length as u64 };
        
        Some((total_paths_u64, total_length_u64, total_length as f64 / total_paths as f64))
    }

    /// Sample paths uniformly at random from an acyclic DWA.
    /// 
    /// Each path from start to a final state has equal probability of being sampled.
    /// Returns a vector of paths, where each path is a sequence of (label, next_state) pairs.
    /// 
    /// # Panics
    /// Panics if the DWA is cyclic (use `is_cyclic()` to check first).
    /// 
    /// # Algorithm
    /// 1. Count paths from each state to final states (backward pass)
    /// 2. For each sample, walk from start choosing transitions proportionally to path counts
    /// 
    /// The path count from state s is: paths_to_final[s] = (1 if s is final else 0) + sum of paths_to_final[t] for each transition s -> t
    pub fn sample_paths(&self, num_samples: usize, rng: &mut impl rand::Rng) -> Vec<Vec<(Label, StateID)>> {
        let n = self.states.len();
        if n == 0 { return vec![]; }
        assert!(!self.is_cyclic(), "sample_paths requires an acyclic DWA");
        
        let start = self.body.start_state;
        if start >= n { return vec![]; }
        
        // Build reverse adjacency for topological sort
        let mut in_degree = vec![0usize; n];
        for u in 0..n {
            for &v in self.states[u].transitions.values() {
                if v < n { in_degree[v] += 1; }
            }
        }
        
        // Topological sort (Kahn's algorithm)
        let mut topo = Vec::with_capacity(n);
        let mut queue = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 { queue.push_back(i); }
        }
        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &v in self.states[u].transitions.values() {
                if v < n {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 { queue.push_back(v); }
                }
            }
        }
        assert_eq!(topo.len(), n, "Topological sort failed - DWA is cyclic?");
        
        // Backward pass: count paths from each state to any final state
        // paths_from[u] = number of paths from u to any final state
        // Use u128 to avoid overflow
        let mut paths_from = vec![0u128; n];
        
        // Process in reverse topological order (leaves first)
        for &u in topo.iter().rev() {
            // If this state is final, that's one path (ending here)
            if self.states[u].final_weight.is_some() {
                paths_from[u] += 1;
            }
            // Add paths through each outgoing transition
            for &v in self.states[u].transitions.values() {
                if v < n {
                    paths_from[u] += paths_from[v];
                }
            }
        }
        
        let total_paths = paths_from[start];
        if total_paths == 0 { return vec![]; }
        
        // Sample paths
        // For uniform sampling with u128 counts, we use probability sampling:
        // probability of choosing successor s = paths_from[s] / current_paths
        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            let mut path = Vec::new();
            let mut current = start;
            
            loop {
                // Collect outgoing transitions with positive path counts
                let mut choices: Vec<(Label, StateID, u128)> = Vec::new();
                for (&label, &next) in &self.states[current].transitions {
                    if next < n && paths_from[next] > 0 {
                        choices.push((label, next, paths_from[next]));
                    }
                }
                
                // For uniform path sampling, we use paths_from[current] to determine probabilities.
                // paths_from[current] = (1 if current is final) + sum(paths_from[successor])
                // So:
                // - Probability of ending here (if final) = 1 / paths_from[current]
                // - Probability of going to successor s = paths_from[s] / paths_from[current]
                let current_paths = paths_from[current];
                if current_paths == 0 {
                    // Dead end
                    break;
                }
                
                // Can we end here?
                let can_end = self.states[current].final_weight.is_some();
                let end_prob = if can_end { 1.0 / current_paths as f64 } else { 0.0 };
                
                // Choose: end here or continue through a transition
                // Use floating point for probability sampling to handle u128 values
                let roll: f64 = rng.gen();
                
                if roll < end_prob {
                    // End path here
                    break;
                } else {
                    // Continue through a transition
                    // Scale roll to [0, 1) over transition probabilities
                    let trans_roll = if end_prob >= 1.0 { 0.0 } else { (roll - end_prob) / (1.0 - end_prob) };
                    
                    // Sum of transition probabilities = (current_paths - (1 if can_end else 0)) / current_paths
                    let trans_paths = current_paths - if can_end { 1 } else { 0 };
                    
                    let mut cumulative = 0u128;
                    let mut selected = None;
                    for &(label, next, count) in &choices {
                        cumulative += count;
                        if trans_roll < (cumulative as f64 / trans_paths as f64) {
                            selected = Some((label, next));
                            break;
                        }
                    }
                    
                    if let Some((label, next)) = selected {
                        path.push((label, next));
                        current = next;
                    } else {
                        // Shouldn't happen in normal cases
                        // Fall back to picking the last choice
                        if let Some(&(label, next, _)) = choices.last() {
                            path.push((label, next));
                            current = next;
                        } else {
                            break;
                        }
                    }
                }
            }
            
            samples.push(path);
        }
        
        samples
    }

    /// Returns the total number of distinct paths from start to any final state.
    /// Returns None if the DWA is cyclic (infinite paths) or empty.
    /// Note: returns u128 to handle astronomical path counts without overflow.
    pub fn count_paths(&self) -> Option<u128> {
        let n = self.states.len();
        if n == 0 { return None; }
        if self.is_cyclic() { return None; }
        
        let start = self.body.start_state;
        if start >= n { return None; }
        
        // Build reverse adjacency for topological sort
        let mut in_degree = vec![0usize; n];
        for u in 0..n {
            for &v in self.states[u].transitions.values() {
                if v < n { in_degree[v] += 1; }
            }
        }
        
        // Topological sort
        let mut topo = Vec::with_capacity(n);
        let mut queue = std::collections::VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 { queue.push_back(i); }
        }
        while let Some(u) = queue.pop_front() {
            topo.push(u);
            for &v in self.states[u].transitions.values() {
                if v < n {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 { queue.push_back(v); }
                }
            }
        }
        if topo.len() != n { return None; }
        
        // Backward pass: count paths from each state to any final state
        let mut paths_from = vec![0u128; n];
        for &u in topo.iter().rev() {
            if self.states[u].final_weight.is_some() {
                paths_from[u] += 1;
            }
            for &v in self.states[u].transitions.values() {
                if v < n {
                    paths_from[u] += paths_from[v];
                }
            }
        }
        
        if paths_from[start] == 0 { return None; }
        Some(paths_from[start])
    }

    /// Estimates average path length by sampling.
    /// Returns None if the DWA is cyclic or has no paths.
    pub fn estimate_average_path_length(&self, num_samples: usize) -> Option<f64> {
        if self.is_cyclic() { return None; }
        if self.count_paths()? == 0 { return None; }
        
        let mut rng = rand::thread_rng();
        let paths = self.sample_paths(num_samples, &mut rng);
        
        if paths.is_empty() { return None; }
        
        let total_length: usize = paths.iter().map(|p| p.len()).sum();
        Some(total_length as f64 / paths.len() as f64)
    }

    /// Counts the total number of ranges across all weights in this DWA.
    /// This includes final weights and transition weights.
    /// Note: If the same weight object appears multiple times, its ranges are counted each time.
    pub fn num_ranges(&self) -> usize {
        let mut total = 0;
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                total += fw.num_ranges();
            }
            for w in state.trans_weights.values() {
                total += w.num_ranges();
            }
        }
        total
    }

    /// Counts the total number of ranges across unique (interned) weights in this DWA.
    /// If the same interned weight appears multiple times, it is only counted once.
    pub fn num_ranges_interned(&self) -> usize {
        use std::collections::HashSet;
        
        // Track unique weights by their intern ID
        let mut seen: HashSet<usize> = HashSet::new();
        let mut total = 0;
        
        let mut process_weight = |w: &Weight| {
            // Get the intern ID as a unique identifier
            let ptr = w.intern_id();
            if seen.insert(ptr) {
                total += w.num_ranges();
            }
        };
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                process_weight(fw);
            }
            for w in state.trans_weights.values() {
                process_weight(w);
            }
        }
        total
    }
    
    /// Analyze weight distribution to understand range fragmentation
    pub fn analyze_weights(&self) {
        use std::collections::HashMap;
        
        // Collect all unique weights
        let mut weight_usage: HashMap<usize, (usize, usize)> = HashMap::new(); // intern_id -> (count, num_ranges)
        let mut range_histogram: HashMap<usize, usize> = HashMap::new(); // num_ranges -> count
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                let ptr = fw.intern_id();
                let entry = weight_usage.entry(ptr).or_insert((0, fw.num_ranges()));
                entry.0 += 1;
                *range_histogram.entry(fw.num_ranges()).or_insert(0) += 1;
            }
            for w in state.trans_weights.values() {
                let ptr = w.intern_id();
                let entry = weight_usage.entry(ptr).or_insert((0, w.num_ranges()));
                entry.0 += 1;
                *range_histogram.entry(w.num_ranges()).or_insert(0) += 1;
            }
        }
        
        let unique_weights = weight_usage.len();
        let total_usages: usize = weight_usage.values().map(|(c, _)| c).sum();
        let total_ranges_unique: usize = weight_usage.values().map(|(_, r)| r).sum();
        
        crate::debug!(6, "DWA Weight Analysis:");
        crate::debug!(6, "  Unique weights: {}", unique_weights);
        crate::debug!(6, "  Total weight usages: {}", total_usages);
        crate::debug!(6, "  Total ranges (unique): {}", total_ranges_unique);
        crate::debug!(6, "  Avg ranges per unique weight: {:.2}", total_ranges_unique as f64 / unique_weights as f64);
        
        let mut ranges: Vec<_> = range_histogram.into_iter().collect();
        ranges.sort();
        crate::debug!(6, "  Range distribution:");
        for (num_ranges, count) in ranges.iter().take(20) {
            crate::debug!(6, "    {} ranges: {} weights", num_ranges, count);
        }
        if ranges.len() > 20 {
            crate::debug!(6, "    ... and {} more categories", ranges.len() - 20);
        }
    }

    pub fn is_cyclic(&self) -> bool {
        let n = self.states.len();
        if n == 0 { return false; }
        
        // 0: unvisited, 1: visiting, 2: visited
        let mut color = vec![0u8; n];
        
        for i in 0..n {
            if color[i] == 0 {
                if self.is_cyclic_dfs(i, &mut color) {
                    return true;
                }
            }
        }
        false
    }
    
    fn is_cyclic_dfs(&self, u: usize, color: &mut [u8]) -> bool {
        color[u] = 1;
        
        for &v in self.states[u].transitions.values() {
            if v >= self.states.len() { continue; }
            if color[v] == 1 {
                return true;
            }
            if color[v] == 0 {
                if self.is_cyclic_dfs(v, color) {
                    return true;
                }
            }
        }
        
        color[u] = 2;
        false
    }

    pub fn optimize_for_visualization(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }


        let start = self.body.start_state;
        if start >= n {
            return;
        }

        let mut forward: Vec<Weight> = vec![Weight::zeros(); n];
        forward[start] = weight_all();

        let mut changed = true;
        while changed {
            changed = false;
            for u in 0..n {
                let fu = forward[u].clone();
                if fu.is_empty() {
                    continue;
                }

                let state = &self.states[u];
                for (lbl, &v) in &state.transitions {
                    if v >= n {
                        continue;
                    }
                    let w = state
                        .trans_weights
                        .get(lbl)
                        .cloned()
                        .unwrap_or_else(weight_all);
                    let mut flow = fu.clone();
                    flow &= &w;
                    if !flow.is_subset_of(&forward[v]) {
                        forward[v] |= &flow;
                        changed = true;
                    }
                }
            }
        }

        // 2. Backward tokens: for each state s, tokens that can go from s to some
        // final state while satisfying all transition, state, and final weights.
        let mut backward: Vec<Weight> = vec![Weight::zeros(); n];
        for s in 0..n {
            if let Some(fw) = &self.states[s].final_weight {
                backward[s] |= fw;
            }
        }

        changed = true;
        while changed {
            changed = false;
            for u in (0..n).rev() {
                let mut bu_new = backward[u].clone();
                let state = &self.states[u];
                for (lbl, &v) in &state.transitions {
                    if v >= n {
                        continue;
                    }
                    let w = state
                        .trans_weights
                        .get(lbl)
                        .cloned()
                        .unwrap_or_else(weight_all);
                    let contribution = &w & &backward[v];
                    if !contribution.is_subset_of(&bu_new) {
                        bu_new |= &contribution;
                    }
                }
                if !bu_new.is_subset_of(&backward[u]) {
                    backward[u] |= &bu_new;
                    changed = true;
                }
            }
        }

        // 3. Apply trimming to states and transitions.
        for s in 0..n {
            // Final weights: tokens must be reachable from the start.
            if let Some(fw) = &mut self.states[s].final_weight {
                *fw &= &forward[s];
                if fw.is_empty() {
                    self.states[s].final_weight = None;
                }
            }

            // Transitions: w_new = w & forward[u] & backward[v].
            let labels: Vec<Label> = self.states[s].transitions.keys().copied().collect();
            for lbl in labels {
                let to = match self.states[s].transitions.get(&lbl) {
                    Some(&t) => t,
                    None => continue,
                };
                if to >= n {
                    self.states[s].transitions.remove(&lbl);
                    self.states[s].trans_weights.remove(&lbl);
                    continue;
                }

                let old_w = self.states[s]
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    .unwrap_or_else(weight_all);

                let mut new_w = old_w;
                new_w &= &forward[s];
                new_w &= &backward[to];

                if new_w.is_empty() {
                    self.states[s].transitions.remove(&lbl);
                    self.states[s].trans_weights.remove(&lbl);
                } else if let Some(w_mut) = self.states[s].trans_weights.get_mut(&lbl) {
                    *w_mut = new_w;
                } else {
                    self.states[s].trans_weights.insert(lbl, new_w);
                }
            }

            // Default transitions: weights that exist without an explicit target.
            // We treat these as staying in state `s` and narrow them using the
            // same forward/backward information as for state weights.
            let default_labels: Vec<Label> = self.states[s]
                .trans_weights
                .keys()
                .filter(|lbl| !self.states[s].transitions.contains_key(lbl))
                .copied()
                .collect();

            for lbl in default_labels {
                let old_w = self.states[s]
                    .trans_weights
                    .get(&lbl)
                    .cloned()
                    .unwrap_or_else(weight_all);

                let mut new_w = old_w;
                new_w &= &forward[s];
                new_w &= &backward[s];

                if new_w.is_empty() {
                    self.states[s].trans_weights.remove(&lbl);
                } else if let Some(w_mut) = self.states[s].trans_weights.get_mut(&lbl) {
                    *w_mut = new_w;
                }
            }
        }
    }
}

impl Display for DWA {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight { writeln!(f, "    final_weight: {}", w)?; }
            for (on, to) in &state.transitions {
                let w = state.trans_weights.get(on).cloned().unwrap_or_else(weight_all);
                if w.is_all_fast() {
                    writeln!(f, "    {} -> {}", format_pos_code(*on), to)?;
                } else {
                    writeln!(f, "    {} -> {} (weight: {})", format_pos_code(*on), to, w)?;
                }
            }
        }
        Ok(())
    }
}

impl DWA {
    /// Export the DWA to a JSON-serializable format for Python analysis
    pub fn to_json_value(&self) -> serde_json::Value {
        use serde_json::{json, Map, Value};
        
        // Helper to convert Weight to JSON representation
        fn weight_to_json(w: &Weight) -> Value {
            if w.is_all_fast() {
                json!({"is_all": true})
            } else if w.is_empty() {
                json!({"is_empty": true})
            } else {
                // Export as ranges
                let rsb = w.to_rsb();
                let ranges: Vec<(usize, usize)> = rsb.ranges()
                    .map(|r| (*r.start(), *r.end()))
                    .collect();
                json!({
                    "ranges": ranges,
                    "len": w.len()
                })
            }
        }
        
        let states: Vec<Value> = self.states.0.iter().enumerate().map(|(id, state)| {
            let mut state_obj = Map::new();
            state_obj.insert("id".to_string(), json!(id));
            
            // Transitions as list of {label, target, weight}
            let transitions: Vec<Value> = state.transitions.iter().map(|(label, target)| {
                let weight = state.trans_weights.get(label)
                    .cloned()
                    .unwrap_or_else(weight_all);
                json!({
                    "label": label,
                    "target": target,
                    "weight": weight_to_json(&weight)
                })
            }).collect();
            state_obj.insert("transitions".to_string(), json!(transitions));
            
            // Final weight
            if let Some(ref fw) = state.final_weight {
                state_obj.insert("final_weight".to_string(), weight_to_json(fw));
            }
            
            Value::Object(state_obj)
        }).collect();
        
        json!({
            "start_state": self.body.start_state,
            "num_states": self.states.len(),
            "num_transitions": self.states.num_transitions(),
            "states": states
        })
    }
    
    /// Export the DWA to a JSON file
    pub fn export_to_json_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json_value = self.to_json_value();
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &json_value)?;
        Ok(())
    }
}
