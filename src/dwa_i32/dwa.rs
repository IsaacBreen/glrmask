// src/precompute4/weighted_automata/dwa.rs

#![allow(dead_code)]
#![allow(clippy::needless_borrow)]

pub(crate) use super::common::{format_pos_code, Label, StateID, Weight};
use std::collections::{BTreeMap, HashMap, HashSet};
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

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
        if let Some(fw) = &mut self.final_weight {
            fw.clip_to_max(max);
            if fw.is_empty() {
                self.final_weight = None;
            }
        }
        for w in self.trans_weights.values_mut() {
            w.clip_to_max(max);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    pub fn find_actual_max(&self) -> (Option<usize>, bool) {
        let mut max_val: Option<usize> = None;
        let mut has_unbounded = false;
        for state in &self.0 {
            if let Some(fw) = &state.final_weight {
                if fw.is_empty() {
                    continue;
                }
                if fw.is_all_fast() {
                    has_unbounded = true;
                } else if let Some(m) = fw.max_item() {
                    if m == usize::MAX {
                        has_unbounded = true;
                    } else {
                        max_val = Some(max_val.map_or(m, |cur| cur.max(m)));
                    }
                }
            }
            for w in state.trans_weights.values() {
                if w.is_empty() {
                    continue;
                }
                if w.is_all_fast() {
                    has_unbounded = true;
                } else if let Some(m) = w.max_item() {
                    if m == usize::MAX {  // Skip if extends to MAX
                        has_unbounded = true;
                    } else {
                        max_val = Some(max_val.map_or(m, |cur| cur.max(m)));
                    }
                }
            }
        }
        (max_val, has_unbounded)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DWABody {
    pub start_state: StateID,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DWA {
    pub states: DWAStates,
    pub body: DWABody,
}

#[derive(Debug, Clone)]
pub struct DWAStats {
    pub states: usize,
    pub transitions: usize,
    pub unique_state_pairs: usize,
    pub ranges: usize,
    pub ranges_interned: usize,
    pub transition_multiplicity_hist: BTreeMap<usize, usize>,
}

impl Display for DWAStats {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "states={}, transitions={}, unique_state_pairs={}, ranges={}, ranges_interned={}, transition_multiplicity_hist={:?}",
            self.states,
            self.transitions,
            self.unique_state_pairs,
            self.ranges,
            self.ranges_interned,
            self.transition_multiplicity_hist
        )
    }
}

impl DWA {
    pub fn new() -> Self {
        let mut states = DWAStates::default();
        let start = states.add_state();
        DWA { states, body: DWABody { start_state: start } }
    }
    pub fn new_empty() -> Self {
        DWA { states: DWAStates::default(), body: DWABody::default() }
    }
    pub fn add_state(&mut self) -> StateID { self.states.add_state() }
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
        let mut acc = Weight::all();

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
        let (actual_max, has_unbounded) = match self.states.find_actual_max() {
            (Some(max), has_unbounded) => (max, has_unbounded),
            (None, _) => return false, // No non-ALL, non-empty weights to trim
        };

        if !has_unbounded {
            return false;
        }

        let debug_ranges = crate::r#macro::is_debug_level_enabled(5);
        
        // Count ranges before trimming
        let ranges_before = if debug_ranges { self.num_ranges() } else { 0 };
        
        // Clip all weights to [0, actual_max]
        self.states.clip_weights(actual_max);
        
        // Count ranges after trimming
        let ranges_after = if debug_ranges { self.num_ranges() } else { 0 };
        
        let changed = if debug_ranges {
            ranges_before != ranges_after
        } else {
            has_unbounded
        };
        if changed && debug_ranges {
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
        let debug_ranges = crate::r#macro::is_debug_level_enabled(5);
        let ranges_before = if debug_ranges { self.num_ranges() } else { 0 };
        self.states.clip_weights(domain_max);
        let ranges_after = if debug_ranges { self.num_ranges() } else { 0 };
        
        let changed = if debug_ranges { ranges_before != ranges_after } else { true };
        if changed && debug_ranges {
            crate::debug!(5, "trim_weights_to_domain: clipped to max={}, ranges {} -> {} ({:.1}% reduction)",
                domain_max, ranges_before, ranges_after,
                100.0 * (1.0 - ranges_after as f64 / ranges_before as f64));
        }
        
        changed
    }

    pub fn stats(&self) -> DWAStats {
        let mut transition_multiplicity_hist: BTreeMap<usize, usize> = BTreeMap::new();
        let mut unique_state_pairs = 0usize;
        let mut dst_counts: HashMap<StateID, usize> = HashMap::new();
        for state in self.states.0.iter() {
            dst_counts.clear();
            for &dst in state.transitions.values() {
                *dst_counts.entry(dst).or_insert(0) += 1;
            }
            unique_state_pairs += dst_counts.len();
            for count in dst_counts.values() {
                *transition_multiplicity_hist.entry(*count).or_insert(0) += 1;
            }
        }
        DWAStats {
            states: self.states.len(),
            transitions: self.states.num_transitions(),
            unique_state_pairs,
            ranges: self.num_ranges(),
            ranges_interned: self.num_ranges_interned(),
            transition_multiplicity_hist,
        }
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

    /// Counts the total number of ranges across unique interned RangeSets in this DWA.
    /// 
    /// This properly handles multi-level interning:
    /// - For RangeSet weights: counts the interned Arc<RangeSetBlaze>
    /// - For Factorized weights: counts unique RangeSets in all pairs
    /// - For RangeMap weights: counts unique RangeSets in all map values
    /// 
    /// If the same Arc<RangeSetBlaze> appears multiple times (across weights or
    /// within a weight's structure), it is only counted once.
    ///
    /// For RangeMap weights, we also count the outer RangeMapBlaze range entries
    /// (deduped by Arc<RangeMapWeight> pointer) since these represent the outer
    /// dimension fragmentation.
    pub fn num_ranges_interned(&self) -> usize {
        use crate::datastructures::AbstractWeight;
        use std::collections::HashSet;
        use std::sync::Arc;

        let mut seen_weight_ptrs: HashSet<usize> = HashSet::new();
        let mut seen_rangeset_ptrs: HashSet<usize> = HashSet::new();
        let mut total_outer_ranges = 0usize;
        let mut total_inner_ranges = 0usize;

        let mut process_weight = |w: &Weight| {
            match w {
                AbstractWeight::RangeSet(rsb) => {
                    let ptr = Arc::as_ptr(&rsb.inner) as usize;
                    if seen_rangeset_ptrs.insert(ptr) {
                        total_inner_ranges += rsb.ranges_len();
                    }
                }
                AbstractWeight::Factorized(fw) => {
                    for (tsid_set, token_set) in &fw.pairs {
                        let ptr1 = Arc::as_ptr(&tsid_set.inner) as usize;
                        let ptr2 = Arc::as_ptr(&token_set.inner) as usize;
                        if seen_rangeset_ptrs.insert(ptr1) {
                            total_inner_ranges += tsid_set.ranges_len();
                        }
                        if seen_rangeset_ptrs.insert(ptr2) {
                            total_inner_ranges += token_set.ranges_len();
                        }
                    }
                }
                AbstractWeight::RangeMap(rm) => {
                    let weight_ptr = Arc::as_ptr(rm) as usize;
                    if seen_weight_ptrs.insert(weight_ptr) {
                        // Count outer RangeMapBlaze range entries (once per unique weight)
                        total_outer_ranges += rm.map.range_values().count();
                    }
                    // Count inner RangeSets (deduped across ALL weights)
                    for (_, tsid_set) in rm.map.range_values() {
                        let ptr = Arc::as_ptr(&tsid_set.inner) as usize;
                        if seen_rangeset_ptrs.insert(ptr) {
                            total_inner_ranges += tsid_set.ranges_len();
                        }
                    }
                }
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

        total_outer_ranges + total_inner_ranges
    }

    /// Analyze weight distribution to understand range fragmentation
    pub fn analyze_weights(&self) {
        use std::collections::HashMap;
        // Collect all unique weights
        let mut weight_usage: HashMap<Weight, (usize, usize)> = HashMap::new(); // weight -> (count, num_ranges)
        let mut range_histogram: HashMap<usize, usize> = HashMap::new(); // num_ranges -> count
        
        for state in &self.states.0 {
            if let Some(fw) = &state.final_weight {
                let entry = weight_usage
                    .entry(fw.clone())
                    .or_insert((0, fw.num_ranges()));
                entry.0 += 1;
                *range_histogram.entry(fw.num_ranges()).or_insert(0) += 1;
            }
            for w in state.trans_weights.values() {
                let entry = weight_usage
                    .entry(w.clone())
                    .or_insert((0, w.num_ranges()));
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

    /// Analyze RangeMapWeight structure for optimization opportunities.
    /// 
    /// For each unique RangeMapWeight, examines:
    /// - Number of outer entries (token ranges)
    /// - Adjacent entries with different tsid_sets and how they differ
    /// - Potential range reduction from normalizing tsid_sets
    pub fn analyze_rangemap_weights(&self) {
        use crate::datastructures::abstract_weight::AbstractWeight;
        use crate::datastructures::rangemap_weight::RangeMapWeight;
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut unique_rms: HashMap<usize, (&Arc<RangeMapWeight>, usize)> = HashMap::new(); // arc_ptr -> (weight, usage_count)
        
        for state in &self.states.0 {
            let process = |w: &Weight| {
                if let AbstractWeight::RangeMap(rm) = w {
                    let key = Arc::as_ptr(rm) as usize;
                    (key, rm.clone())
                } else {
                    (0, Arc::new(RangeMapWeight::from_map(Default::default(), 0)))
                }
            };
            if let Some(fw) = &state.final_weight {
                if let AbstractWeight::RangeMap(rm) = fw {
                    let key = Arc::as_ptr(rm) as usize;
                    unique_rms.entry(key).or_insert((rm, 0)).1 += 1;
                }
            }
            for w in state.trans_weights.values() {
                if let AbstractWeight::RangeMap(rm) = w {
                    let key = Arc::as_ptr(rm) as usize;
                    unique_rms.entry(key).or_insert((rm, 0)).1 += 1;
                }
            }
        }

        if unique_rms.is_empty() {
            eprintln!("WEIGHT_DIAG: No RangeMapWeights found");
            return;
        }

        let total_weights = unique_rms.len();
        let mut total_outer_ranges = 0usize;
        let mut total_inner_ranges = 0usize;
        let mut adjacent_diff_count = 0usize;
        let mut adjacent_same_count = 0usize;
        let mut diff_size_histogram: HashMap<usize, usize> = HashMap::new(); // symmetric_diff_size -> count
        let mut potential_savings = 0usize;

        for (_, (rm, _usage)) in &unique_rms {
            let entries: Vec<_> = rm.map.range_values().collect();
            total_outer_ranges += entries.len();
            for (_, tsid_set) in &entries {
                total_inner_ranges += tsid_set.ranges_len();
            }

            // Analyze adjacent entries
            for i in 1..entries.len() {
                let (_, prev_tsid_set) = &entries[i-1];
                let (_, curr_tsid_set) = &entries[i];
                if prev_tsid_set == curr_tsid_set {
                    adjacent_same_count += 1;
                } else {
                    adjacent_diff_count += 1;
                    // Compute symmetric difference size
                    let union = &(**prev_tsid_set) | &(**curr_tsid_set);
                    let intersection = &(**prev_tsid_set) & &(**curr_tsid_set);
                    let sym_diff_size = union.len() - intersection.len();
                    *diff_size_histogram.entry(sym_diff_size).or_insert(0) += 1;
                    
                    // If we could merge by widening both to union, how many ranges saved?
                    // Merging removes 1 outer range entry
                    potential_savings += 1;
                }
            }
        }

        eprintln!("WEIGHT_DIAG: === RangeMapWeight Analysis ===");
        eprintln!("WEIGHT_DIAG: Unique weights: {}", total_weights);
        eprintln!("WEIGHT_DIAG: Total outer ranges (token ranges): {}", total_outer_ranges);
        eprintln!("WEIGHT_DIAG: Total inner ranges (tsid ranges across all entries): {}", total_inner_ranges);
        eprintln!("WEIGHT_DIAG: Adjacent pairs same tsid_set: {}", adjacent_same_count);
        eprintln!("WEIGHT_DIAG: Adjacent pairs different tsid_set: {}", adjacent_diff_count);
        eprintln!("WEIGHT_DIAG: Max potential savings (if all merges possible): {} outer ranges", potential_savings);
        
        let mut diffs: Vec<_> = diff_size_histogram.iter().collect();
        diffs.sort_by_key(|(size, _)| *size);
        eprintln!("WEIGHT_DIAG: Symmetric difference size distribution (adjacent pairs):");
        for (diff_size, count) in diffs.iter().take(20) {
            eprintln!("WEIGHT_DIAG:   sym_diff={}: {} pairs", diff_size, count);
        }
        if diffs.len() > 20 {
            eprintln!("WEIGHT_DIAG:   ... and {} more categories", diffs.len() - 20);
        }

        // Analyze how many pairs could merge if we just widened tsid_sets
        // (i.e., added the "missing" tsids)
        let small_diff_mergeable: usize = diff_size_histogram.iter()
            .filter(|(size, _)| **size <= 5)
            .map(|(_, count)| count)
            .sum();
        eprintln!("WEIGHT_DIAG: Pairs with sym_diff <= 5 (easy merges): {} ({:.1}% of different pairs)", 
            small_diff_mergeable,
            100.0 * small_diff_mergeable as f64 / adjacent_diff_count.max(1) as f64);
        eprintln!("WEIGHT_DIAG: ================================");
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
        forward[start] = Weight::all();

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
                        .unwrap_or_else(Weight::all);
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
                        .unwrap_or_else(Weight::all);
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
                    .unwrap_or_else(Weight::all);

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
                    .unwrap_or_else(Weight::all);

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
        fn format_weight_ranges(weight: &Weight) -> String {
            let ranges: Vec<String> = weight
                .to_rsb_allow_expansion()
                .ranges()
                .map(|r| format!("{}..={}", r.start(), r.end()))
                .collect();
            format!("[{}]", ranges.join(", "))
        }

        writeln!(f, "DWA (start: {})", self.body.start_state)?;
        for (id, state) in self.states.0.iter().enumerate() {
            writeln!(f, "  State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "    final_weight: {}", format_weight_ranges(w))?;
            }
            for (on, to) in &state.transitions {
                let w = state.trans_weights.get(on).cloned().unwrap_or_else(Weight::all);
                writeln!(
                    f,
                    "    {} -> {} (weight: {})",
                    format_pos_code(*on),
                    to,
                    format_weight_ranges(&w)
                )?;
            }
        }
        Ok(())
    }
}

impl DWA {
    /// Propagate final_weights through default transitions using transition
    /// signature matching.
    ///
    /// In the parser DWA, deeper stack depths use DEFAULT_TRANSITION_SYMBOL
    /// transitions to fall through to shallower depths. The determinizer treats
    /// DEFAULT as a regular label, so DWA states at deeper depths may lack
    /// final_weight even though semantically equivalent shallower states have it.
    ///
    /// The fix: for each state S with DEFAULT → T and no final_weight, find a
    /// state Q that has the SAME transitions as S but also has final_weight.
    /// Propagate Q's final_weight to S, and widen the incoming transition
    /// weights to S so the walk accumulator can carry the accepting tokens.
    pub fn propagate_final_weights_through_defaults(&mut self) -> usize {
        let default_label = crate::precompute4::utils::DEFAULT_TRANSITION_SYMBOL;

        // Step 1: Build FULL transition signature → final_weight map.
        // Signature = ALL transitions (including DEFAULT), i.e. label → target pairs.
        // This ensures only structurally identical states (same transitions, same targets)
        // can share final_weight. Using only non-DEFAULT transitions was too coarse
        // and caused false positive over-acceptance.
        let mut sig_to_final: std::collections::BTreeMap<
            BTreeMap<Label, StateID>,
            Option<Weight>,
        > = std::collections::BTreeMap::new();

        for state in self.states.iter() {
            let sig: BTreeMap<Label, StateID> = state
                .transitions
                .iter()
                .map(|(&l, &t)| (l, t))
                .collect();

            let entry = sig_to_final.entry(sig).or_insert(None);
            if let Some(fw) = &state.final_weight {
                if let Some(existing) = entry {
                    *existing = &*existing | fw;
                } else {
                    *entry = Some(fw.clone());
                }
            }
        }

        // Step 2: For each state with DEFAULT but no final_weight, look up
        // its FULL transition signature. If any state with the same
        // signature has final_weight, propagate it.
        let mut propagated_states: Vec<(usize, Weight)> = Vec::new();
        let mut unmatched_default_states: Vec<usize> = Vec::new();
        for sid in 0..self.states.len() {
            if self.states[sid].final_weight.is_some() {
                continue;
            }
            if !self.states[sid].transitions.contains_key(&default_label) {
                continue;
            }

            let own_sig: BTreeMap<Label, StateID> = self.states[sid]
                .transitions
                .iter()
                .map(|(&l, &t)| (l, t))
                .collect();

            if let Some(Some(fw)) = sig_to_final.get(&own_sig) {
                if !fw.is_empty() {
                    self.states[sid].final_weight = Some(fw.clone());
                    propagated_states.push((sid, fw.clone()));
                }
            } else {
                unmatched_default_states.push(sid);
            }
        }

        // Step 2b: Follow DEFAULT chains for unmatched states.
        // Some states form chains: deep --DEFAULT--> mid --DEFAULT--> shallow.
        // The signature-based matching in Step 2 may only propagate to mid (from
        // shallow). Now follow DEFAULT from deep to mid (which now has final_weight).
        // Iterate until fixed point for arbitrary chain depth.
        let mut chain_propagated = 0usize;
        loop {
            let mut newly_propagated = Vec::new();
            for &sid in &unmatched_default_states {
                if self.states[sid].final_weight.is_some() {
                    continue; // already got it in a previous iteration
                }
                let target = match self.states[sid].transitions.get(&default_label) {
                    Some(&t) => t,
                    None => continue,
                };
                if let Some(fw) = self.states[target].final_weight.clone() {
                    if !fw.is_empty() {
                        newly_propagated.push((sid, fw));
                    }
                }
            }
            if newly_propagated.is_empty() {
                break;
            }
            for (sid, fw) in &newly_propagated {
                self.states[*sid].final_weight = Some(fw.clone());
                propagated_states.push((*sid, fw.clone()));
            }
            chain_propagated += newly_propagated.len();
        }
        if chain_propagated > 0 {
            crate::debug!(5, "propagate_final_weights: {} additional states via DEFAULT chain following", chain_propagated);
        }
        
        // Debug: count states still lacking final_weight despite having DEFAULT
        {
            let mut still_missing = 0usize;
            let mut chain_examples: Vec<String> = Vec::new();
            for sid in 0..self.states.len() {
                if self.states[sid].final_weight.is_some() { continue; }
                if !self.states[sid].transitions.contains_key(&default_label) { continue; }
                still_missing += 1;
                if chain_examples.len() < 5 {
                    // Follow the DEFAULT chain, also showing non-DEFAULT transitions at each step
                    let mut chain = Vec::new();
                    let mut cur = sid;
                    for _ in 0..15 {
                        let n_trans = self.states[cur].transitions.len();
                        let has_default = self.states[cur].transitions.contains_key(&default_label);
                        let has_fw = self.states[cur].final_weight.is_some();
                        let non_default_labels: Vec<Label> = self.states[cur].transitions.keys()
                            .filter(|&&l| l != default_label)
                            .copied()
                            .collect();
                        chain.push(format!("s{}(trans={},def={},fw={},other={:?})", 
                            cur, n_trans, has_default, has_fw, non_default_labels));
                        if has_fw { break; }
                        if let Some(&t) = self.states[cur].transitions.get(&default_label) {
                            cur = t;
                        } else {
                            break;
                        }
                    }
                    chain_examples.push(format!("  chain: {}", chain.join(" -> ")));
                }
            }
            if still_missing > 0 {
                crate::debug!(5, "propagate_final_weights: {} states STILL missing final_weight with DEFAULT", still_missing);
                for ex in &chain_examples {
                    crate::debug!(5, "{}", ex);
                }
            }
        }

        // Step 3: Widen incoming transition weights to propagated states.
        // The DWA transitions leading to propagated states may not carry the
        // accepting tokens (because the NWA at deeper depths didn't have
        // final_weight, so subtract_final_weights_from_outgoing didn't account
        // for them). Widen these transitions so the walk accumulator can carry
        // the accepting tokens through to the final_weight check.
        if !propagated_states.is_empty() {
            let prop_map: std::collections::BTreeMap<usize, &Weight> = propagated_states
                .iter()
                .map(|(sid, fw)| (*sid, fw))
                .collect();

            for src_id in 0..self.states.len() {
                let targets: Vec<(Label, StateID)> = self.states[src_id]
                    .transitions
                    .iter()
                    .map(|(&l, &t)| (l, t))
                    .collect();

                for (label, target) in targets {
                    if let Some(fw) = prop_map.get(&target) {
                        if let Some(tw) = self.states[src_id].trans_weights.get_mut(&label) {
                            *tw = &*tw | *fw;
                        }
                    }
                }
            }
        }

        let count = propagated_states.len();
        crate::debug!(5, "propagate_final_weights: {} states, DWA has {} states", count, self.states.len());
        count
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
                let ranges: Vec<(usize, usize)> = w.to_rsb_allow_expansion().ranges()
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
                    .unwrap_or_else(Weight::all);
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
