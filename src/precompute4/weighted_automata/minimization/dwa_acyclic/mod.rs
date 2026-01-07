//! Acyclic DWA Minimization - Complete Rewrite
//!
//! This implements a mathematically-grounded 4-pass minimization algorithm:
//!
//! **Pass 0: Pruning**
//! - Remove unreachable states (forward from start)
//! - Remove dead-end states (backward from finals)
//!
//! **Pass 1: Weight Pushing**
//! - Compute B[q] = weights that can finish (reach accepting) from q
//! - Push zeros toward start: ω_T(q,a) ← ω_T(q,a) ∩ B[target]
//!
//! **Pass 2: Weight Relaxation**
//! - Compute R[q] = weights that survive on ALL paths from start to q
//! - Recompute B[q] after pushing
//! - Add don't-cares: ω_T(q,a) ∪= (W \ R[q]) ∪ (W \ B[target])
//!
//! **Pass 3: State Merging**
//! - Bottom-up signature hashing
//! - States with identical signatures (structure + weights) merge
//!
//! Correctness: Each pass preserves semantic equivalence (path weight unchanged).
//! Complexity: O((|Q| + |δ|) · |W|)

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap, VecDeque};

impl DWA {
    /// Minimize an acyclic DWA to minimal state count.
    ///
    /// Uses the 4-pass algorithm: Prune → Push → Relax → Merge.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }
        self.minimize_internal_acyclic();
    }

    /// Internal minimization algorithm.
    pub fn minimize_internal_acyclic(&mut self) -> bool {
        let initial_states = self.states.len();
        if initial_states == 0 {
            return false;
        }

        // Pass 0: Pruning
        self.pass0_prune();
        if self.states.len() == 0 {
            return initial_states > 0;
        }

        // Pass 1: Weight Pushing
        self.pass1_weight_push();

        // Pass 2: Weight Relaxation
        self.pass2_weight_relax();

        // Pass 3: State Merging
        self.pass3_state_merge();

        self.states.len() < initial_states
    }

    // ========================================================================
    // PASS 0: PRUNING
    // ========================================================================

    /// Remove unreachable states (forward from start) and dead-ends (backward from finals).
    fn pass0_prune(&mut self) {
        self.prune_unreachable_acyclic();
        self.prune_dead_ends_acyclic();
    }

    /// Remove states not reachable from the start state.
    pub fn prune_unreachable_acyclic(&mut self) -> bool {
        let before = self.states.len();
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // BFS forward from start
        let mut reachable = vec![false; n];
        let mut queue = VecDeque::new();
        
        if self.body.start_state < n {
            queue.push_back(self.body.start_state);
            reachable[self.body.start_state] = true;
        }

        while let Some(s) = queue.pop_front() {
            for &t in self.states[s].transitions.values() {
                if t < n && !reachable[t] {
                    reachable[t] = true;
                    queue.push_back(t);
                }
            }
        }

        if reachable.iter().all(|&r| r) {
            return false;
        }

        self.rebuild_keeping_only(&reachable);
        self.states.len() < before
    }

    /// Remove states that cannot reach any final state.
    pub fn prune_dead_ends_acyclic(&mut self) -> bool {
        let before = self.states.len();
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Build reverse adjacency
        let mut reverse_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for u in 0..n {
            for &v in self.states[u].transitions.values() {
                if v < n {
                    reverse_adj[v].push(u);
                }
            }
        }

        // BFS backward from finals
        let mut can_accept = vec![false; n];
        let mut queue = VecDeque::new();

        for i in 0..n {
            if self.states[i].final_weight.is_some() {
                can_accept[i] = true;
                queue.push_back(i);
            }
        }

        while let Some(v) = queue.pop_front() {
            for &u in &reverse_adj[v] {
                if !can_accept[u] {
                    can_accept[u] = true;
                    queue.push_back(u);
                }
            }
        }

        if can_accept.iter().all(|&c| c) {
            return false;
        }

        self.rebuild_keeping_only(&can_accept);
        self.states.len() < before
    }

    // ========================================================================
    // PASS 1: WEIGHT PUSHING
    // ========================================================================

    /// Push zeros toward the start: ω_T(q,a) ← ω_T(q,a) ∩ B[target]
    ///
    /// B[q] = weights that can survive from q to some accepting state
    fn pass1_weight_push(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Compute B[q] in reverse topological order
        let topo_order = self.reverse_topological_order();
        let mut b: Vec<Weight> = vec![Weight::zeros(); n];

        for &q in &topo_order {
            // Start with final weight if accepting
            let mut b_q = self.states[q]
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);

            // Union (trans_weight ∩ B[target]) for all outgoing transitions
            for (&label, &target) in &self.states[q].transitions {
                if target >= n {
                    continue;
                }
                let tw = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                b_q = &b_q | &(&tw & &b[target]);
            }

            b[q] = b_q;
        }

        // Push: ω_T(q,a) ← ω_T(q,a) ∩ B[target]
        for q in 0..n {
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    *tw = tw.clone() & &b[target];
                }
            }
        }
    }

    // ========================================================================
    // PASS 2: WEIGHT RELAXATION
    // ========================================================================

    /// Add don't-cares for weights blocked forward or backward.
    ///
    /// KEY INSIGHT: We process in REVERSE topological order, updating B dynamically.
    /// When we relax a final weight, we also update B[q]. This ensures that when
    /// we later process transitions pointing to q, we use the POST-relaxation
    /// value of B, not a stale pre-relaxation value.
    ///
    /// Algorithm:
    /// 1. Compute R[q] = weights that CAN reach q (via some path), using UNION
    /// 2. Process states in reverse topological order (leaves to roots):
    ///    a. Initialize B[q] from final weight (if any)
    ///    b. Add incoming B contributions from already-processed successors
    ///    c. Relax final weight: add (W \ R[q]) 
    ///    d. Update B[q] to include the relaxed final weight
    ///    e. Relax all outgoing transitions: add (W \ R[q]) | (W \ B[target])
    ///    f. Update B[q] to reflect the relaxed transitions
    fn pass2_weight_relax(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Compute forward topological order for R computation
        let fwd_topo = self.topological_order();
        
        // Compute R[q] (forward reachable weights) in topological order
        // R[q] = weights that CAN reach q via SOME path (UNION over paths)
        // 
        // This is the correct definition because:
        // - W \ R[q] = weights that are blocked on ALL paths to q
        // - These are true "don't cares" - they can never contribute to accepting
        let mut r: Vec<Weight> = vec![Weight::zeros(); n];

        // R[start] = W (all weights can reach start from start)
        if self.body.start_state < n {
            r[self.body.start_state] = Weight::all();
        }

        // Build reverse adjacency with transition weights (for R computation)
        let mut incoming: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n];
        for u in 0..n {
            for (&label, &v) in &self.states[u].transitions {
                if v < n {
                    incoming[v].push((u, label));
                }
            }
        }

        // R[q] = ∪_{(p,a): δ(p,a)=q} (R[p] ∩ ω_T(p,a))
        // Union over all incoming edges - weight can reach q if ANY path allows it
        for &q in &fwd_topo {
            if q == self.body.start_state {
                continue; // Already set to W
            }
            
            if incoming[q].is_empty() {
                // No incoming edges (and not start) -> unreachable, R = ∅
                continue;
            }

            // Start with empty, UNION all incoming paths
            let mut r_q = Weight::zeros();
            for &(p, label) in &incoming[q] {
                let tw = self.states[p]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                // Add weights that can come through this edge
                r_q = &r_q | &(&r[p] & &tw);
            }
            r[q] = r_q;
        }

        // Process in REVERSE topological order, computing B and relaxing simultaneously.
        // This ensures we process leaves (sinks) before their predecessors.
        // When we get to a state, all its successors are already relaxed and B is up-to-date.
        let rev_topo = self.reverse_topological_order();
        let mut b: Vec<Weight> = vec![Weight::zeros(); n];

        for &q in &rev_topo {
            // Step 1: Compute initial B[q] from final weight and successors
            // Since we're in reverse topo order, all successors have been processed
            let mut b_q = self.states[q]
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);

            for (&label, &target) in &self.states[q].transitions {
                if target >= n {
                    continue;
                }
                let tw = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                // B[target] is already final (post-relaxation) since we processed target first
                b_q = &b_q | &(&tw & &b[target]);
            }

            // Step 2: Compute cannot_reach = W \ R[q]
            // These weights can never reach this state, so they're don't-cares
            let cannot_reach = r[q].complement();

            // Step 3: Relax final weight if present
            // Add cannot_reach to final weight (if can't reach, doesn't matter what final says)
            if let Some(ref mut fw) = self.states.0[q].final_weight {
                let relaxed_fw = fw.clone() | &cannot_reach;
                *fw = relaxed_fw.clone();
                // Update b_q to reflect the relaxed final weight
                // This is crucial: b_q needs to include the don't-cares we just added
                b_q = &b_q | &relaxed_fw;
            }

            // Step 4: Relax outgoing transition weights
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                // B[target] is already post-relaxation (processed earlier in reverse topo)
                let cannot_finish_from_target = b[target].complement();
                let dont_care = &cannot_reach | &cannot_finish_from_target;

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    let relaxed_tw = tw.clone() | &dont_care;
                    *tw = relaxed_tw.clone();
                    // Update b_q to reflect the relaxed transition weight
                    b_q = &b_q | &(&relaxed_tw & &b[target]);
                }
            }

            // Store final B[q] for use by predecessors
            b[q] = b_q;
        }
    }

    // ========================================================================
    // PASS 3: STATE MERGING
    // ========================================================================

    /// Merge states with identical signatures (bottom-up).
    ///
    /// After relaxation, states that can merge have exactly identical weights.
    fn pass3_state_merge(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        let rev_topo = self.reverse_topological_order();

        // repr[q] = representative state for q
        let mut repr: Vec<usize> = (0..n).collect();

        // Map from signature to representative
        let mut sig_to_repr: HashMap<StateSignature, usize> = HashMap::new();

        // Process bottom-up (sinks first)
        for &q in &rev_topo {
            // Build signature using already-computed representatives
            let sig = self.compute_signature(q, &repr);

            if let Some(&existing_repr) = sig_to_repr.get(&sig) {
                repr[q] = existing_repr;
            } else {
                repr[q] = q;
                sig_to_repr.insert(sig, q);
            }
        }

        // Count distinct classes
        let num_classes = repr.iter().enumerate().filter(|&(i, &r)| i == r).count();
        if num_classes == n {
            return; // No merging possible
        }

        // Rebuild with merged states
        self.rebuild_merged(&repr);
    }

    /// Compute signature for state q.
    fn compute_signature(&self, q: usize, repr: &[usize]) -> StateSignature {
        let is_final = self.states[q].final_weight.is_some();
        let final_weight = self.states[q].final_weight.clone();

        let mut transitions: Vec<(i32, usize, Weight)> = Vec::new();
        for (&label, &target) in &self.states[q].transitions {
            let target_repr = if target < repr.len() { repr[target] } else { target };
            let tw = self.states[q]
                .trans_weights
                .get(&label)
                .cloned()
                .unwrap_or_else(Weight::all);
            transitions.push((label, target_repr, tw));
        }
        transitions.sort_by_key(|(l, _, _)| *l);

        StateSignature {
            is_final,
            final_weight,
            transitions,
        }
    }

    /// Rebuild DWA with merged states.
    fn rebuild_merged(&mut self, repr: &[usize]) {
        let n = self.states.len();

        // Find representative states
        let mut is_repr = vec![false; n];
        for q in 0..n {
            if repr[q] == q {
                is_repr[q] = true;
            }
        }

        // Map old → new indices
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];
        let mut new_idx = 0;
        for q in 0..n {
            if is_repr[q] {
                old_to_new[q] = Some(new_idx);
                new_idx += 1;
            }
        }

        // Non-representatives map to their representative's new index
        for q in 0..n {
            if !is_repr[q] {
                old_to_new[q] = old_to_new[repr[q]];
            }
        }

        // Build new states
        let mut new_states: Vec<DWAState> = Vec::with_capacity(new_idx);
        for q in 0..n {
            if !is_repr[q] {
                continue;
            }

            let old_state = &self.states[q];
            let mut new_state = DWAState {
                final_weight: old_state.final_weight.clone(),
                transitions: BTreeMap::new(),
                trans_weights: BTreeMap::new(),
                state_weight: old_state.state_weight.clone(),
            };

            // Remap transitions
            for (&label, &target) in &old_state.transitions {
                if let Some(new_target) = old_to_new[target] {
                    new_state.transitions.insert(label, new_target);
                    if let Some(tw) = old_state.trans_weights.get(&label) {
                        new_state.trans_weights.insert(label, tw.clone());
                    }
                }
            }

            new_states.push(new_state);
        }

        // Update start state
        let new_start = old_to_new[self.body.start_state].unwrap_or(0);

        self.states = DWAStates(new_states);
        self.body.start_state = new_start;
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    /// Compute topological order (sources first).
    fn topological_order(&self) -> Vec<usize> {
        let n = self.states.len();
        if n == 0 {
            return vec![];
        }

        // Compute in-degrees
        let mut in_degree = vec![0usize; n];
        for state in &self.states.0 {
            for &v in state.transitions.values() {
                if v < n {
                    in_degree[v] += 1;
                }
            }
        }

        // Start with sources (in_degree == 0)
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in self.states[u].transitions.values() {
                if v < n {
                    in_degree[v] -= 1;
                    if in_degree[v] == 0 {
                        queue.push_back(v);
                    }
                }
            }
        }

        order
    }

    /// Compute reverse topological order (sinks first).
    pub(crate) fn reverse_topological_order(&self) -> Vec<usize> {
        let n = self.states.len();
        if n == 0 {
            return vec![];
        }

        // Compute out-degrees
        let mut out_degree = vec![0usize; n];
        let mut reverse_adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (u, state) in self.states.0.iter().enumerate() {
            for &v in state.transitions.values() {
                if v < n {
                    out_degree[u] += 1;
                    reverse_adj[v].push(u);
                }
            }
        }

        // Start with sinks (out_degree == 0)
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if out_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &u in &reverse_adj[v] {
                out_degree[u] -= 1;
                if out_degree[u] == 0 {
                    queue.push_back(u);
                }
            }
        }

        order
    }

    /// Rebuild DWA keeping only states where keep[i] is true.
    fn rebuild_keeping_only(&mut self, keep: &[bool]) {
        let n = self.states.len();
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];
        let mut new_idx = 0;
        for i in 0..n {
            if keep[i] {
                old_to_new[i] = Some(new_idx);
                new_idx += 1;
            }
        }
        if new_idx == n {
            return; // Nothing to remove
        }

        let mut new_states = Vec::with_capacity(new_idx);
        for (i, state) in self.states.0.drain(..).enumerate() {
            if !keep[i] {
                continue;
            }
            let mut new_state = state;
            // Remap transitions
            new_state.transitions.retain(|_, t| keep[*t]);
            for t in new_state.transitions.values_mut() {
                *t = old_to_new[*t].unwrap();
            }
            new_state.trans_weights.retain(|l, _| new_state.transitions.contains_key(l));
            new_states.push(new_state);
        }
        self.states.0 = new_states;
        self.body.start_state = old_to_new[self.body.start_state].unwrap_or(0);
    }

    // ========================================================================
    // LEGACY API COMPATIBILITY
    // ========================================================================

    /// Lightweight version - just prunes, no full minimization.
    pub fn minimize_lightweight_acyclic(&mut self) {
        self.pass0_prune();
    }

    /// Single pass - runs full minimization once.
    pub fn minimize_single_pass_acyclic(&mut self) {
        self.minimize_internal_acyclic();
    }

    /// RustFST-based minimization (for comparison/benchmarking).
    pub fn minimize_with_rustfst_full_acyclic(&mut self) -> bool {
        self.minimize_internal_acyclic()
    }

    pub fn push_weights_into_transitions_and_finals_acyclic(&mut self) -> bool {
        self.pass1_weight_push();
        true
    }

    pub fn push_weights_to_initial_acyclic(&mut self) -> bool {
        false // Not used in new algorithm
    }

    pub fn residuated_push_acyclic(&mut self) -> bool {
        false // Not used in new algorithm
    }

    pub fn minimize_states_acyclic(&mut self) -> bool {
        self.minimize_internal_acyclic()
    }

    pub fn loosen_weights_for_minimize_acyclic(&mut self) -> bool {
        self.pass2_weight_relax();
        true
    }

    // Legacy methods for compatibility with old API
    pub fn compute_live_sets(&self) -> Vec<Weight> {
        let n = self.states.len();
        if n == 0 {
            return vec![];
        }

        let topo_order = self.reverse_topological_order();
        let mut b: Vec<Weight> = vec![Weight::zeros(); n];

        for &q in &topo_order {
            let mut b_q = self.states[q]
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);

            for (&label, &target) in &self.states[q].transitions {
                if target >= n {
                    continue;
                }
                let tw = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                b_q = &b_q | &(&tw & &b[target]);
            }

            b[q] = b_q;
        }

        b
    }

    pub fn normalize_weights(&mut self, live: &[Weight]) -> Weight {
        let n = self.states.len();
        if n == 0 {
            return Weight::zeros();
        }

        let start = self.body.start_state;
        let start_live = if start < n {
            live[start].clone()
        } else {
            Weight::zeros()
        };

        // Trim each transition's weight to live(target)
        for q in 0..n {
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                let live_target = &live[target];

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    *tw = tw.clone() & live_target;
                }
            }
        }

        start_live
    }

    pub fn merge_by_signature(&mut self, _live: &[Weight]) {
        self.pass3_state_merge();
    }

    pub fn relax_edges(&mut self, live: &[Weight]) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        for q in 0..n {
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                let dead_target = live[target].complement();

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    *tw = tw.clone() | &dead_target;
                }
            }
        }
    }
}

/// State signature for merging.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StateSignature {
    is_final: bool,
    final_weight: Option<Weight>,
    transitions: Vec<(i32, usize, Weight)>,
}
