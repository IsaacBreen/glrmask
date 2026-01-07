// src/precompute4/weighted_automata/minimization/dwa_acyclic/mod.rs

use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{HashMap, VecDeque};

/// Represents the structural "skeleton" of a state.
/// Two states can only be merged if they have identical skeletons.
/// The skeleton consists of the sorted transitions (Labels and Target State IDs).
/// Weights are NOT part of the skeleton; they are handled during the clustering phase.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct StateSkeleton {
    // Sorted list of (Label, TargetStateID).
    // TargetStateID refers to the ID in the NEW (minimized) set of states.
    transitions: Vec<(Label, StateID)>,
}

impl DWA {
    /// Minimize an acyclic DWA to minimal state count using a "Push-Weights-to-Front" strategy.
    ///
    /// The Algorithm:
    /// 1. Prune unreachable/dead states.
    /// 2. Compute `L[u]` (Max Reachable Weight) for every state `u`.
    ///    `L[u]` is the union of all weights accepted by all paths starting at `u`.
    /// 3. Iterate states in Reverse Topological Order (Leaves -> Roots).
    /// 4. For each state `u`, generate its `Skeleton` (based on minimized targets).
    /// 5. Attempt to merge `u` into an existing cluster of states with the same Skeleton.
    ///    Merging is valid if the weights are "Compatible".
    ///    Compatibility ensures that the union of weights in the merged state, when filtered
    ///    by the incoming constraint `L`, behaves exactly like the original state.
    /// 6. Reconstruct the DWA.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // 1. Prune dead nodes to ensure topological sort is clean and efficient
        self.pass0_prune();
        if self.states.len() == 0 {
            return;
        }

        // 2. Compute Reverse Topological Order
        let rev_order = self.reverse_topological_order();

        // 3. Compute Max Reachable Weights (L[u])
        // L[u] = FinalWeight[u] U Union(TransWeight[u, l] & L[target])
        let l_weights = self.compute_push_weights(&rev_order);

        // 4. Bottom-Up Clustering
        // Stores the new minimized states
        let mut new_states: Vec<DWAState> = Vec::new();
        // Maps OldStateID -> NewStateID
        let mut old_to_new: Vec<StateID> = vec![0; self.states.len()];

        // Registry to find candidate states for merging.
        // Map<Skeleton, List<(NewStateID, List<ConstraintL>)>>
        // We store the list of L-weights of all original states merged into NewStateID
        // to perform the validity check.
        let mut registry: HashMap<StateSkeleton, Vec<(StateID, Vec<Weight>)>> = HashMap::new();

        for &u in &rev_order {
            let old_state = &self.states[u];

            // Build Skeleton using mapped targets
            let mut transitions: Vec<(Label, StateID)> = old_state.transitions.iter()
                .map(|(&lbl, &t)| (lbl, old_to_new[t]))
                .collect();
            transitions.sort_by_key(|t| t.0);
            let skeleton = StateSkeleton { transitions: transitions.clone() };

            let u_l = &l_weights[u];
            let mut merged_target_id = None;

            // Try to find a compatible existing state in the registry
            if let Some(candidates) = registry.get_mut(&skeleton) {
                for (cand_id, constraints) in candidates.iter_mut() {
                    let cand_state = &mut new_states[*cand_id];

                    // Check compatibility
                    if Self::can_merge(cand_state, constraints, old_state, u_l) {
                        // Merge!
                        Self::perform_merge(cand_state, old_state);
                        constraints.push(u_l.clone());
                        merged_target_id = Some(*cand_id);
                        break;
                    }
                }
            }

            if let Some(id) = merged_target_id {
                old_to_new[u] = id;
            } else {
                // Create new state
                let new_id = new_states.len();
                // StateWeight is set to None (ALL) for internal states.
                // The restriction is applied by the incoming edges (L[u]).
                let mut new_state = DWAState {
                    transitions: old_state.transitions.iter()
                        .map(|(&l, &t)| (l, old_to_new[t]))
                        .collect(),
                    trans_weights: old_state.trans_weights.clone(),
                    final_weight: old_state.final_weight.clone(),
                    state_weight: None,
                };

                // If the transition targets in old_state were raw, we need to map them in the new state struct too
                // (Already done in map construction above, but DWAState needs BTreeMap)
                let mut new_trans_map = std::collections::BTreeMap::new();
                for &(lbl, target) in &skeleton.transitions {
                    new_trans_map.insert(lbl, target);
                }
                new_state.transitions = new_trans_map;

                new_states.push(new_state);
                old_to_new[u] = new_id;

                registry.entry(skeleton)
                    .or_default()
                    .push((new_id, vec![u_l.clone()]));
            }
        }

        // 5. Reconstruct Start State
        // The Start State needs special handling because there is no "Incoming Edge" to hold L[start].
        // We must apply L[start] to the state_weight of the start state.

        let mapped_start = old_to_new[self.body.start_state];
        let start_req = &l_weights[self.body.start_state];

        // We can simply clone the mapped state and apply the weight.
        // This ensures we don't accidentally restrict a shared state used deep in the graph.
        let mut root_state = new_states[mapped_start].clone();

        // Combine with original state_weight if any
        if let Some(orig_sw) = &self.states[self.body.start_state].state_weight {
            root_state.apply_weight(orig_sw);
        }
        root_state.apply_weight(start_req);

        new_states.push(root_state);
        self.body.start_state = new_states.len() - 1;
        self.states = DWAStates(new_states);
    }

    /// Computes L[u]: The union of all path weights starting from u.
    fn compute_push_weights(&self, rev_order: &[usize]) -> Vec<Weight> {
        let mut l = vec![Weight::zeros(); self.states.len()];

        for &u in rev_order {
            let state = &self.states[u];
            let mut acc = Weight::zeros();

            // 1. Final Weight contribution
            if let Some(fw) = &state.final_weight {
                acc |= fw;
            }

            // 2. Transition contributions
            for (&lbl, &target) in &state.transitions {
                if target < self.states.len() {
                    let w_trans = state.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);
                    let branch_contrib = &w_trans & &l[target];
                    acc |= &branch_contrib;
                }
            }

            // 3. State Weight restriction (local to u)
            if let Some(sw) = &state.state_weight {
                acc &= sw;
            }

            l[u] = acc;
        }
        l
    }

    /// Checks if `u` can be safely merged into `cluster_state`.
    ///
    /// Conditions:
    /// 1. Cluster -> U Safety: The existing weights in Cluster (`W_curr`), when masked by `L_u`,
    ///    must not exceed `W_u`.
    ///    Actually, since `W_u` is added to `W_curr`, we strictly need: `L_u & W_curr <= W_u`.
    ///
    /// 2. U -> Cluster Safety: The weight `W_u`, when masked by any `L_v` from the cluster,
    ///    must not exceed `W_v` (where `W_v <= W_curr`).
    ///    Strictly: `L_v & W_u <= W_curr`.
    fn can_merge(
        cluster_state: &DWAState,
        cluster_constraints: &[Weight],
        u_state: &DWAState,
        u_l: &Weight
    ) -> bool {
        // Check Final Weights
        {
            let w_curr = cluster_state.final_weight.clone().unwrap_or_else(Weight::zeros);
            let w_u = u_state.final_weight.clone().unwrap_or_else(Weight::zeros);

            // 1. Cluster -> U
            let check1 = u_l & &w_curr;
            if !check1.is_subset_of(&w_u) { return false; }

            // 2. U -> Cluster
            for v_l in cluster_constraints {
                let check2 = v_l & &w_u;
                if !check2.is_subset_of(&w_curr) { return false; }
            }
        }

        // Check Transitions
        // Note: Skeletons match, so keys are identical.
        for lbl in cluster_state.transitions.keys() {
            let w_curr = cluster_state.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);
            let w_u = u_state.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);

            // 1. Cluster -> U
            let check1 = u_l & &w_curr;
            if !check1.is_subset_of(&w_u) { return false; }

            // 2. U -> Cluster
            for v_l in cluster_constraints {
                let check2 = v_l & &w_u;
                if !check2.is_subset_of(&w_curr) { return false; }
            }
        }

        true
    }

    /// Updates `cluster_state` to include the weights of `u_state`.
    fn perform_merge(cluster_state: &mut DWAState, u_state: &DWAState) {
        // Union Final Weight
        let fw_u = u_state.final_weight.clone().unwrap_or_else(Weight::zeros);
        if let Some(fw_curr) = &mut cluster_state.final_weight {
            *fw_curr |= &fw_u;
        } else if !fw_u.is_empty() {
            // If cluster was empty/zero (None usually implies 0 in this context if explicit field is Option)
            // But struct def says Option<Weight>. Usually None means 0 for Final, All for Trans.
            // Let's stick to DWA defs: final_weight None => Zero.
            cluster_state.final_weight = Some(fw_u);
        }

        // Union Transition Weights
        for (lbl, _) in &cluster_state.transitions {
            let w_u = u_state.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);

            // We need to union w_u into w_curr.
            // Map entry might be missing (implicit All) or present.
            // Best to normalize to explicit weights.
            let w_curr = cluster_state.trans_weights.entry(*lbl).or_insert_with(Weight::all);
            *w_curr |= &w_u;
        }
    }

    // ========================================================================
    // PRUNING & TOPOLOGY HELPERS
    // ========================================================================

    pub(crate) fn pass0_prune(&mut self) {
        self.prune_unreachable_acyclic();
        self.prune_dead_ends_acyclic();
    }

    fn prune_unreachable_acyclic(&mut self) {
        let n = self.states.len();
        if n == 0 { return; }
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
        if !reachable.iter().all(|&r| r) {
            self.rebuild_keeping_only(&reachable);
        }
    }

    fn prune_dead_ends_acyclic(&mut self) {
        let n = self.states.len();
        if n == 0 { return; }
        let mut reverse_adj = vec![Vec::new(); n];
        for u in 0..n {
            for &v in self.states[u].transitions.values() {
                if v < n { reverse_adj[v].push(u); }
            }
        }
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
        if !can_accept.iter().all(|&c| c) {
            self.rebuild_keeping_only(&can_accept);
        }
    }

    fn reverse_topological_order(&self) -> Vec<usize> {
        let n = self.states.len();
        if n == 0 { return vec![]; }
        let mut out_degree = vec![0usize; n];
        let mut reverse_adj = vec![Vec::new(); n];
        for (u, state) in self.states.0.iter().enumerate() {
            for &v in state.transitions.values() {
                if v < n {
                    out_degree[u] += 1;
                    reverse_adj[v].push(u);
                }
            }
        }
        let mut queue = VecDeque::new();
        for i in 0..n {
            if out_degree[i] == 0 { queue.push_back(i); }
        }
        let mut order = Vec::with_capacity(n);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &u in &reverse_adj[v] {
                out_degree[u] -= 1;
                if out_degree[u] == 0 { queue.push_back(u); }
            }
        }
        order
    }

    fn rebuild_keeping_only(&mut self, keep: &[bool]) {
        let n = self.states.len();
        let mut old_to_new = vec![None; n];
        let mut new_idx = 0;
        for i in 0..n {
            if keep[i] {
                old_to_new[i] = Some(new_idx);
                new_idx += 1;
            }
        }
        let mut new_states = Vec::with_capacity(new_idx);
        for (i, state) in self.states.0.drain(..).enumerate() {
            if keep[i] {
                let mut new_state = state;
                new_state.transitions.retain(|_, t| keep[*t]);
                for t in new_state.transitions.values_mut() {
                    *t = old_to_new[*t].unwrap();
                }
                new_state.trans_weights.retain(|l, _| new_state.transitions.contains_key(l));
                new_states.push(new_state);
            }
        }
        self.states.0 = new_states;
        self.body.start_state = old_to_new[self.body.start_state].unwrap_or(0);
    }
}