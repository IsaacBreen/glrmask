// src/precompute4/weighted_automata/minimization/dwa_acyclic/mod.rs

use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// A signature representing the "semantic behavior" of a state.
/// Two states are equivalent if and only if they have the same Signature.
///
/// The weights in this signature must be "Relaxed" weights (normalized relative to
/// the weights pushed forward).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StateSignature {
    /// The normalized final weight.
    final_weight: Option<Weight>,
    /// Normalized transitions: (Label, TargetStateID, EdgeWeight)
    /// TargetStateID refers to the ID in the *new* (minimized) set of states.
    transitions: Vec<(Label, StateID, Weight)>,
}

impl DWA {
    /// Minimize an acyclic DWA to minimal state count.
    ///
    /// This uses a bottom-up construction approach (Revuz-like algorithm extended for Weights).
    ///
    /// Algorithm:
    /// 1. Prune unreachable/dead states.
    /// 2. Compute Reverse Topological Order.
    /// 3. Iterate from leaves (end) to root (start):
    ///    a. Compute "Push Weight" (L[u]): The union of all weights that can successfully
    ///       finish a path starting from `u`. This includes `u`'s state_weight, final_weight,
    ///       and weights of outgoing transitions combined with the L of targets.
    ///    b. "Relax" internals: Create a canonical signature where all local weights are
    ///       OR'd with (NOT L[u]). This effectively erases constraints that are already
    ///       enforced by the Push Weight.
    ///    c. Deduplicate: Check if this signature exists in the new state registry.
    ///       If yes, reuse ID. If no, create new state.
    ///    d. Map `u` -> `(new_id, L[u])`.
    /// 4. Reconstruct the DWA.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // 1. Prune dead nodes to ensure topological sort is clean and efficient
        self.pass0_prune();

        if self.states.len() == 0 {
            return;
        }

        // 2. Compute processing order (Leaves -> Roots)
        let rev_order = self.reverse_topological_order();

        // Registry: Signature -> New State ID
        // Used to find existing equivalent states.
        let mut registry: HashMap<StateSignature, StateID> = HashMap::new();

        // New States vector being built
        let mut new_states: Vec<DWAState> = Vec::with_capacity(self.states.len());

        // Mapping: Old State ID -> (New State ID, Pushed Weight)
        // The "Pushed Weight" is the constraint that must be applied to any edge *entering* this state.
        let mut state_mapping: Vec<Option<(StateID, Weight)>> = vec![None; self.states.len()];

        // 3. Bottom-Up Processing
        for &u in &rev_order {
            let old_state = &self.states[u];

            // --- Step A: Calculate the "Push Weight" (L[u]) ---
            // L[u] is the Union of all token sets that can survive from this node to an accept state.
            // L[u] = (StateWeight) INTERSECT [ (FinalWeight) UNION (Union over trans(t): Weight(t) & L[t]) ]

            // 1. Accumulate future possibilities from transitions
            let mut future_union = Weight::zeros();

            // Add contribution from being a final state
            if let Some(fw) = &old_state.final_weight {
                future_union = &future_union | fw;
            }

            // Add contributions from transitions
            // Note: We use the already-computed Pushed Weight of the targets.
            for (&lbl, &old_target) in &old_state.transitions {
                if old_target >= self.states.len() { continue; } // Should be pruned, but safe check

                if let Some((_, target_push_weight)) = &state_mapping[old_target] {
                    let trans_w = old_state.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);

                    // The path is valid if it passes the transition weight AND the target's requirements
                    let path_w = &trans_w & target_push_weight;
                    future_union = &future_union | &path_w;
                }
            }

            // 2. Restrict by State Weight
            // If the state itself has a filter, it limits everything passing through.
            let mut push_weight = future_union;
            if let Some(sw) = &old_state.state_weight {
                push_weight = &push_weight & sw;
            }

            // --- Step B: Build Canonical Signature (Relaxation) ---
            // We create a "Residual State" where all weights are relaxed by `push_weight`.
            // Effectively: Weight_New = Weight_Old | (NOT push_weight).
            // This turns constraints that are fully captured by `push_weight` into `ALL`.

            let dont_care = push_weight.complement();

            // 1. Relax Final Weight
            // Effective final weight is (fw & sw).
            // We assume State Weight is absorbed into Push Weight, so we check just FW relative to Push.
            // Actually, correct logic: The generic node must behave such that:
            //    NodeInput(W) -> InternalCheck -> Result
            // We are moving `push_weight` to `NodeInput`.
            // So `InternalCheck` can be `OriginalCheck | !PushWeight`.

            let mut sig_final_weight = None;
            if let Some(fw) = &old_state.final_weight {
                // Effective local finality requires both state_weight and final_weight
                let effective_fw = if let Some(sw) = &old_state.state_weight {
                    fw & sw
                } else {
                    fw.clone()
                };

                let relaxed = &effective_fw | &dont_care;
                if !relaxed.is_all_fast() {
                    sig_final_weight = Some(relaxed);
                }
            } else {
                // If it wasn't final, can it become final?
                // No, struct definition implies final_weight Option is structural.
                // But conceptually, if push_weight is empty (dead state), everything relaxes to ALL.
                // We handle non-final states by keeping None.
            }

            // 2. Relax Transitions
            let mut sig_transitions = Vec::new();
            for (&lbl, &old_target) in &old_state.transitions {
                if let Some((new_target_id, target_push_w)) = &state_mapping[old_target] {
                    // Effective path weight: TransW & TargetPushW & StateW
                    let old_trans_w = old_state.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);
                    let sw = old_state.state_weight.as_ref().cloned().unwrap_or_else(Weight::all);

                    let effective_w = &(&old_trans_w & target_push_w) & &sw;

                    // Relax: w | !L[u]
                    let relaxed_w = &effective_w | dont_care.clone();

                    sig_transitions.push((lbl, *new_target_id, relaxed_w));
                }
            }

            // Sort to ensure canonical order for hashing
            sig_transitions.sort_by(|a, b| a.0.cmp(&b.0));

            let signature = StateSignature {
                final_weight: sig_final_weight,
                transitions: sig_transitions,
            };

            // --- Step C: Deduplicate ---
            let new_id = if let Some(&existing_id) = registry.get(&signature) {
                existing_id
            } else {
                let id = new_states.len();

                // Convert Signature back to DWAState
                let mut trans_map = BTreeMap::new();
                let mut weights_map = BTreeMap::new();

                for (l, t, w) in &signature.transitions {
                    trans_map.insert(*l, *t);
                    if !w.is_all_fast() {
                        weights_map.insert(*l, w.clone());
                    }
                }

                new_states.push(DWAState {
                    transitions: trans_map,
                    trans_weights: weights_map,
                    final_weight: signature.final_weight.clone(),
                    state_weight: None, // State weight has been pushed out!
                });

                registry.insert(signature, id);
                id
            };

            // --- Step D: Map ---
            state_mapping[u] = Some((new_id, push_weight));
        }

        // 4. Reconstruct DWA
        // The start state needs special handling.
        // We have `start_mapping = (new_start_id, global_push_weight)`.
        // The `global_push_weight` must be applied. Since we can't put weights on the
        // "incoming arrow" to start, we apply it to the Start State's `state_weight`.

        if let Some((mapped_start, start_req)) = state_mapping[self.body.start_state].clone() {
            // Optimization: If the mapped start state is "fresh" (not used by others)
            // or if we don't care about cloning, we can just apply the weight.
            // However, `mapped_start` might be a merged state used deep in the graph.
            // We must create a dedicated entry point if the weight is restrictive.

            if start_req.is_all_fast() {
                self.states = DWAStates(new_states);
                self.body.start_state = mapped_start;
            } else {
                // Must create a new start state that clones the behavior of `mapped_start`
                // but restricts it by `start_req`.
                // Actually, simply cloning the state in `new_states` and adding state_weight works.
                let mut root_state = new_states[mapped_start].clone();
                root_state.apply_weight(&start_req);

                new_states.push(root_state);
                let new_root_id = new_states.len() - 1;

                self.states = DWAStates(new_states);
                self.body.start_state = new_root_id;
            }
        } else {
            // Start state was pruned (unreachable or dead)
            self.states = DWAStates(vec![DWAState::default()]);
            self.body.start_state = 0;
        }
    }

    // ========================================================================
    // HELPER: PRUNING (Copied from previous context to ensure completeness)
    // ========================================================================

    pub(crate) fn pass0_prune(&mut self) {
        self.prune_unreachable_acyclic();
        self.prune_dead_ends_acyclic();
    }

    pub(crate) fn prune_unreachable_acyclic(&mut self) {
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

    pub(crate) fn prune_dead_ends_acyclic(&mut self) {
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

    pub(crate) fn reverse_topological_order(&self) -> Vec<usize> {
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

    pub fn minimize_internal_acyclic(&mut self) -> bool {
        self.minimize_acyclic();
        true
    }
}