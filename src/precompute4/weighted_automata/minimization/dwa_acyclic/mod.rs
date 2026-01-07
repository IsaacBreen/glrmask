use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates, DWABuildError};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

/// A hashable signature for a DWA state, used for minimization.
/// Represents the "local behavior" of the state after canonicalization.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct StateSignature {
    final_weight: Option<Weight>, // None implies Zero
    // Sorted list of transitions: (Label, TargetStateID, Weight)
    transitions: Vec<(Label, StateID, Weight)>,
}

impl DWA {
    /// Minimizes an acyclic DWA.
    /// This algorithm performs:
    /// 1. Reverse topological sort.
    /// 2. Weight pushing (reverse): Pushes common weights from outgoing transitions/finality
    ///    to the incoming edges (and state weight), canonicalizing the state.
    /// 3. Minimization: Merges states with identical signatures (final weights and transitions).
    pub fn minimize_acyclic(&mut self) {
        // 1. Compute Topological Order (or Reverse Topological Order directly)
        // Since we need to process children before parents, we want Reverse Topological Order.
        // We can use a DFS post-order traversal.
        let mut visited = vec![false; self.states.len()];
        let mut order = Vec::with_capacity(self.states.len());

        for i in 0..self.states.len() {
            if !visited[i] {
                self.dfs_post_order(i, &mut visited, &mut order);
            }
        }
        // `order` now contains states such that children appear before parents (mostly).
        // Actually, in post-order, a node is added after its children. 
        // So iterating `order` effectively processes children first.

        // 2. Weight Pushing & Normalization
        // We maintain `push_weights`: Map from StateID to the Weight that was pushed "out" of it (to the left).
        let mut push_weights: Vec<Weight> = vec![Weight::all(); self.states.len()];

        // We can rebuild the states in-place or separate?
        // We need to modify transitions to point to 'old' IDs but with updated weights.
        // And we need to apply pulls from children.
        // Since `order` ensures we process a node *after* its children, we can apply children's push weights immediately.

        for &u in order.iter() {
            // Step 2a: Apply push weights from children to current outgoing transitions
            // (and effectively to self).
            let mut trans_keys: Vec<Label> = self.states[u].transitions.keys().cloned().collect();
            for &lbl in &trans_keys {
                let v = self.states[u].transitions[&lbl];
                if v >= self.states.len() { continue; } // Bound check

                // Get the weight pushed from child v
                let pushed_from_v = &push_weights[v];

                // Update transition weight: w_uv = w_uv & pushed_from_v
                if let Some(w) = self.states[u].trans_weights.get_mut(&lbl) {
                    *w &= pushed_from_v;
                }
            }

            // Step 2b: Calculate maximal pullable weight K from self
            // K = Union(all outgoing transition weights) U final_weight
            // Start with Empty (Zero)
            let mut k = Weight::zeros();

            if let Some(fw) = &self.states[u].final_weight {
                k |= fw;
            }

            for w in self.states[u].trans_weights.values() {
                k |= w;
            }

            // Step 2c: Determine the total weight to push to parents (P_u)
            // P_u = state_weight(u) & K
            // If state_weight is None, it is ALL.
            let sw = self.states[u].state_weight.clone().unwrap_or_else(Weight::all);
            let mut p_u = sw;
            p_u &= &k;

            push_weights[u] = p_u; // Store to be used by parents

            // Step 2d: Normalize state `u`
            // - State Weight becomes ALL (we pushed the restriction)
            // - Outgoing weights `w` become `w / K` (conceptually).
            //   Since we lack exact division/complement, we use the heuristic:
            //   If `w == K`, replace with ALL. Otherwise keep `w` (since w = w & K is invariant).
            //   (This is valid because we pushed K to the left. Effectively `w_new = w U !K` is the ideal, 
            //    and `w==K` covers the case where we can simplify to ALL).

            self.states[u].state_weight = None; // effectively ALL

            if let Some(fw) = &mut self.states[u].final_weight {
                if *fw == k {
                    *fw = Weight::all();
                }
                // Else leave as is
            }

            for w in self.states[u].trans_weights.values_mut() {
                if *w == k {
                    *w = Weight::all();
                }
            }
        }

        // 3. Minimization (Merge equivalent states)
        let mut signature_map: HashMap<StateSignature, StateID> = HashMap::new();
        let mut old_to_new: Vec<Option<StateID>> = vec![None; self.states.len()];
        let mut new_states = DWAStates::default();

        // Process in the same order (bottom-up) to ensure targets are already mapped
        for &u in order.iter() {
            // Build signature
            let fw = self.states[u].final_weight.clone();

            let mut transitions = Vec::new();
            for (&lbl, &old_target) in &self.states[u].transitions {
                if let Some(w) = self.states[u].trans_weights.get(&lbl) {
                    // Map old target to new target
                    // Since we process bottom-up, targets should be processed unless there's a cycle.
                    // (Method is minimize_acyclic, assuming no cycles).
                    // If cycle exists, this might panic or use unmapped None.
                    // For robustness, if unmapped, it might be a back-edge in a cycle?
                    // But we rely on acyclic property.
                    if let Some(new_target) = old_to_new[old_target] {
                        transitions.push((lbl, new_target, w.clone()));
                    } else {
                        // This implies a cycle or broken logic for acyclic assumption.
                        // We'll treat it as a dead end or map to self? 
                        // Let's assume strict acyclic and mapped.
                        // However, if the target is 'u' itself (self-loop), it's a cycle.
                        // We will just skip (effectively deleting the transition) or panic?
                        // Let's skip safely.
                    }
                }
            }

            // Sort transitions to canonicalize signature
            transitions.sort_by(|a, b| a.0.cmp(&b.0));

            let sig = StateSignature {
                final_weight: fw,
                transitions,
            };

            if let Some(&existing_id) = signature_map.get(&sig) {
                old_to_new[u] = Some(existing_id);
            } else {
                let new_id = new_states.add_state();
                // Copy data to new state
                new_states[new_id].final_weight = sig.final_weight.clone();
                new_states[new_id].state_weight = None; // Normalized
                for (lbl, target, w) in &sig.transitions {
                    new_states[new_id].transitions.insert(*lbl, *target);
                    new_states[new_id].trans_weights.insert(*lbl, w.clone());
                }

                signature_map.insert(sig, new_id);
                old_to_new[u] = Some(new_id);
            }
        }

        // 4. Update DWA
        let old_start = self.body.start_state;
        let start_push = if old_start < push_weights.len() {
            push_weights[old_start].clone()
        } else {
            Weight::all()
        };

        // If the graph was empty or something
        let new_start = if let Some(ns) = old_to_new.get(old_start).and_then(|x| *x) {
            ns
        } else {
            // Create a dead/empty start state if original was invalid or lost
            new_states.add_state()
        };

        self.states = new_states;
        self.body.start_state = new_start;

        // Apply the push weight of the old start state to the new start state
        // This preserves the global weight of the automata.
        self.apply_weight_inplace(&start_push);
    }

    fn dfs_post_order(&self, u: StateID, visited: &mut Vec<bool>, order: &mut Vec<StateID>) {
        visited[u] = true;
        if let Some(state) = self.states.0.get(u) {
            for &v in state.transitions.values() {
                if v < visited.len() && !visited[v] {
                    self.dfs_post_order(v, visited, order);
                }
            }
        }
        order.push(u);
    }
}

// Ensure `Weight` implements Eq, Hash for signature map?
// `Weight` is likely complex. If it doesn't implement Hash, we can't use HashMap easily.
// Since `Weight` is in `common` and we can't see it, we assume it derives `PartialEq, Eq`.
// For `Hash`, it might be missing. If so, `StateSignature` needs a workaround.
// However, since we cannot modify `Weight`, let's assume it is Hashable or use BTreeMap for the signature map if Weight is Ord.
// If Weight is not Hash/Ord, we are in trouble.
// Given `DWA` uses `BTreeMap<Label, Weight>` for trans_weights implies Weight is likely Clone + ...? No, Key is Label.
// If Weight is not comparable, we can't canonicalize.
// Let's assume Weight implements `PartialEq` and `Eq`.
// We will use a Vec linear scan if Hash is not available? No, that's O(N^2).
// User requirement implies "essentially bitsets". Bitsets usually Hash.

// Since I cannot check `Weight` traits, I will wrap `Weight` in a helper if needed,
// but `StateSignature` deriving Hash requires `Weight` to implement Hash.
// If compilation fails due to missing Hash, the user needs to add it to Weight.