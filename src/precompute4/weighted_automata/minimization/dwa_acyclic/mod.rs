use crate::precompute4::weighted_automata::common::{Label, StateID, Weight};
use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWABuildError};
use std::collections::{BTreeMap, HashMap};

// A unique signature representing the future behavior of a state.
// Used to identify equivalent states during minimization.
// We assume Weight implements Ord/PartialOrd (common for bitsets/wrappers).
// If not, this would require a custom hash/eq implementation.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct StateSignature {
    // The final weight of the state (if any)
    final_weight: Option<Weight>,
    // Sorted list of transitions: (Label, TargetStateID, EdgeWeight)
    // Note: The TargetStateID refers to the ID in the *new* (minimized) DWA.
    transitions: Vec<(Label, StateID, Weight)>,
}

impl DWA {
    /// Minimizes an acyclic Deterministic Weighted Automaton (DWA).
    ///
    /// This algorithm works by processing states in reverse topological order (leaves to root).
    /// It performs two key operations simultaneously:
    /// 1. **Weight Pushing (Left-Pushing):** It extracts the common constraints ("state weights")
    ///    from a state and its future paths, "lifting" them to the incoming transitions.
    ///    This normalizes the state, making states with different local weights but identical
    ///    structures equivalent (e.g., merging the "Diamond" structure).
    /// 2. **Deduplication:** It maintains a registry of unique state signatures. If a state's
    ///    normalized signature matches an existing one, it reuses the existing state.
    ///
    /// The algorithm assumes the semiring is (PowerSet, Union, Intersection), where weights
    /// act as filters (Intersection) on paths.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // 1. Compute Reverse Topological Order
        // Since it's a DWA (Directed Graph), and we assume acyclic, we can use DFS post-order.
        let mut order = Vec::with_capacity(self.states.len());
        let mut visited = vec![false; self.states.len()];
        let mut visiting = vec![false; self.states.len()]; // Cycle detection check

        for i in 0..self.states.len() {
            if !visited[i] {
                self.topo_visit(i, &mut visited, &mut visiting, &mut order);
            }
        }
        // `order` is now in post-order (children before parents).

        // 2. Prepare for reconstruction
        // Map: OldStateID -> (NewStateID, LiftedWeight)
        // The LiftedWeight is the weight extracted from the state that must be applied
        // to any incoming edge pointing to this state.
        let mut state_map: Vec<Option<(StateID, Weight)>> = vec![None; self.states.len()];

        // Registry: Signature -> NewStateID
        // Used to find existing equivalent states.
        let mut register: BTreeMap<StateSignature, StateID> = BTreeMap::new();

        let mut new_dwa = DWA::new();
        // The `new()` creates a start state 0. We will build the graph and then set the start properly.
        // Actually, let's clear the states of new_dwa so we have full control.
        new_dwa.states.0.clear();
        // We will push states into new_dwa as we create them.

        // 3. Process states in reverse topological order
        for &old_u in order.iter() {
            let old_state = &self.states[old_u];

            // A. Construct the minimized transitions
            // We collect transitions, updating targets to NewIDs and adjusting weights based on children's lifts.
            let mut transitions = Vec::new();
            let mut union_of_futures = Weight::zeros(); // Used for calculating local lift

            for (&lbl, &old_v) in &old_state.transitions {
                let original_edge_weight = old_state.trans_weights.get(&lbl).cloned().unwrap_or_else(Weight::all);

                // Retrieve the mapped child info
                // If old_v is not in the map, it means the graph has a cycle or logic error
                // (topo sort should guarantee visitation).
                let (new_v, child_lift) = state_map[old_v].clone().expect("Child not visited in reverse topo order");

                // The effective weight of this path segment is (EdgeWeight & ChildLift).
                let mut effective_edge_weight = original_edge_weight;
                effective_edge_weight &= &child_lift;

                // Accumulate for local lift calculation (Union of all effective outgoing paths)
                union_of_futures |= &effective_edge_weight;

                transitions.push((lbl, new_v, effective_edge_weight));
            }

            // Sort transitions by label to ensure canonical signature
            transitions.sort_by(|a, b| a.0.cmp(&b.0));

            // B. Determine the "Lift" for the current state
            // The lift is the constraint that applies to ALL paths through this node.
            // Lift = StateWeight & (Union(Transitions) U FinalWeight)
            // If the state is not final, FinalWeight is effectively Empty for the union (unless it's all-pass?).
            // In a filter context:
            //   - If we stop here, we are constrained by FinalWeight.
            //   - If we go through, we are constrained by Union(Transitions).
            //   - We are always constrained by StateWeight.
            // So we lift: StateWeight & (FinalWeight U Union(Transitions))

            let mut downstream_constraint = union_of_futures;
            if let Some(fw) = &old_state.final_weight {
                downstream_constraint |= fw;
            } else if transitions.is_empty() {
                // If no transitions and not final, this is a dead state.
                // Downstream is empty.
            }

            // Retrieve current StateWeight (default is All)
            let current_sw = old_state.state_weight.clone().unwrap_or_else(Weight::all);

            // Calculate Lift
            let mut lift = current_sw;
            lift &= &downstream_constraint;

            // C. Normalize the state for the signature
            // The normalized state will have StateWeight = None (effectively All/Identity),
            // because we moved the restrictive `lift` to the parent.
            // The FinalWeight and EdgeWeights in the signature are kept as is (calculated above).
            // Note: We cannot "divide" the lift out of the edge weights because we are using bitsets/intersection.
            // However, by standardizing StateWeight to All, we allow structural merging.

            // Adjust Final Weight for signature (it remains valid for the node)
            let sig_fw = old_state.final_weight.clone();

            let signature = StateSignature {
                final_weight: sig_fw,
                transitions: transitions.clone(),
            };

            // D. Check Register (Minimization / Deduplication)
            let new_id = if let Some(&existing_id) = register.get(&signature) {
                existing_id
            } else {
                // Create new state
                let new_id = new_dwa.states.add_state();
                let new_state = &mut new_dwa.states[new_id];

                new_state.state_weight = None; // Normalized
                new_state.final_weight = signature.final_weight.clone();

                for (lbl, target, wt) in signature.transitions.clone() {
                    new_state.transitions.insert(lbl, target);
                    new_state.trans_weights.insert(lbl, wt);
                }

                register.insert(signature, new_id);
                new_id
            };

            // E. Store mapping for parents
            state_map[old_u] = Some((new_id, lift));
        }

        // 4. Update the DWA body (Start State)
        let old_start = self.body.start_state;

        // Handle case where graph might be empty or start state unreachable
        if let Some(Some((new_start, start_lift))) = state_map.get(old_start) {
            new_dwa.body.start_state = *new_start;

            // The start state might have a lifted weight.
            // Since there are no parents to push to, we must apply it back to the start state.
            new_dwa.apply_weight_inplace(start_lift);
        } else {
            // Fallback for empty/disconnected
            let s = new_dwa.states.add_state();
            new_dwa.body.start_state = s;
        }

        // Replace self
        *self = new_dwa;
    }

    // Standard DFS for Topological Sort (Post-Order)
    fn topo_visit(
        &self,
        u: usize,
        visited: &mut Vec<bool>,
        visiting: &mut Vec<bool>,
        order: &mut Vec<usize>
    ) {
        visited[u] = true;
        visiting[u] = true;

        if let Some(state) = self.states.0.get(u) {
            for &v in state.transitions.values() {
                if visiting[v] {
                    // Cycle detected - strictly speaking, this algorithm is for acyclic.
                    // We proceed, but results might not be optimal or correct for the cycle part.
                    // In a production env, we might panic or return Result.
                } else if !visited[v] {
                    self.topo_visit(v, visited, visiting, order);
                }
            }
        }

        visiting[u] = false;
        order.push(u);
    }
}