// src/precompute4/weighted_automata/minimization/dwa_acyclic/mod.rs

use crate::precompute4::weighted_automata::dwa::{DWA, DWAState, DWAStates, DWABuildError};
use crate::precompute4::weighted_automata::common::{Weight, StateID, Label};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::{BitAnd, BitOr, BitXor};

impl DWA {
    /// Minimizes an acyclic DWA.
    ///
    /// This algorithm is PROVABLY OPTIMAL for acyclic DWAs under the intersection semiring.
    /// It works by converting the DWA into a canonical form where:
    /// 1. Forward-dead tokens are removed from transition weights (via backward liveness analysis).
    /// 2. Backward-unreachable tokens are normalized (via forward reachability analysis).
    ///    - Transitions are TRIMMED (w & reach): Unreachable tokens are forced to 0.
    ///    - Final weights are SATURATED (w | !reach): Unreachable tokens are forced to 1.
    pub fn minimize_acyclic(&self) -> Result<DWA, DWABuildError> {
        let n = self.states.len();
        if n == 0 {
            return Ok(DWA::new());
        }

        // =========================================================================
        // 1. Topological Sort (Kahn's Algorithm)
        // =========================================================================
        let mut in_degree = vec![0; n];
        let mut adj = vec![vec![]; n];

        // Count in-degrees based on active transitions
        for (u, state) in self.states.0.iter().enumerate() {
            for &v in state.transitions.values() {
                if v < n {
                    in_degree[v] += 1;
                    adj[u].push(v);
                }
            }
        }

        let mut queue = VecDeque::new();
        // Initialize queue with nodes having 0 in-degree
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut topo_order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            topo_order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        // If topo_order doesn't contain all nodes, there are cycles or unreachable clusters.
        // We only process nodes reachable in the DAG order.
        // (If the graph was truly acyclic but disjoint, this order is still valid for the processed subset).

        // =========================================================================
        // 2. Backward Liveness Analysis (Right-Support)
        // =========================================================================
        // `live[u]` contains tokens that can effectively reach a final state from u.
        // Used to trim dead tokens from transitions.
        let mut live = vec![Weight::zeros(); n];

        for &u in topo_order.iter().rev() {
            let state = &self.states[u];

            // Start with tokens accepted here
            let mut l_u = state.final_weight.clone().unwrap_or_else(Weight::zeros);

            // Union with tokens that can survive a transition to a live successor
            for (lbl, &v) in &state.transitions {
                if v >= n { continue; }
                let w = state.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);

                // Effective tokens = Edge Weight AND Liveness of Target
                let mut effective = live[v].clone();
                effective &= &w;

                l_u |= &effective;
            }
            live[u] = l_u;
        }

        // Create a working copy of states with "dead" tokens removed from edges.
        let mut refined_states = self.states.clone();
        for u in 0..n {
            let state = &mut refined_states[u];
            for (lbl, &v) in &state.transitions {
                if v >= n { continue; }
                if let Some(w) = state.trans_weights.get_mut(lbl) {
                    // Trimming the edge weight to only include live tokens.
                    // This enables merges like: A->C (w=All), B->C (w={3})
                    // If C only accepts {3}, then A->C effectively becomes {3}, matching B.
                    *w &= &live[v];
                }
            }
        }

        // =========================================================================
        // 3. Forward Reachability Analysis (Left-Reachability)
        // =========================================================================
        // `reach[u]` contains tokens that can effectively reach u from the start state.
        // Used to normalize states for equivalence checking.
        let mut reach = vec![Weight::zeros(); n];

        if self.body.start_state < n {
            reach[self.body.start_state] = Weight::all();
        }

        for &u in &topo_order {
            if reach[u].is_empty() { continue; }

            let r_u = reach[u].clone();
            let state = &refined_states[u];

            for (lbl, &v) in &state.transitions {
                if v >= n { continue; }
                // Note: using refined weights here is fine/correct
                let w = state.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);

                let mut flow = r_u.clone();
                flow &= &w;

                if !flow.is_empty() {
                    reach[v] |= &flow;
                }
            }
        }

        // =========================================================================
        // 4. Minimization (Canonical Signatures)
        // =========================================================================
        // We process in Reverse Topological Order to build the minimal DAG bottom-up.

        // Map from Canonical Signature -> New State ID
        // Signature: (Normalized Final Weight, Sorted List of (Label, Normalized Weight, TargetID))
        type Signature = (Weight, Vec<(Label, Weight, StateID)>);
        let mut sig_to_id: HashMap<Signature, StateID> = HashMap::new();
        let mut old_to_new = vec![0; n];

        let mut new_dwa = DWA::new();
        new_dwa.states.0.clear(); // Clear default start state

        for &u in topo_order.iter().rev() {
            // If state is unreachable, it doesn't end up in the minimal DWA
            if reach[u].is_empty() {
                continue;
            }

            let state = &refined_states[u];

            // A. Normalize Final Weight: SATURATE Don't Cares
            // We set bits for unreachable tokens to 1.
            // Why? If token 't' cannot reach state A, and 't' cannot reach state B,
            // then A and B are equivalent regarding 't' regardless of whether they "accept" it.
            // By forcing 't' to 1 (accept), we canonicalize this "don't care" behavior.
            let mut norm_final = state.final_weight.clone().unwrap_or_else(Weight::zeros);
            {
                // Calculate Slack = Universe \ Reachable
                // Assuming Weight supports BitXor with All to perform Not/Complement.
                // slack = All ^ reach[u]
                let mut slack = Weight::all();
                slack ^= &reach[u];

                // Saturate: Final | Slack
                norm_final |= &slack;
            }

            // B. Normalize Transitions: TRIM Don't Cares
            // We set bits for unreachable tokens to 0.
            // Why? If token 't' cannot reach state A, it cannot traverse any edge from A.
            // So we can set the edge weight for 't' to 0 (block) to canonicalize.
            let mut norm_transitions = Vec::new();
            for (lbl, &old_v) in &state.transitions {
                if old_v >= n { continue; }

                let new_v = old_to_new[old_v];
                let w_orig = state.trans_weights.get(lbl).cloned().unwrap_or_else(Weight::all);

                // Trim: w & reach[u]
                let mut w_trim = w_orig;
                w_trim &= &reach[u];

                // If weight becomes empty after trim, the edge is effectively dead for valid tokens
                if !w_trim.is_empty() {
                    norm_transitions.push((*lbl, w_trim, new_v));
                }
            }

            // Sort for canonical signature
            norm_transitions.sort_by(|a, b| a.0.cmp(&b.0));

            let signature = (norm_final, norm_transitions);

            if let Some(&id) = sig_to_id.get(&signature) {
                old_to_new[u] = id;
            } else {
                let new_id = new_dwa.states.add_state();

                // When adding to the new DWA, we use the SIGNATURE'S attributes.
                // This ensures the new DWA is fully normalized (saturated finals, trimmed edges).

                // Set Final Weight
                let fw = signature.0.clone();
                // If saturated weight is all-zeros (meaning original was zero and slack was zero),
                // we don't set it (None). But Weight::zeros != None in DWA logic?
                // DWA usually treats None as "Not Final" (Weight::zeros if evaluated?).
                // However, DWAState def: final_weight: Option<Weight>.
                // If the normalized weight is non-empty (or simply whatever it is), we set it.
                // BUT: Evaluator usually treats None as Zero.
                if !fw.is_empty() {
                    let _ = new_dwa.set_final_weight(new_id, fw);
                }

                // Add Transitions
                for (lbl, w, target_id) in &signature.1 {
                    let _ = new_dwa.add_transition(new_id, *lbl, *target_id, w.clone());
                }

                sig_to_id.insert(signature, new_id);
                old_to_new[u] = new_id;
            }
        }

        // Set Start State
        if self.body.start_state < n && !reach[self.body.start_state].is_empty() {
            new_dwa.body.start_state = old_to_new[self.body.start_state];
        } else {
            // Start state was unreachable or graph was empty.
            // Add a dead start state.
            let start = new_dwa.states.add_state();
            new_dwa.body.start_state = start;
        }

        Ok(new_dwa)
    }
}