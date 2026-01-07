//! Forbidden-set computation for acyclic DWA minimization.
//!
//! Computes g(q) = unavoidably forbidden tokens from state q.
//! This is the set of tokens that will ALWAYS be forbidden regardless of suffix.

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::DWA;
use std::collections::VecDeque;

impl DWA {
    /// Compute g(q) for all states: the unavoidably forbidden tokens from each state.
    ///
    /// Uses reverse topological order DP:
    /// ```
    /// g(q) = bF(q) ∩ ⋂_{a: δ(q,a) defined} (b(q,a) ∪ g(δ(q,a)))
    /// ```
    ///
    /// Where:
    /// - bF(q) = complement of final weight (U if non-final)
    /// - b(q,a) = complement of transition weight
    pub fn compute_unavoidably_forbidden(&self) -> Vec<Weight> {
        let n = self.states.len();
        if n == 0 {
            return vec![];
        }

        // Compute reverse topological order using Kahn's algorithm
        let topo_order = self.reverse_topological_order();

        // Initialize g[q] for all states
        let mut g: Vec<Weight> = vec![Weight::zeros(); n];

        // Process in reverse topological order (sinks first)
        for &q in &topo_order {
            // bF(q) = U \ F(q) if final, else U (meaning "reject everything")
            let b_final = match &self.states[q].final_weight {
                Some(fw) if !fw.is_empty() => fw.complement(),
                _ => Weight::all(), // Non-final or empty final = everything forbidden
            };

            // Start with the "epsilon" case (empty suffix): forbidden = bF(q)
            let mut g_q = b_final;

            // For each outgoing transition, intersect with (b(q,a) ∪ g(target))
            for (&label, &target) in &self.states[q].transitions {
                if target >= n {
                    continue;
                }

                // b(q,a) = complement of transition weight
                let b_trans = self.states[q]
                    .trans_weights
                    .get(&label)
                    .map(|w| w.complement())
                    .unwrap_or_else(Weight::zeros); // No weight = all allowed = none forbidden

                // b(q,a) ∪ g(target)
                let combined = &b_trans | &g[target];

                // Intersect with current g_q
                g_q = &g_q & &combined;
            }

            g[q] = g_q;
        }

        g
    }

    /// Compute reverse topological order (sinks first, sources last).
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

        // If we didn't visit all states, there's a cycle (shouldn't happen for acyclic)
        if order.len() != n {
            crate::debug!(4, "Warning: acyclic DWA has cycle? Only {} of {} states in topo order", order.len(), n);
        }

        order
    }
}
