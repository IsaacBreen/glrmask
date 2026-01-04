// src/precompute4/weighted_automata/residuated_push.rs

//! Residuated weight pushing for DWAs.
//!
//! Classical weight pushing requires division, which doesn't exist for intersection.
//! However, Boolean algebras have **residuation**: the residual `a ⇒ b = ¬a ∪ b`
//! satisfies `a ∩ (a ⇒ b) = b` when `b ⊆ a`.
//!
//! This module implements two variants:
//! - `residuated_push()`: Full pushing with tightening AND loosening
//! - `residuated_push_prune_only()`: Simpler version that only tightens (no ¬ρ(q) term)
//!
//! ## Algorithm
//!
//! **Phase 1: Compute backward potentials ρ(q)**
//! The set of tokens that can reach acceptance from state q:
//! ```text
//! ρ(q) = final_weight(q) ∪ ⋃_{(q,σ,r) ∈ δ} (w(q→r) ∩ ρ(r))
//! ```
//! This is a greatest fixed point (start with ALL, shrink until stable).
//!
//! **Phase 2: Residuated reweighting**
//! ```text
//! w★(q→r) = ¬ρ(q) ∪ (w(q→r) ∩ ρ(r))
//! ```
//! - The `w(q→r) ∩ ρ(r)` term **tightens** by removing tokens that can't reach acceptance from r
//! - The `¬ρ(q)` term **loosens** by adding "don't care" bits for tokens already excluded at q
//!
//! **Phase 3: Absorb initial potential into start state's outgoing edges**
//! ```text
//! w★(start→r) = w★(start→r) ∩ ρ(start)
//! ```

use super::common::Weight;
use super::dwa::DWA;

impl DWA {
    /// Full residuated weight pushing with tightening AND loosening.
    ///
    /// This redistributes weights to make them appear earlier in paths,
    /// enabling faster rejection of invalid tokens during mask computation.
    ///
    /// The algorithm preserves path weights: for any path from start to a final state,
    /// the accumulated weight (intersection of all edge weights) is unchanged.
    pub fn residuated_push(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Phase 1: Compute backward potentials ρ(q)
        // ρ(q) = tokens that can reach acceptance from state q
        let rho = self.compute_backward_potentials();

        // Phase 2: Apply residuated reweighting
        // w★(q→r) = ¬ρ(q) ∪ (w(q→r) ∩ ρ(r))
        for q in 0..n {
            let rho_q = &rho[q];
            let complement_rho_q = rho_q.complement();

            // Collect transitions to avoid borrow issues
            let transitions: Vec<_> = self.states[q]
                .transitions
                .iter()
                .map(|(&label, &target)| (label, target))
                .collect();

            for (label, target) in transitions {
                let w_qr = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let rho_r = &rho[target];

                // Tighten: w(q→r) ∩ ρ(r)
                let tightened = &w_qr & rho_r;

                // Loosen: ¬ρ(q) ∪ tightened
                let new_weight = &complement_rho_q | &tightened;

                if new_weight.is_empty() {
                    // Edge becomes useless - remove it
                    self.states[q].transitions.remove(&label);
                    self.states[q].trans_weights.remove(&label);
                } else {
                    self.states[q].trans_weights.insert(label, new_weight);
                }
            }
        }

        // Phase 3: Absorb initial potential into start state edges
        // w★(start→r) = w★(start→r) ∩ ρ(start)
        let start = self.body.start_state;
        if start < n {
            let rho_start = rho[start].clone();

            let start_transitions: Vec<_> = self.states[start]
                .transitions
                .iter()
                .map(|(&label, &target)| (label, target))
                .collect();

            for (label, _target) in start_transitions {
                if let Some(w) = self.states[start].trans_weights.get_mut(&label) {
                    *w &= &rho_start;
                    if w.is_empty() {
                        self.states[start].transitions.remove(&label);
                        self.states[start].trans_weights.remove(&label);
                    }
                }
            }
        }
    }

    /// Simpler prune-only variant that only tightens edges (no loosening).
    ///
    /// This removes tokens from edge weights that cannot reach acceptance,
    /// without adding "don't care" bits. Always safe and sufficient for early rejection.
    ///
    /// ```text
    /// w'(q→r) = w(q→r) ∩ ρ(r)
    /// ```
    pub fn residuated_push_prune_only(&mut self) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        // Phase 1: Compute backward potentials
        let rho = self.compute_backward_potentials();

        // Phase 2: Apply prune-only reweighting
        // w'(q→r) = w(q→r) ∩ ρ(r)
        for q in 0..n {
            let transitions: Vec<_> = self.states[q]
                .transitions
                .iter()
                .map(|(&label, &target)| (label, target))
                .collect();

            for (label, target) in transitions {
                let w_qr = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let rho_r = &rho[target];

                // Tighten only: w(q→r) ∩ ρ(r)
                let new_weight = &w_qr & rho_r;

                if new_weight.is_empty() {
                    // Edge becomes useless - remove it
                    self.states[q].transitions.remove(&label);
                    self.states[q].trans_weights.remove(&label);
                } else {
                    self.states[q].trans_weights.insert(label, new_weight);
                }
            }
        }
    }

    /// Compute backward potentials ρ(q) for all states.
    ///
    /// ρ(q) = the set of tokens that can reach acceptance from state q
    ///      = final_weight(q) ∪ ⋃_{(q,σ,r) ∈ δ} (w(q→r) ∩ ρ(r))
    ///
    /// This is computed as a greatest fixed point: start with ALL, shrink until stable.
    pub(crate) fn compute_backward_potentials(&self) -> Vec<Weight> {
        let n = self.states.len();

        // Initialize: ρ(q) = ALL for all states
        let mut rho: Vec<Weight> = vec![Weight::all(); n];

        // Compute reverse topological order (or just iterate backwards)
        // For simplicity, we use a worklist algorithm that converges

        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 10000;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Process states in reverse order (helps convergence for acyclic graphs)
            for q in (0..n).rev() {
                // Compute new ρ(q)
                // Start with final_weight if final, else empty
                let mut rho_new = self.states[q]
                    .final_weight
                    .clone()
                    .unwrap_or_else(Weight::zeros);

                // Add contributions from outgoing edges
                // ⋃_{(q,σ,r) ∈ δ} (w(q→r) ∩ ρ(r))
                for (&label, &target) in &self.states[q].transitions {
                    if target >= n {
                        continue;
                    }

                    let w_qr = self.states[q]
                        .trans_weights
                        .get(&label)
                        .cloned()
                        .unwrap_or_else(Weight::all);

                    let contribution = &w_qr & &rho[target];
                    rho_new |= &contribution;
                }

                // Greatest fixed point: we shrink from ALL
                // New value should be a subset of (or equal to) old value
                // We're computing union (supremum), so we need intersection with old
                let rho_intersected = &rho[q] & &rho_new;
                if rho_intersected != rho[q] {
                    rho[q] = rho_intersected;
                    changed = true;
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            crate::debug!(1, "Warning: backward potential computation did not converge after {} iterations", MAX_ITERATIONS);
        }

        rho
    }
}


