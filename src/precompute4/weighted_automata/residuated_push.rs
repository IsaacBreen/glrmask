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
    fn compute_backward_potentials(&self) -> Vec<Weight> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precompute4::weighted_automata::common::Label;
    use crate::precompute4::weighted_automata::dwa::{DWABody, DWAStates};

    /// Helper to build a simple DWA for testing.
    fn build_simple_dwa() -> DWA {
        // A --[{1,2}]--> B --[{2}]--> C (final, weight {1,2})
        //
        // Before pushing:
        //   Path weight = {1,2} ∩ {2} ∩ {1,2} = {2}
        //
        // Potentials: ρ(C) = {1,2}, ρ(B) = {2} ∩ {1,2} = {2}, ρ(A) = {1,2} ∩ {2} = {2}
        //
        // After prune-only pushing:
        //   A --[{1,2} ∩ {2}]--> B --[{2} ∩ {1,2}]--> C
        //   = A --[{2}]--> B --[{2}]--> C
        //   Path weight = {2} ∩ {2} ∩ {1,2} = {2} ✓

        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        let w_12 = Weight::from_iter([1, 2]);
        let w_2 = Weight::from_item(2);

        // A -> B on label 0 with weight {1,2}
        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, w_12.clone());

        // B -> C on label 1 with weight {2}
        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, w_2.clone());

        // C is final with weight {1,2}
        states[c].final_weight = Some(w_12.clone());

        DWA {
            body: DWABody { start_state: a },
            states,
        }
    }

    #[test]
    fn test_backward_potentials() {
        let dwa = build_simple_dwa();
        let rho = dwa.compute_backward_potentials();

        // ρ(C) should be {1,2} (the final weight)
        assert_eq!(rho[2], Weight::from_iter([1, 2]));

        // ρ(B) = w(B→C) ∩ ρ(C) = {2} ∩ {1,2} = {2}
        assert_eq!(rho[1], Weight::from_item(2));

        // ρ(A) = w(A→B) ∩ ρ(B) = {1,2} ∩ {2} = {2}
        assert_eq!(rho[0], Weight::from_item(2));
    }

    #[test]
    fn test_residuated_push_prune_only() {
        let mut dwa = build_simple_dwa();

        // Compute original path weight
        let orig_weight = dwa.eval_word_weight(&[0, 1]);
        assert_eq!(orig_weight, Weight::from_item(2));

        dwa.residuated_push_prune_only();

        // After pushing, A->B weight should be tightened to {2}
        assert_eq!(
            dwa.states[0].trans_weights.get(&0),
            Some(&Weight::from_item(2))
        );

        // B->C weight should remain {2}
        assert_eq!(
            dwa.states[1].trans_weights.get(&1),
            Some(&Weight::from_item(2))
        );

        // Path weight should be preserved
        let new_weight = dwa.eval_word_weight(&[0, 1]);
        assert_eq!(new_weight, orig_weight);
    }

    #[test]
    fn test_residuated_push_full() {
        let mut dwa = build_simple_dwa();

        // Compute original path weight
        let orig_weight = dwa.eval_word_weight(&[0, 1]);

        dwa.residuated_push();

        // Path weight should be preserved
        let new_weight = dwa.eval_word_weight(&[0, 1]);
        assert_eq!(new_weight, orig_weight);
    }

    #[test]
    fn test_push_removes_dead_edges() {
        // Build DWA with an edge that leads nowhere useful:
        // A --[{1}]--> B --[{2}]--> C (final, weight {3})
        //
        // Path weight = {1} ∩ {2} ∩ {3} = {} (empty!)
        // So the entire path should be pruned.

        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_item(1));

        states[b].transitions.insert(1, c);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        states[c].final_weight = Some(Weight::from_item(3));

        let mut dwa = DWA {
            body: DWABody { start_state: a },
            states,
        };

        dwa.residuated_push_prune_only();

        // All edges should be removed since path weight is empty
        assert!(dwa.states[0].transitions.is_empty());
        assert!(dwa.states[1].transitions.is_empty());
    }

    #[test]
    fn test_push_with_branching() {
        // Build DWA with branching:
        //       ┌─[{1,2}]─→ B ─[{2}]─→ D (final {2})
        // A ─┤
        //       └─[{1,3}]─→ C ─[{3}]─→ E (final {3})
        //
        // ρ(D) = {2}, ρ(B) = {2}, contribution to ρ(A) from B path: {1,2} ∩ {2} = {2}
        // ρ(E) = {3}, ρ(C) = {3}, contribution to ρ(A) from C path: {1,3} ∩ {3} = {3}
        // ρ(A) = {2} ∪ {3} = {2,3}

        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();
        let d = states.add_state();
        let e = states.add_state();

        // A -> B on label 0 with weight {1,2}
        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_iter([1, 2]));

        // A -> C on label 1 with weight {1,3}
        states[a].transitions.insert(1, c);
        states[a].trans_weights.insert(1, Weight::from_iter([1, 3]));

        // B -> D on label 2 with weight {2}
        states[b].transitions.insert(2, d);
        states[b].trans_weights.insert(2, Weight::from_item(2));

        // C -> E on label 3 with weight {3}
        states[c].transitions.insert(3, e);
        states[c].trans_weights.insert(3, Weight::from_item(3));

        // D is final with weight {2}
        states[d].final_weight = Some(Weight::from_item(2));

        // E is final with weight {3}
        states[e].final_weight = Some(Weight::from_item(3));

        let mut dwa = DWA {
            body: DWABody { start_state: a },
            states,
        };

        // Compute original path weights
        let orig_weight_bd = dwa.eval_word_weight(&[0, 2]);
        let orig_weight_ce = dwa.eval_word_weight(&[1, 3]);

        dwa.residuated_push_prune_only();

        // After pushing:
        // A->B should be tightened to {1,2} ∩ {2} = {2}
        assert_eq!(
            dwa.states[0].trans_weights.get(&0),
            Some(&Weight::from_item(2))
        );

        // A->C should be tightened to {1,3} ∩ {3} = {3}
        assert_eq!(
            dwa.states[0].trans_weights.get(&1),
            Some(&Weight::from_item(3))
        );

        // Path weights should be preserved
        let new_weight_bd = dwa.eval_word_weight(&[0, 2]);
        let new_weight_ce = dwa.eval_word_weight(&[1, 3]);
        assert_eq!(new_weight_bd, orig_weight_bd);
        assert_eq!(new_weight_ce, orig_weight_ce);
    }

    #[test]
    fn test_push_with_cycle() {
        // Build DWA with a cycle:
        // A --[{1,2}]--> B --[{2}]--> B (self-loop)
        //                    └─[{2}]─→ C (final {2})
        //
        // ρ(C) = {2}
        // ρ(B) = {2} ∩ {2} ∪ {2} ∩ ρ(B) -- fixed point is {2}
        // ρ(A) = {1,2} ∩ {2} = {2}

        let mut states = DWAStates::default();
        let a = states.add_state();
        let b = states.add_state();
        let c = states.add_state();

        // A -> B on label 0 with weight {1,2}
        states[a].transitions.insert(0, b);
        states[a].trans_weights.insert(0, Weight::from_iter([1, 2]));

        // B -> B on label 1 with weight {2} (self-loop)
        states[b].transitions.insert(1, b);
        states[b].trans_weights.insert(1, Weight::from_item(2));

        // B -> C on label 2 with weight {2}
        states[b].transitions.insert(2, c);
        states[b].trans_weights.insert(2, Weight::from_item(2));

        // C is final with weight {2}
        states[c].final_weight = Some(Weight::from_item(2));

        let mut dwa = DWA {
            body: DWABody { start_state: a },
            states,
        };

        // Compute original path weights
        let orig_direct = dwa.eval_word_weight(&[0, 2]);
        let orig_loop_once = dwa.eval_word_weight(&[0, 1, 2]);

        dwa.residuated_push_prune_only();

        // A->B should be tightened to {2}
        assert_eq!(
            dwa.states[0].trans_weights.get(&0),
            Some(&Weight::from_item(2))
        );

        // Path weights should be preserved
        assert_eq!(dwa.eval_word_weight(&[0, 2]), orig_direct);
        assert_eq!(dwa.eval_word_weight(&[0, 1, 2]), orig_loop_once);
    }
}
