// src/precompute4/weighted_automata/push.rs

//! Boolean Semiring Weight Pushing for DWAs.
//!
//! Since our semiring (sets with ∪ as plus, ∩ as times) has no division operation,
//! we use an alternative approach based on backward potentials and residuation.
//!
//! ## Algorithm (Corrected)
//!
//! **Phase 1: Compute Backward Potentials d[q]**
//! d[q] = all tokens that can reach acceptance from state q
//! ```text
//! d[q] = final_weight(q) ∪ ⋃_{(q,σ,r) ∈ δ} (w(q→r) ∩ d[r])
//! ```
//!
//! **Phase 2: Reweight Edges**
//! - `w'(q→r) = w(q→r) ∩ d[r]`
//!
//! We tighten all edges to only allow flow that is consistent with the target's future.
//! This effectively "pushes" the constraint early. Since any bits outside d[target]
//! are irrelevant (they die at target anyway), removing them makes edges canonical
//! (edges distiguished only by "dead" bits become identical).
//!
//! **Phase 3: Reweight Final Weights**
//! ```text
//! final'(q) = final(q) ∪ ¬d[q]
//! ```
//!
//! This makes states with identical outgoing structure have identical final weights,
//! enabling minimization to merge them.

use super::common::{Label, Weight};
use super::dwa::DWA;

impl DWA {
    /// Push weights toward start state to enable state merging.
    ///
    /// This algorithm uses backward potentials and residuation to redistribute
    /// weights without requiring division. After pushing:
    /// - States with identical outgoing behavior will have identical weights
    /// - Standard minimization can then merge these states
    ///
    /// Returns true if any weights were changed.
    pub fn residuated_push(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Phase 1: Compute backward potentials d[q]
        // d[q] = all tokens that can reach acceptance from state q
        let d = self.compute_backward_potentials();
        let mut changed = false;

        // Phase 2: Reweight transitions
        for q in 0..n {
            // Collect transitions to avoid borrow issues
            let transitions: Vec<_> = self.states[q]
                .transitions
                .iter()
                .map(|(&label, &target)| (label, target))
                .collect();

            for (label, target) in transitions {
                if target >= n {
                    continue;
                }

                let w = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let d_target = &d[target];

                // Compute new weight: w' = w ∩ d[target]
                // We tighten the edge to only allow flow that is valid for the target's future.
                // This pulls weights from the future back into the transition.
                let new_weight = &w & d_target;

                if new_weight.is_empty() {
                    self.states[q].transitions.remove(&label);
                    self.states[q].trans_weights.remove(&label);
                    changed = true;
                } else if self.states[q].trans_weights.get(&label) != Some(&new_weight) {
                    self.states[q].trans_weights.insert(label, new_weight);
                    changed = true;
                }
            }
        }

        // Phase 3: Reweight final weights
        // final'(q) = final(q) ∪ ¬d[q]
        // This canonicalizes final weights.
        // NOTE: We MUST NOT loosen the start state's final weight if we are tightening edges,
        // because the start state has no incoming edges to "push" into.
        // Actually, loosening the start state's final weight is safe if we only care about
        // the intersection with the initial weight (which is 'all').
        let start_state = self.body.start_state;
        for q in 0..n {
            if q == start_state {
                continue;
            }
            let d_q = &d[q];
            let complement_d_q = d_q.complement();

            if let Some(fw) = &self.states[q].final_weight {
                let new_fw = fw | &complement_d_q;
                if new_fw != *fw {
                    self.states[q].final_weight = Some(new_fw);
                    changed = true;
                }
            }
        }

        changed
    }

    /// Compute backward potentials d[q] for all states.
    ///
    /// d[q] = final_weight(q) ∪ ⋃_{(q,σ,r) ∈ δ} (w(q→r) ∩ d[r])
    ///
    /// This represents all tokens that can reach acceptance from state q.
    pub(crate) fn compute_backward_potentials(&self) -> Vec<Weight> {
        let n = self.states.len();
        
        // Initialize: d[q] = final_weight(q) or empty
        let mut d: Vec<Weight> = (0..n)
            .map(|q| {
                self.states[q]
                    .final_weight
                    .clone()
                    .unwrap_or_else(Weight::zeros)
            })
            .collect();

        // Fixed-point iteration: propagate backwards until stable
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = usize::MAX;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Process in reverse order (helps convergence for DAGs)
            for q in (0..n).rev() {
                let mut new_d = self.states[q]
                    .final_weight
                    .clone()
                    .unwrap_or_else(Weight::zeros);

                // Add contributions from outgoing edges
                for (&label, &target) in &self.states[q].transitions {
                    if target >= n {
                        continue;
                    }

                    let w = self.states[q]
                        .trans_weights
                        .get(&label)
                        .cloned()
                        .unwrap_or_else(Weight::all);

                    // Contribution: w(q→target) ∩ d[target]
                    let contribution = &w & &d[target];
                    new_d |= &contribution;
                }

                if new_d != d[q] {
                    d[q] = new_d;
                    changed = true;
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            crate::debug!(1, "Warning: backward potential computation did not converge");
        }

        d
    }

}

