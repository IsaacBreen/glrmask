//! Boolean Semiring Weight Pushing for DWAs.
//!
//! Since our semiring (sets with ∪ as plus, ∩ as times) has no division operation,
//! we use an alternative approach based on backward potentials and residuation.
//!
//! ## Algorithm
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
//! **Phase 3: Reweight Final Weights**
//! ```text
//! final'(q) = final(q) ∪ ¬d[q]
//! ```

use crate::dwa_i32::common::{Label, Weight, weight_all, weight_complement};
use crate::dwa_i32::dwa::DWA;

impl DWA {
    /// Push weights toward start state to enable state merging.
    pub fn residuated_push_cyclic(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Phase 1: Compute backward potentials d[q]
        let d = self.compute_backward_potentials_cyclic();
        let mut changed = false;

        // Phase 2: Reweight transitions
        for q in 0..n {
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
                    .unwrap_or_else(weight_all);

                let d_target = &d[target];
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
        let start_state = self.body.start_state;
        for q in 0..n {
            if q == start_state {
                continue;
            }
            let d_q = &d[q];
            let complement_d_q = weight_complement(d_q);

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
    pub(crate) fn compute_backward_potentials_cyclic(&self) -> Vec<Weight> {
        let n = self.states.len();
        
        let mut d: Vec<Weight> = (0..n)
            .map(|q| {
                self.states[q]
                    .final_weight
                    .clone()
                    .unwrap_or_else(Weight::zeros)
            })
            .collect();

        let mut changed = true;
        let mut iterations: usize = 0;

        while changed {
            changed = false;
            iterations += 1;

            for q in (0..n).rev() {
                let mut new_d = self.states[q]
                    .final_weight
                    .clone()
                    .unwrap_or_else(Weight::zeros);

                for (&label, &target) in &self.states[q].transitions {
                    if target >= n {
                        continue;
                    }

                    let w = self.states[q]
                        .trans_weights
                        .get(&label)
                        .cloned()
                        .unwrap_or_else(weight_all);

                    let contribution = &w & &d[target];
                    new_d |= &contribution;
                }

                if new_d != d[q] {
                    d[q] = new_d;
                    changed = true;
                }
            }
        }

        // Note: iterations variable available for debugging if convergence issues arise
        let _ = iterations;

        d
    }
}
