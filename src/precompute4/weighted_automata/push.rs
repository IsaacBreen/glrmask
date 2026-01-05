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
                // This strips away "junk" bits that don't matter, canonicalizing the edge.
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
        // This canonicalizes final weights so equivalent states become identical
        // NOTE: Skip the start state - it has no incoming transitions to tighten,
        // so loosening its final weight would change semantics.
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
            // Note: If state was not final (fw = None), it stays non-final.
            // The "garbage" ¬d[q] only matters for the actual final weight intersection.
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

    /// Compute forward potentials f[q] for all states.
    ///
    /// f[q] = all weights that can reach state q from the start state.
    ///
    /// For the start state: f[start] = Weight::all() (any token can "be at" start)
    /// For other states: f[q] = ⋃_{(p,σ,q) ∈ δ} (f[p] ∩ w(p→q))
    pub(crate) fn compute_forward_potentials(&self) -> Vec<Weight> {
        let n = self.states.len();
        if n == 0 {
            return vec![];
        }

        let start = self.body.start_state;

        // Initialize: f[start] = all, f[others] = empty
        let mut f: Vec<Weight> = (0..n)
            .map(|q| {
                if q == start {
                    Weight::all()
                } else {
                    Weight::zeros()
                }
            })
            .collect();

        // Build reverse graph: for each state q, list of (predecessor, label, weight)
        let mut preds: Vec<Vec<(usize, Label, Weight)>> = vec![Vec::new(); n];
        for (p, st) in self.states.0.iter().enumerate() {
            for (&label, &target) in &st.transitions {
                if target < n {
                    let w = st
                        .trans_weights
                        .get(&label)
                        .cloned()
                        .unwrap_or_else(Weight::all);
                    preds[target].push((p, label, w));
                }
            }
        }

        // Fixed-point iteration: propagate forwards until stable
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = usize::MAX;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            // Process in forward order (helps convergence for DAGs)
            for q in 0..n {
                if q == start {
                    continue; // f[start] is always Weight::all()
                }

                let mut new_f = Weight::zeros();

                // Add contributions from incoming edges
                for (p, _label, w) in &preds[q] {
                    // Contribution: f[p] ∩ w(p→q)
                    let contribution = &f[*p] & w;
                    new_f |= &contribution;
                }

                if new_f != f[q] {
                    f[q] = new_f;
                    changed = true;
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            crate::debug!(1, "Warning: forward potential computation did not converge");
        }

        f
    }

    /// Bidirectional weight refinement: adjusts weights based on reachability analysis.
    ///
    /// This pass computes:
    /// - f[q] = forward potentials (weights that can reach state q from start)
    /// - d[q] = backward potentials (weights that can reach acceptance from q)
    /// - useful[q] = f[q] ∩ d[q] (weights valid for complete accepting runs through q)
    ///
    /// Behavior controlled by BIDIR_LOOSEN env var:
    /// - BIDIR_LOOSEN=0 or unset (tighten): Intersect with useful[q] = f[q] ∩ d[q]
    ///   - final'(q) = final(q) ∩ useful[q]
    ///   - w'(q→r) = w(q→r) ∩ useful[q]
    /// - BIDIR_LOOSEN=1 (loosen): Union with complement of f[q] (Forward Loosening)
    ///   - MUST BE ACYCLIC (checked). If cyclic, does nothing.
    ///   - final'(q) = final(q) ∪ ¬f[q]
    ///   - w'(q→r) = w(q→r) ∪ ¬f[q]
    ///
    /// Returns true if any weights were changed.
    pub fn bidirectional_weight_refinement(&mut self) -> bool {
        let n = self.states.len();
        if n == 0 {
            return false;
        }

        // Check env var for mode
        let loosen = std::env::var("BIDIR_LOOSEN")
            .map(|v| v == "1")
            .unwrap_or(false);

        // For loosening, we only support acyclic graphs
        if loosen && self.is_cyclic() {
            crate::debug!(4, "Skipping bidirectional loosening because DWA is cyclic");
            return false;
        }

        // Compute potentials
        let f = self.compute_forward_potentials();
        
        // For tightening, we use useful = f ∩ d. For loosening, we use f.
        let useful: Vec<Weight> = if loosen {
             Vec::new() // Not used in this mode directly
        } else {
             let d = self.compute_backward_potentials();
             (0..n).map(|q| &f[q] & &d[q]).collect()
        };

        let mut changed = false;
        let start_state = self.body.start_state;

        for q in 0..n {
            // Determine the mask to apply
            // Tighten: intersect with useful[q]
            // Loosen: union with complement of f[q]
            let (tighten_mask, loosen_mask) = if loosen {
                (None, Some(f[q].complement()))
            } else {
                (Some(&useful[q]), None)
            };

            // Skip start state in loosen mode to preserve semantics (though f[start]=ALL, so !f[start]=EMPTY, so no-op anyway)
            if loosen && q == start_state {
                continue;
            }

            // Skip if tightening and useful is ALL - nothing to refine
            if !loosen && *tighten_mask.unwrap() == Weight::all() {
                continue;
            }

            // Adjust final weight
            if let Some(fw) = &self.states[q].final_weight {
                let new_fw = if loosen {
                    fw | loosen_mask.as_ref().unwrap()
                } else {
                    fw & tighten_mask.unwrap()
                };
                if new_fw != *fw {
                    if !loosen && new_fw.is_empty() {
                        self.states[q].final_weight = None;
                    } else {
                        self.states[q].final_weight = Some(new_fw);
                    }
                    changed = true;
                }
            }

            // Adjust outgoing transitions
            let labels: Vec<Label> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                let w = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);

                let new_w = if loosen {
                    &w | loosen_mask.as_ref().unwrap()
                } else {
                    &w & tighten_mask.unwrap()
                };

                if !loosen && new_w.is_empty() {
                    self.states[q].transitions.remove(&label);
                    self.states[q].trans_weights.remove(&label);
                    changed = true;
                } else if self.states[q].trans_weights.get(&label) != Some(&new_w) {
                    self.states[q].trans_weights.insert(label, new_w);
                    changed = true;
                }
            }
        }

        changed
    }
}

