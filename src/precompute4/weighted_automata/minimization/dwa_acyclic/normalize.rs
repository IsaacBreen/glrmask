//! Canonical weight normalization for acyclic DWA minimization.
//!
//! Uses live-set trimming for overlap-compatible merging:
//! W(p,a) := W(p,a) ∩ live(target)
//!
//! This removes dead tokens from edges, enabling states with different
//! final/edge weights to merge when differences are on non-overlapping tokens.

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::DWA;

impl DWA {
    /// Normalize weights by trimming to live sets.
    ///
    /// For every transition p -a-> q:
    ///   W(p,a) := W(p,a) ∩ live(q)
    ///
    /// This removes dead tokens from edges, which enables overlap-compatible merging.
    /// Returns the live set of the start state (for potential use in post-processing).
    pub fn normalize_weights(&mut self, live: &[Weight]) -> Weight {
        let n = self.states.len();
        if n == 0 {
            return Weight::zeros();
        }

        let start = self.body.start_state;
        let start_live = if start < n {
            live[start].clone()
        } else {
            Weight::zeros()
        };

        // Trim each transition's weight to live(target)
        for q in 0..n {
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                let live_target = &live[target];

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    // Trim: W(p,a) := W(p,a) ∩ live(target)
                    *tw = tw.clone() & live_target;
                }
            }
        }

        start_live
    }

    /// Apply the initial forbidden set by folding it into start state's outgoing edges.
    /// (Kept for compatibility but may not be needed with new algorithm)
    pub fn apply_initial_forbidden(&mut self, _initial_live: &Weight) {
        // With live-set trimming, this is already handled by the trimming step
        // No additional work needed
    }

    /// Optional: Relax edges back to include dead tokens (for canonical form).
    /// W(p,a) := W(p,a) | dead(target)
    pub fn relax_edges(&mut self, live: &[Weight]) {
        let n = self.states.len();
        if n == 0 {
            return;
        }

        for q in 0..n {
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                // dead(target) = complement of live(target)
                let dead_target = live[target].complement();

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    // Relax: W(p,a) := W(p,a) | dead(target)
                    *tw = tw.clone() | &dead_target;
                }
            }
        }
    }
}
