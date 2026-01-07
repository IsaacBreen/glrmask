//! Canonical weight normalization for acyclic DWA minimization.
//!
//! Redistributes weights using the CORRECT formula that enables backward pushing:
//! b_norm(q,a) = (b(q,a) ∪ g(target)) \ g(source)
//!
//! This "pulls weight backward" by subtracting g(source) LAST, removing from
//! the edge any forbidden info that's already guaranteed from the source state.

use crate::precompute4::weighted_automata::common::Weight;
use crate::precompute4::weighted_automata::dwa::DWA;

impl DWA {
    /// Normalize weights using the g(q) values computed earlier.
    ///
    /// Returns the initial forbidden set I = g(q0) that should be applied
    /// as a start-state constraint.
    ///
    /// CORRECT Transformation (in forbidden-set domain):
    /// - b_norm(q,a) = (b(q,a) ∪ g(target)) \ g(source)
    /// - bF_norm(q) = bF(q) \ g(q)
    /// - I = g(q0)
    ///
    /// The key insight: subtracting g(source) LAST removes any forbidden info
    /// that's already guaranteed from the source, effectively "pulling weight backward".
    pub fn normalize_weights(&mut self, g: &[Weight]) -> Weight {
        let n = self.states.len();
        if n == 0 {
            return Weight::all();
        }

        let start = self.body.start_state;
        let initial_forbidden = if start < n {
            g[start].clone()
        } else {
            Weight::all()
        };

        // Normalize each state's weights
        for q in 0..n {
            let g_source = &g[q];

            // Normalize final weight: bF_norm = bF \ g(source)
            // In forbidden: bF_norm = bF \ g = bF ∩ complement(g)
            // In allowed: F_norm = complement(bF_norm) = complement(bF ∩ !g) = F ∪ g
            // So: F_norm = F | g (loosen final weight with unavoidably-forbidden tokens)
            if let Some(ref mut fw) = self.states.0[q].final_weight {
                *fw = fw.clone() | g_source;
            }

            // Normalize transition weights
            let labels: Vec<i32> = self.states[q].transitions.keys().copied().collect();
            for label in labels {
                let target = self.states[q].transitions[&label];
                if target >= n {
                    continue;
                }

                let g_target = &g[target];

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    // CORRECT formula in forbidden world:
                    // b_norm = (b ∪ g_target) \ g_source
                    //        = (b ∪ g_target) ∩ complement(g_source)
                    //
                    // Converting to allowed (w = complement(b)):
                    // w_norm = complement(b_norm)
                    //        = complement((b ∪ g_target) ∩ !g_source)
                    //        = complement(b ∪ g_target) ∪ g_source    [De Morgan]
                    //        = (complement(b) ∩ complement(g_target)) ∪ g_source
                    //        = (w ∩ !g_target) ∪ g_source
                    //        = (w & g_target.complement()) | g_source

                    let g_target_complement = g_target.complement();
                    let restricted = tw.clone() & &g_target_complement;
                    *tw = &restricted | g_source;
                }
            }
        }

        initial_forbidden
    }

    /// Apply the initial forbidden set by folding it into start state's outgoing edges.
    ///
    /// In forbidden world: each outgoing edge from start gets ∪ g(start)
    /// In allowed world: each outgoing edge from start gets ∩ complement(g(start))
    pub fn apply_initial_forbidden(&mut self, initial_forbidden: &Weight) {
        if initial_forbidden.is_empty() {
            return; // Nothing to apply
        }

        let start = self.body.start_state;
        if start >= self.states.len() {
            return;
        }

        // I is the forbidden set, so allowed = complement(I)
        let initial_allowed = initial_forbidden.complement();

        // Apply to start state's outgoing transition weights
        let labels: Vec<i32> = self.states[start].transitions.keys().copied().collect();
        for label in labels {
            if let Some(tw) = self.states.0[start].trans_weights.get_mut(&label) {
                *tw = tw.clone() & &initial_allowed;
            }
        }

        // Also apply to start state's final weight if any
        if let Some(ref mut fw) = self.states.0[start].final_weight {
            *fw = fw.clone() & &initial_allowed;
        }
    }
}
