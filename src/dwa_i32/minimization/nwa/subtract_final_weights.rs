//! Subtract final weights from outgoing transitions.
//!
//! For any state with a final weight, subtract that weight from all outgoing transitions.
//! This prunes paths that continue after a word has already been accepted with a given weight.

use crate::dwa_i32::nwa::NWA;

impl NWA {
    /// Subtract final weights from outgoing transitions.
    ///
    /// For any state with a final weight, subtract that weight from all outgoing transitions
    /// (both epsilon and labeled). This operation helps prune paths that continue after
    /// a word has already been accepted with a given weight.
    ///
    /// Returns `true` if any changes were made to the NWA.
    pub fn subtract_final_weights_from_outgoing(&mut self) -> bool {
        let mut changed = false;
        for i in 0..self.states.len() {
            if let Some(final_weight) = self.states[i].final_weight.clone() {
                if final_weight.is_empty() {
                    continue;
                }
                let state = &mut self.states.0[i];

                // Epsilon transitions
                for (_, w) in &mut state.epsilons {
                    let old_w = w.clone();
                    *w -= &final_weight;
                    if *w != old_w {
                        changed = true;
                    }
                }

                // Labeled transitions
                for targets in state.transitions.values_mut() {
                    for (_, w) in targets {
                        let old_w = w.clone();
                        *w -= &final_weight;
                        if *w != old_w {
                            changed = true;
                        }
                    }
                }
            }
        }
        changed
    }
}
