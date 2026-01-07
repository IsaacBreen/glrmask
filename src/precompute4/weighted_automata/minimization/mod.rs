//! Minimization passes for DWA and NWA.
//!
//! This module provides optimization passes for weighted automata:
//! - Pruning unreachable and dead-end states
//! - Weight pushing (toward start/final)
//! - State minimization via partition refinement

pub mod common;
pub mod dwa_acyclic;
pub mod dwa_cyclic;
pub mod nwa;

pub use common::{Partition, MAX_OPTIMIZE_ITERATIONS, DwaPass};
pub use nwa::NwaPass;

use crate::precompute4::weighted_automata::dwa::DWA;
use crate::precompute4::weighted_automata::Weight;

impl DWA {
    pub fn minimize(&mut self) {
        if self.is_cyclic() {
            self.minimize_cyclic();
        } else {
            self.minimize_acyclic();
        }
    }

    pub fn minimize_internal(&mut self) -> bool {
        if self.is_cyclic() {
            self.minimize_internal_cyclic()
        } else {
            self.minimize_acyclic();
            true
        }
    }

    pub fn minimize_lightweight(&mut self) {
        if self.is_cyclic() {
            self.minimize_lightweight_cyclic();
        } else {
            self.minimize_lightweight_acyclic();
        }
    }

    pub fn minimize_single_pass(&mut self) {
        if self.is_cyclic() {
            self.minimize_single_pass_cyclic();
        } else {
            self.minimize_single_pass_acyclic();
        }
    }

    pub fn minimize_with_rustfst_full(&mut self) -> bool {
        if self.is_cyclic() {
            self.minimize_with_rustfst_full_cyclic()
        } else {
            self.minimize_with_rustfst_full_acyclic()
        }
    }

    // Dispatchers for individual passes
    pub fn prune_unreachable(&mut self) -> bool {
        if self.is_cyclic() {
            self.prune_unreachable_cyclic()
        } else {
            todo!()
        }
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        if self.is_cyclic() {
            self.prune_dead_ends_cyclic()
        } else {
            todo!()
        }
    }

    pub fn push_weights_into_transitions_and_finals(&mut self) -> bool {
        if self.is_cyclic() {
            self.push_weights_into_transitions_and_finals_cyclic()
        } else {
            self.push_weights_into_transitions_and_finals_acyclic()
        }
    }

    pub fn push_weights_to_initial(&mut self) -> bool {
        if self.is_cyclic() {
            self.push_weights_to_initial_cyclic()
        } else {
            self.push_weights_to_initial_acyclic()
        }
    }

    pub fn residuated_push(&mut self) -> bool {
        if self.is_cyclic() {
            self.residuated_push_cyclic()
        } else {
            self.residuated_push_acyclic()
        }
    }

    pub fn minimize_states(&mut self) -> bool {
        if self.is_cyclic() {
            self.minimize_states_cyclic()
        } else {
            self.minimize_states_acyclic()
        }
    }

    pub fn loosen_weights_for_minimize(&mut self) -> bool {
        if self.is_cyclic() {
            self.loosen_weights_for_minimize_cyclic()
        } else {
            self.loosen_weights_for_()
        }
    }


    // ========================================================================
    // LEGACY API COMPATIBILITY
    // ========================================================================

    /// Lightweight version - just prunes, no full minimization.
    pub fn minimize_lightweight_acyclic(&mut self) {
        self.pass0_prune();
        self.minimize();
    }

    /// Single pass - runs full minimization once.
    pub fn minimize_single_pass_acyclic(&mut self) {
        self.minimize_acyclic();
    }

    /// RustFST-based minimization (for comparison/benchmarking).
    pub fn minimize_with_rustfst_full_acyclic(&mut self) -> bool {
        self.minimize_acyclic();
        true
    }

    pub fn push_weights_into_transitions_and_finals_acyclic(&mut self) -> bool {
        false
    }

    pub fn push_weights_to_initial_acyclic(&mut self) -> bool {
        false // Not used in new algorithm
    }

    pub fn residuated_push_acyclic(&mut self) -> bool {
        false // Not used in new algorithm
    }

    pub fn minimize_states_acyclic(&mut self) -> bool {
        self.minimize_acyclic();
        true
    }

    pub fn loosen_weights_for_(&mut self) -> bool {
        false
    }

    // Legacy methods for compatibility with old API
    pub fn compute_live_sets(&self) -> Vec<Weight> {
        let n = self.states.len();
        if n == 0 {
            return vec![];
        }

        // let topo_order = self.reverse_topological_order();
        let topo_order: Vec<_> = todo!();
        let mut b: Vec<Weight> = vec![Weight::zeros(); n];

        for &q in &topo_order {
            let mut b_q = self.states[q]
                .final_weight
                .clone()
                .unwrap_or_else(Weight::zeros);

            for (&label, &target) in &self.states[q].transitions {
                if target >= n {
                    continue;
                }
                let tw = self.states[q]
                    .trans_weights
                    .get(&label)
                    .cloned()
                    .unwrap_or_else(Weight::all);
                b_q = &b_q | &(&tw & &b[target]);
            }

            b[q] = b_q;
        }

        b
    }

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
                    *tw = tw.clone() & live_target;
                }
            }
        }

        start_live
    }

    pub fn merge_by_signature(&mut self, _live: &[Weight]) {

    }

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

                let dead_target = live[target].complement();

                if let Some(tw) = self.states.0[q].trans_weights.get_mut(&label) {
                    *tw = tw.clone() | &dead_target;
                }
            }
        }
    }
}
