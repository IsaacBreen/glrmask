//! Minimization passes for DWA and NWA.
//!
//! This module provides optimization passes for weighted automata:
//! - Pruning unreachable and dead-end states
//! - Weight pushing (toward start/final)
//! - State minimization via partition refinement
//!
//! ## Minimization Algorithm
//!
//! The primary minimization algorithm is the "exact" Diamond-aware algorithm in
//! `dwa_cyclic::minimize_exact`. This algorithm:
//!
//! 1. Uses SCC decomposition to handle cycles
//! 2. Computes "needed sets" (what tokens can reach acceptance from each state)
//! 3. Uses exact graph coloring to find optimal state merging
//! 4. Handles the "Diamond" case where states with disjoint domains can merge
//!
//! This is significantly more effective than traditional partition refinement
//! because it can merge states that have different behavior, as long as their
//! "active" token domains are disjoint.

pub mod common;
mod consolidate_ranges;  // Range consolidation pass (general, works for any DWA)
pub mod dwa_acyclic;
pub mod dwa_cyclic;
pub mod graph_coloring;
pub mod nwa;

pub use common::{Partition, MAX_OPTIMIZE_ITERATIONS, DwaPass};
pub use nwa::NwaPass;

use crate::dwa_i32::dwa::DWA;
use crate::dwa_i32::Weight;

impl DWA {
    /// Apply DWA optimization passes based on a named config.
    /// Config names: "SpecializedDWA", "SpecializedDWALightweight", etc.
    pub fn optimize(&mut self, config_name: &str) {
        let passes = match config_name {
            // Full minimize - good quality but slow for large DWAs
            "SpecializedDWA" => vec![DwaPass::PruneDeadEnds, DwaPass::Minimize],
            // Lightweight - just pruning, faster but larger output
            "SpecializedDWALightweight" => vec![DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable],
            // Single pass minimize - one round of state merging, faster than full
            "SpecializedDWASinglePass" => {
                // Run single pass minimize directly
                if self.is_cyclic() {
                    self.minimize_single_pass_cyclic();
                } else {
                    self.minimize_acyclic();
                }
                return;
            },
            _ => vec![DwaPass::Minimize], // Default: just minimize
        };
        
        for pass in passes {
            match pass {
                DwaPass::PruneUnreachable => { self.prune_unreachable(); },
                DwaPass::PruneDeadEnds => { self.prune_dead_ends(); },
                DwaPass::PushWeights => { self.push_weights_into_transitions_and_finals(); },
                DwaPass::PushWeightsToInitial => { self.push_weights_to_initial(); },
                DwaPass::ResidualPush => { self.residuated_push(); },
                DwaPass::Minimize => { self.minimize_states(); },
                DwaPass::ConsolidateRanges => { self.consolidate_ranges(); },
            }
        }
    }

    /// Minimizes the DWA to its optimal state count.
    /// Dispatches to acyclic or cyclic implementation based on graph structure.
    /// Both paths now use the exact Diamond-aware algorithm with SCC support.
    pub fn minimize(&mut self) {
        // TEMPORARY: Force cyclic version for all DWAs to test correctness
        // TODO: Restore acyclic dispatch once cyclic is verified to work correctly
        // if self.is_cyclic() {
        //     self.minimize_cyclic();
        // } else {
        //     self.minimize_acyclic();
        // }
        self.minimize_cyclic();
    }

    /// Basic pruning without full minimization.
    /// Removes unreachable and dead-end states in O(n) time.
    /// Does NOT do state merging via graph coloring.
    pub fn prune_basic(&mut self) {
        // Use the cyclic versions since they work for acyclic too
        self.prune_unreachable_cyclic();
        self.prune_dead_ends_cyclic();
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
            // Acyclic minimization handles pruning internally
            false
        }
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        if self.is_cyclic() {
            self.prune_dead_ends_cyclic()
        } else {
            // Acyclic minimization handles pruning internally
            false
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
        let is_cyc = self.is_cyclic();
        let before = self.states.len();
        crate::debug!(4, "minimize_states: is_cyclic={}, {} states before", is_cyc, before);
        // TEMPORARY: Force cyclic version for all DWAs to test correctness
        // TODO: Restore acyclic dispatch once cyclic is verified to work correctly
        // if is_cyc {
        //     self.minimize_states_cyclic()
        // } else {
        //     self.minimize_acyclic();
        //     true
        // }
        self.minimize_states_cyclic()
    }

    pub fn loosen_weights_for_minimize(&mut self) -> bool {
        if self.is_cyclic() {
            self.loosen_weights_for_minimize_cyclic()
        } else {
            false  // Not used in acyclic algorithm
        }
    }

    // ========================================================================
    // ACYCLIC PASS STUBS  
    // (acyclic algorithm handles these differently or doesn't need them)
    // ========================================================================

    /// RustFST-based minimization (for comparison/benchmarking).
    pub fn minimize_with_rustfst_full_acyclic(&mut self) -> bool {
        self.minimize_acyclic();
        true
    }

    pub fn push_weights_into_transitions_and_finals_acyclic(&mut self) -> bool {
        false  // Acyclic algorithm has weight pushing built-in
    }

    pub fn push_weights_to_initial_acyclic(&mut self) -> bool {
        false  // Not used in acyclic algorithm
    }

    pub fn residuated_push_acyclic(&mut self) -> bool {
        false  // Not used in acyclic algorithm
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
