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
            self.minimize_internal_acyclic()
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
            self.prune_unreachable_acyclic()
        }
    }

    pub fn prune_dead_ends(&mut self) -> bool {
        if self.is_cyclic() {
            self.prune_dead_ends_cyclic()
        } else {
            self.prune_dead_ends_acyclic()
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
            self.loosen_weights_for_minimize_acyclic()
        }
    }
}
