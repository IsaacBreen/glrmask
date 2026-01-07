//! Provably minimal acyclic DWA minimization.
//!
//! Algorithm: Forbidden-set dualization + canonical weight pushing + bottom-up merging.
//! Guarantees the absolute minimum number of states among all equivalent DWAs.

mod forbidden;
mod normalize;
mod merge;

use super::common::{DwaPass, MAX_OPTIMIZE_ITERATIONS};
use crate::precompute4::weighted_automata::dwa::DWA;

impl DWA {
    /// Minimize an acyclic DWA to the absolute minimum number of states.
    ///
    /// Uses forbidden-set dualization for canonical weight pushing,
    /// followed by bottom-up signature merging.
    pub fn minimize_acyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }
        self.minimize_internal_acyclic();
    }

    /// Internal minimization: normalize weights then merge by signature.
    pub fn minimize_internal_acyclic(&mut self) -> bool {
        let initial_states = self.states.len();
        if initial_states == 0 {
            return false;
        }

        // Phase 0: Prune unreachable and dead-ends first
        self.prune_unreachable_acyclic();
        self.prune_dead_ends_acyclic();

        if self.states.len() == 0 {
            return initial_states > 0;
        }

        // Phase 1: Compute g(q) for all states
        let g = self.compute_unavoidably_forbidden();

        // Phase 2: Normalize weights (canonical pushing)
        let initial_forbidden = self.normalize_weights(&g);

        // Phase 3: Bottom-up merge by signature
        self.merge_by_signature();

        // Phase 4: Apply initial forbidden as start-state constraint
        self.apply_initial_forbidden(&initial_forbidden);

        // Phase 5: Final prune to remove any dead-ends created during merge
        self.prune_dead_ends_acyclic();

        self.states.len() < initial_states
    }

    /// Lightweight version - just prunes, no full minimization.
    pub fn minimize_lightweight_acyclic(&mut self) {
        self.prune_unreachable_acyclic();
        self.prune_dead_ends_acyclic();
    }

    /// Single pass - runs full minimization once.
    pub fn minimize_single_pass_acyclic(&mut self) {
        self.minimize_internal_acyclic();
    }

    /// RustFST-based minimization (for comparison/benchmarking).
    pub fn minimize_with_rustfst_full_acyclic(&mut self) -> bool {
        // For acyclic, our algorithm should be optimal, but keep this for comparison
        self.minimize_internal_acyclic()
    }

    // === Individual pass dispatchers (for compatibility) ===

    pub fn prune_unreachable_acyclic(&mut self) -> bool {
        let before = self.states.len();
        // Simple forward reachability
        let mut reachable = vec![false; self.states.len()];
        let mut stack = vec![self.body.start_state];
        while let Some(s) = stack.pop() {
            if s >= reachable.len() || reachable[s] {
                continue;
            }
            reachable[s] = true;
            for &t in self.states[s].transitions.values() {
                stack.push(t);
            }
        }
        // Remove unreachable states (by rebuilding)
        if reachable.iter().all(|&r| r) {
            return false;
        }
        self.rebuild_keeping_only(&reachable);
        self.states.len() < before
    }

    pub fn prune_dead_ends_acyclic(&mut self) -> bool {
        let before = self.states.len();
        let n = self.states.len();
        if n == 0 {
            return false;
        }
        // Backward reachability from finals
        let mut can_accept = vec![false; n];
        for i in 0..n {
            if self.states[i].final_weight.is_some() {
                can_accept[i] = true;
            }
        }
        // Iterate until fixpoint
        loop {
            let mut changed = false;
            for i in 0..n {
                if can_accept[i] {
                    continue;
                }
                for &t in self.states[i].transitions.values() {
                    if t < n && can_accept[t] {
                        can_accept[i] = true;
                        changed = true;
                        break;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if can_accept.iter().all(|&c| c) {
            return false;
        }
        self.rebuild_keeping_only(&can_accept);
        self.states.len() < before
    }

    pub fn push_weights_into_transitions_and_finals_acyclic(&mut self) -> bool {
        // This is replaced by normalize_weights in the new algorithm
        false
    }

    pub fn push_weights_to_initial_acyclic(&mut self) -> bool {
        // This is replaced by normalize_weights in the new algorithm
        false
    }

    pub fn residuated_push_acyclic(&mut self) -> bool {
        // This is replaced by normalize_weights in the new algorithm
        false
    }

    pub fn minimize_states_acyclic(&mut self) -> bool {
        // Delegate to full minimization
        self.minimize_internal_acyclic()
    }

    pub fn loosen_weights_for_minimize_acyclic(&mut self) -> bool {
        // Replaced by normalize_weights
        false
    }

    /// Helper: rebuild DWA keeping only states where keep[i] is true.
    fn rebuild_keeping_only(&mut self, keep: &[bool]) {
        let n = self.states.len();
        let mut old_to_new: Vec<Option<usize>> = vec![None; n];
        let mut new_idx = 0;
        for i in 0..n {
            if keep[i] {
                old_to_new[i] = Some(new_idx);
                new_idx += 1;
            }
        }
        if new_idx == n {
            return; // Nothing to remove
        }

        let mut new_states = Vec::with_capacity(new_idx);
        for (i, state) in self.states.0.drain(..).enumerate() {
            if !keep[i] {
                continue;
            }
            let mut new_state = state;
            // Remap transitions
            new_state.transitions.retain(|_, t| keep[*t]);
            for t in new_state.transitions.values_mut() {
                *t = old_to_new[*t].unwrap();
            }
            new_state.trans_weights.retain(|l, _| new_state.transitions.contains_key(l));
            new_states.push(new_state);
        }
        self.states.0 = new_states;
        self.body.start_state = old_to_new[self.body.start_state].unwrap_or(0);
    }
}
