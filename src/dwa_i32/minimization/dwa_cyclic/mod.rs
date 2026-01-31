//! DWA minimization passes for cyclic automata.
//!
//! This module provides cyclic-specific DWA minimization and optimization passes.
//! The primary algorithm is partition refinement which is guaranteed to preserve
//! semantics, though it may not achieve optimal state count.

mod prune_unreachable;
mod prune_dead_ends;
mod push_weights;
mod push_to_initial;
mod residuated_push;
mod loosen_weights;
mod minimize;
mod rebuild;

use super::common::{Partition, MAX_OPTIMIZE_ITERATIONS, DwaPass};
use crate::dwa_i32::common::BENCHMARK_DEBUG;
use crate::dwa_i32::dwa::DWA;

use rustfst::algorithms::minimize_with_config;
use rustfst::prelude::MinimizeConfig;

use std::collections::HashSet;

impl DWA {
    /// Minimizes a cyclic DWA using partition refinement.
    /// 
    /// This is the safe, conservative approach that preserves semantics.
    /// It may not achieve optimal state count but is guaranteed correct.
    pub fn minimize_cyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        // First prune unreachable and dead-end states
        self.prune_unreachable_cyclic();
        self.prune_dead_ends_cyclic();
        
        // Then use simple partition refinement - guaranteed to preserve semantics
        self.minimize_states_cyclic();
    }

    /// Performs linear-time optimizations only (Pruning, Weight Pushing).
    /// Skips the expensive O(N log N) or O(N^2) state minimization.
    /// Useful for template generation where we just want a clean graph quickly.
    pub fn minimize_lightweight_cyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        let ordering = &[
            DwaPass::PruneUnreachable,
            DwaPass::PushWeights,
            DwaPass::PushWeightsToInitial,
            DwaPass::ResidualPush,
            DwaPass::PruneDeadEnds,
        ];

        for iter_num in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut changed_in_iteration = false;
            for &pass in ordering {
                if !pass.is_enabled() {
                    continue;
                }
                let pass_changed = match pass {
                    DwaPass::PruneUnreachable => self.prune_unreachable_cyclic(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends_cyclic(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals_cyclic(),
                    DwaPass::PushWeightsToInitial => self.push_weights_to_initial_cyclic(),
                    DwaPass::ResidualPush => self.residuated_push_cyclic(),
                    DwaPass::ExactMinimize => false,
                    DwaPass::FastMinimize => false,
                    DwaPass::RustfstMinimize => false,
                    DwaPass::ConsolidateRanges => self.consolidate_ranges(),
                    DwaPass::TrimWeights => self.trim_weights(),
                };
                changed_in_iteration |= pass_changed;
            }
            if !changed_in_iteration {
                break;
            }
            if iter_num > 0 && iter_num % 100 == 0 {
                crate::debug!(4, "DWA minimize_lightweight_cyclic iteration {} still changing", iter_num);
            }
        }
    }

    /// Performs a single pass of all optimization passes including minimize.
    /// Unlike minimize(), this does NOT iterate until fixpoint - it runs each pass once.
    /// Useful for terminal DWAs where we want minimize but don't need full convergence.
    pub fn minimize_single_pass_cyclic(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        self.prune_unreachable_cyclic();
        self.push_weights_into_transitions_and_finals_cyclic();
        self.push_weights_to_initial_cyclic();
        self.residuated_push_cyclic();
        self.loosen_weights_for_minimize_cyclic();
        self.prune_dead_ends_cyclic();
        self.minimize_states_cyclic();
    }

    pub fn minimize_with_rustfst_cyclic(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = DWA::from_rustfst(&fst);
    }

    pub fn minimize_with_rustfst_full_cyclic(&mut self) -> bool {
        self.prune_unreachable_cyclic();
        self.push_weights_into_transitions_and_finals_cyclic();
        self.push_weights_to_initial_cyclic();
        self.residuated_push_cyclic();
        self.loosen_weights_for_minimize_cyclic();
        self.prune_dead_ends_cyclic();
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = DWA::from_rustfst(&fst);
        true
    }

    /// Push weights toward initial state using rustfst's push algorithm.
    /// This normalizes the FST so each state's outgoing weights "sum" to one.
    pub fn minimize_internal_cyclic(&mut self) -> bool {
        let initial_num_states = self.states.len();
        if initial_num_states > 1000 {
            crate::debug!(6, "[DWA::minimize_cyclic] Starting minimization. Initial stats: {}", self.stats());
        }
        let mut total_changed = false;

        let ordering = &[
            DwaPass::PruneUnreachable,
            DwaPass::PushWeights,
            DwaPass::PushWeightsToInitial,
            DwaPass::ResidualPush,
            DwaPass::PruneDeadEnds,
            DwaPass::ExactMinimize,
            DwaPass::ConsolidateRanges,
        ];

        let all_passes: HashSet<DwaPass> = ordering.iter().copied().collect();
        let mut history: Vec<HashSet<DwaPass>> = vec![all_passes.clone(), all_passes];

        let mut force_all_passes = false;
        let mut converged = false;

        for iter_num in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = HashSet::new();
            let mut changed_in_iteration = false;

            for &pass in ordering {
                if !pass.is_enabled() {
                    continue;
                }

                let recent_activity = history.iter().any(|s| s.contains(&pass));
                if !force_all_passes && !recent_activity && !changed_in_iteration {
                    continue;
                }

                let pass_changed = match pass {
                    DwaPass::PruneUnreachable => self.prune_unreachable_cyclic(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends_cyclic(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals_cyclic(),
                    DwaPass::PushWeightsToInitial => self.push_weights_to_initial_cyclic(),
                    DwaPass::ResidualPush => self.residuated_push_cyclic(),
                    DwaPass::ExactMinimize => {
                        self.loosen_weights_for_minimize_cyclic();
                        let changed = self.minimize_states_cyclic();
                        if changed && initial_num_states > 1000 {
                            crate::debug!(6, "[DWA::minimize_cyclic] After minimize (iter {}): {}", iter_num, self.stats());
                        }
                        changed
                    },
                    DwaPass::FastMinimize => {
                        self.loosen_weights_for_minimize_cyclic();
                        let changed = self.minimize_states_cyclic();
                        if changed && initial_num_states > 1000 {
                            crate::debug!(6, "[DWA::minimize_cyclic] After fast minimize (iter {}): {}", iter_num, self.stats());
                        }
                        changed
                    },
                    DwaPass::RustfstMinimize => self.minimize_with_rustfst_full_cyclic(),
                    DwaPass::ConsolidateRanges => self.consolidate_ranges(),
                    DwaPass::TrimWeights => self.trim_weights(),
                };
                if pass_changed {
                    current_changing_passes.insert(pass);
                }
                changed_in_iteration |= pass_changed;
            }

            history.push(current_changing_passes);
            if history.len() > 2 {
                history.remove(0);
            }

            total_changed |= changed_in_iteration;
            if !changed_in_iteration {
                if force_all_passes {
                    converged = true;
                    break;
                }
                force_all_passes = true;
            } else {
                force_all_passes = false;
            }
        }

        if !converged {
            let last_changes = history.last().map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default();
            crate::debug!(4, "DWA minimization did not converge after {} iterations. Still changing: {:?}", MAX_OPTIMIZE_ITERATIONS, last_changes);
        }

        if initial_num_states > 1000 {
            crate::debug!(6, "[DWA::minimize_cyclic] Minimization finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        }
        total_changed
    }
}
