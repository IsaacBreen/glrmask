//! DWA minimization passes.

mod prune_unreachable;
mod prune_dead_ends;
mod push_weights;
mod push_to_initial;
mod residuated_push;
mod loosen_weights;
mod minimize;
mod rebuild;

use super::common::{Partition, MAX_OPTIMIZE_ITERATIONS};
use crate::precompute4::weighted_automata::common::BENCHMARK_DEBUG;
use crate::precompute4::weighted_automata::dwa::DWA;

use rustfst::algorithms::minimize_with_config;
use rustfst::prelude::MinimizeConfig;

use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushWeights,
    PushWeightsToInitial,
    ResidualPush,
    Minimize,
}

impl DwaPass {
    pub fn is_enabled(&self) -> bool {
        match self {
            DwaPass::PruneUnreachable => std::env::var("DWA_DISABLE_PRUNE_UNREACHABLE").map(|v| v != "1").unwrap_or(true),
            DwaPass::PruneDeadEnds => std::env::var("DWA_DISABLE_PRUNE_DEAD_ENDS").map(|v| v != "1").unwrap_or(true),
            DwaPass::PushWeights => std::env::var("DWA_DISABLE_PUSH_WEIGHTS").map(|v| v != "1").unwrap_or(true),
            DwaPass::PushWeightsToInitial => std::env::var("DWA_DISABLE_PUSH_WEIGHTS_TO_INITIAL").map(|v| v != "1").unwrap_or(true),
            DwaPass::ResidualPush => std::env::var("DWA_DISABLE_RESIDUAL_PUSH").map(|v| v != "1").unwrap_or(true),
            DwaPass::Minimize => std::env::var("DWA_DISABLE_MINIMIZE").map(|v| v != "1").unwrap_or(true),
        }
    }
}

impl DWA {
    pub fn minimize(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        if BENCHMARK_DEBUG {
            let initial_states = self.states.len();
            let mut internal = self.clone();
            let internal_start = std::time::Instant::now();
            internal.minimize_internal();
            let internal_time = internal_start.elapsed();
            let internal_states = internal.states.len();

            let mut rustfst = self.clone();
            let rustfst_start = std::time::Instant::now();
            rustfst.minimize_with_rustfst_full();
            let rustfst_time = rustfst_start.elapsed();
            let rustfst_states = rustfst.states.len();

            if internal_time + rustfst_time > std::time::Duration::from_secs(1) {
                let state_cmp = match internal_states.cmp(&rustfst_states) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };
                let time_cmp = match internal_time.cmp(&rustfst_time) {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Equal => "=",
                    std::cmp::Ordering::Greater => ">",
                };

                crate::debug!(6, "[DWA Minimize({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.minimize_internal();
        }
    }

    /// Performs linear-time optimizations only (Pruning, Weight Pushing).
    /// Skips the expensive O(N log N) or O(N^2) state minimization.
    /// Useful for template generation where we just want a clean graph quickly.
    pub fn minimize_lightweight(&mut self) {
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
                    DwaPass::PruneUnreachable => self.prune_unreachable(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals(),
                    DwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    DwaPass::ResidualPush => self.residuated_push(),
                    DwaPass::Minimize => false,
                };
                changed_in_iteration |= pass_changed;
            }
            if !changed_in_iteration {
                break;
            }
            if iter_num > 0 && iter_num % 100 == 0 {
                crate::debug!(4, "DWA minimize_lightweight iteration {} still changing", iter_num);
            }
        }
    }

    /// Performs a single pass of all optimization passes including minimize.
    /// Unlike minimize(), this does NOT iterate until fixpoint - it runs each pass once.
    /// Useful for terminal DWAs where we want minimize but don't need full convergence.
    pub fn minimize_single_pass(&mut self) {
        if self.states.len() == 0 {
            return;
        }

        self.prune_unreachable();
        self.push_weights_into_transitions_and_finals();
        self.push_weights_to_initial();
        self.residuated_push();
        self.loosen_weights_for_minimize();
        self.prune_dead_ends();
        self.minimize_states();
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = DWA::from_rustfst(&fst);
    }

    pub fn minimize_with_rustfst_full(&mut self) -> bool {
        self.prune_unreachable();
        self.push_weights_into_transitions_and_finals();
        self.push_weights_to_initial();
        self.residuated_push();
        self.loosen_weights_for_minimize();
        self.prune_dead_ends();
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = DWA::from_rustfst(&fst);
        true
    }

    /// Push weights toward initial state using rustfst's push algorithm.
    /// This normalizes the FST so each state's outgoing weights "sum" to one.
    pub fn minimize_internal(&mut self) -> bool {
        let initial_num_states = self.states.len();
        if initial_num_states > 1000 {
            crate::debug!(6, "[DWA::minimize] Starting minimization. Initial stats: {}", self.stats());
        }
        let mut total_changed = false;

        let ordering = &[
            DwaPass::PruneUnreachable,
            DwaPass::PushWeights,
            DwaPass::PushWeightsToInitial,
            DwaPass::ResidualPush,
            DwaPass::PruneDeadEnds,
            DwaPass::Minimize,
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
                    DwaPass::PruneUnreachable => self.prune_unreachable(),
                    DwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    DwaPass::PushWeights => self.push_weights_into_transitions_and_finals(),
                    DwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    DwaPass::ResidualPush => self.residuated_push(),
                    DwaPass::Minimize => {
                        self.loosen_weights_for_minimize();
                        let changed = self.minimize_states();
                        if changed && initial_num_states > 1000 {
                            crate::debug!(6, "[DWA::minimize] After minimize (iter {}): {}", iter_num, self.stats());
                        }
                        changed
                    },
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
            crate::debug!(6, "[DWA::minimize] Minimization finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        }
        total_changed
    }
}
