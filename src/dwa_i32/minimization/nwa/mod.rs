//! NWA minimization passes.

mod prune_unreachable;
mod prune_dead_ends;
mod push_final_weights;
mod push_to_initial;
mod compress;
mod minimize;
mod rebuild;

use super::common::{Partition, MAX_OPTIMIZE_ITERATIONS};
use crate::dwa_i32::common::BENCHMARK_DEBUG;
use crate::dwa_i32::nwa::NWA;

use rustfst::algorithms::minimize_with_config;
use rustfst::prelude::MinimizeConfig;

use std::collections::HashSet;
use rustfst::algorithms::rm_epsilon::rm_epsilon;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NwaPass {
    PruneUnreachable,
    PruneDeadEnds,
    PushFinalWeights,
    PushWeightsToInitial,
    CompressTransitions,
    Minimize,
    RmEpsilon,
    MinimizeRustfst,  // Full minimize using rustfst
}

impl NWA {
    pub fn rm_epsilon(&mut self) {
        crate::debug!(6, "[NWA] Removing epsilon transitions...");
        let initial_states = self.states.len();
        let mut total_epsilons = 0;
        for st in &self.states.0 {
            total_epsilons += st.epsilons.len();
        }
        crate::debug!(7, "[NWA] Initial number of states: {}, total epsilon transitions: {}", initial_states, total_epsilons);

        let mut fst = self.to_rustfst();
        rm_epsilon(&mut fst).unwrap();
        *self = NWA::from_rustfst(&fst);

        let final_states = self.states.len();
        let mut final_epsilons = 0;
        for st in &self.states.0 {
            final_epsilons += st.epsilons.len();
        }
        crate::debug!(7, "[NWA] Final number of states: {}, total epsilon transitions: {}", final_states, final_epsilons);
    }

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

                crate::debug!(6, "[NWA Minimize({})] Internal: t={:.2?}, s={} | RustFST: t={:.2?}, s={}. [s: {}, t: {}]", initial_states, internal_time, internal_states, rustfst_time, rustfst_states, state_cmp, time_cmp);
            }

            *self = internal;
        } else {
            self.minimize_internal();
        }
    }

    pub fn minimize_with_rustfst(&mut self) {
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, MinimizeConfig::default().with_allow_nondet(true)).unwrap();
        *self = NWA::from_rustfst(&fst);
    }

    pub fn minimize_with_rustfst_full(&mut self) -> bool {
        let min_config = MinimizeConfig::default().with_allow_nondet(true);
        let mut fst = self.to_rustfst();
        minimize_with_config(&mut fst, min_config).unwrap();
        *self = NWA::from_rustfst(&fst);
        true
    }

    pub fn minimize_internal(&mut self) -> bool {
        crate::debug!(6, "[NWA::minimize] Starting minimization. Initial stats: {}", self.stats());
        let mut total_changed = false;

        let ordering = &[
            NwaPass::PruneUnreachable,
            NwaPass::CompressTransitions,
            NwaPass::PushFinalWeights,
            NwaPass::PushFinalWeights,
            NwaPass::PushWeightsToInitial,
            NwaPass::PruneDeadEnds,
            NwaPass::Minimize,
        ];

        let all_passes: HashSet<NwaPass> = ordering.iter().copied().collect();
        let mut history: Vec<HashSet<NwaPass>> = vec![all_passes.clone(), all_passes];

        let mut force_all_passes = false;
        let mut converged = false;

        for _iter_num in 0..MAX_OPTIMIZE_ITERATIONS {
            let mut current_changing_passes = HashSet::new();
            let mut changed_in_iteration = false;
            for &pass in ordering {
                let recent_activity = history.iter().any(|s| s.contains(&pass));
                if !force_all_passes && !recent_activity && !changed_in_iteration {
                    continue;
                }

                let pass_changed = match pass {
                    NwaPass::PruneUnreachable => self.prune_unreachable(),
                    NwaPass::PruneDeadEnds => self.prune_dead_ends(),
                    NwaPass::PushFinalWeights => self.push_final_weights_along_epsilons(),
                    NwaPass::PushWeightsToInitial => self.push_weights_to_initial(),
                    NwaPass::CompressTransitions => self.compress_transitions(),
                    NwaPass::Minimize => self.minimize_states(),
                    NwaPass::RmEpsilon => { self.rm_epsilon(); true },
                    NwaPass::MinimizeRustfst => self.minimize_with_rustfst_full(),
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
            crate::debug!(4, "NWA minimization did not converge after {} iterations. Still changing: {:?}", MAX_OPTIMIZE_ITERATIONS, last_changes);
        }

        crate::debug!(6, "[NWA::minimize] Minimization finished. Total changed: {}. Final stats: {}", total_changed, self.stats());
        total_changed
    }
}
