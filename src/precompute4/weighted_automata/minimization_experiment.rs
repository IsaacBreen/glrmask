#![allow(dead_code)]

use super::common::optimize_debug;
use super::dwa::DWA;
use super::nwa::NWA;
use super::minimization::{DwaPass, NwaPass, MAX_OPTIMIZE_ITERATIONS};
use std::collections::HashSet;

const DWA_PASS_ORDERINGS: &[&[DwaPass]] = &[
    &[DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::Minimize],
    &[DwaPass::Minimize, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::PushWeights],
    &[DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::Minimize],
    &[DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneUnreachable],
    &[DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PushWeights],
    &[DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::Minimize],
    &[DwaPass::PruneUnreachable, DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PushWeights],
    &[DwaPass::PruneUnreachable, DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::Minimize],
    &[DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::Minimize, DwaPass::PushWeights],
    &[DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::Minimize],
    &[DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PruneUnreachable, DwaPass::PushWeights],
    &[DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneUnreachable],
    &[DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::Minimize],
    &[DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PruneUnreachable],
    &[DwaPass::PushWeights, DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable],
    &[DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeights],
    &[DwaPass::Minimize, DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::PruneUnreachable],
    &[DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds],
    &[DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeightsToInitial],
];

pub fn run_dwa_optimization_experiment(dwa: &mut DWA) {
    let initial_clone = dwa.clone();
    let initial_states = dwa.states.len();
    println!("[DWA Optimize] Starting experiment with {} states.", initial_states);

    let mut best_result: Option<(DWA, std::time::Duration, usize)> = None;

    for (i, &ordering) in DWA_PASS_ORDERINGS.iter().enumerate() {
        let mut current_dwa = initial_clone.clone();
        let start_time = std::time::Instant::now();
        let mut iterations = 0;
        let mut timed_out = false;
        let mut last_changing_passes: Vec<DwaPass> = Vec::new();

        loop {
            if iterations >= MAX_OPTIMIZE_ITERATIONS {
                timed_out = true;
                break;
            }
            iterations += 1;

            let mut changed_in_iteration = false;
            let mut current_changing_passes: Vec<DwaPass> = Vec::new();
            for &pass in ordering {
                let changed = match pass {
                    DwaPass::PruneUnreachable => current_dwa.prune_unreachable(),
                    DwaPass::PruneDeadEnds => current_dwa.prune_dead_ends(),
                    DwaPass::PushWeights => current_dwa.push_weights_into_transitions_and_finals(),
                    DwaPass::PushWeightsToInitial => current_dwa.push_weights_to_initial(),
                    DwaPass::ResidualPush => current_dwa.residuated_push(),
                    DwaPass::Minimize => current_dwa.minimize_states(),
                };
                if changed {
                    current_changing_passes.push(pass);
                }
                changed_in_iteration |= changed;
            }
            if !changed_in_iteration {
                last_changing_passes.clear();
                break;
            } else {
                last_changing_passes = current_changing_passes;
            }
        }

        let elapsed = start_time.elapsed();
        let final_states = current_dwa.states.len();

        let ordering_str = format!("{:?}", ordering);
        let timeout_str = if timed_out {
            format!(" (TIMED OUT, changing: {:?})", last_changing_passes)
        } else {
            "".to_string()
        };
        println!("[DWA Optimize] Ordering #{}: {}, Time: {:.2?}, States: {}{}", i, ordering_str, elapsed, final_states, timeout_str);

        if !timed_out && best_result.as_ref().map_or(true, |(_, best_time, best_states)| {
            final_states < *best_states || (final_states == *best_states && elapsed < *best_time)
        }) {
            best_result = Some((current_dwa, elapsed, final_states));
        }
    }

    if let Some((best_dwa, _, _)) = best_result {
        *dwa = best_dwa;
    }
}

const NWA_PASS_ORDERINGS: &[&[NwaPass]] = &[
    &[NwaPass::PruneUnreachable, NwaPass::PushFinalWeights, NwaPass::CompressTransitions, NwaPass::PruneDeadEnds, NwaPass::Minimize],
    &[NwaPass::Minimize, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::PushFinalWeights, NwaPass::CompressTransitions],
    &[NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::Minimize],
    &[NwaPass::PruneUnreachable, NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PruneDeadEnds, NwaPass::Minimize],
    &[NwaPass::Minimize, NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds],
    &[NwaPass::PushFinalWeights, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::CompressTransitions, NwaPass::Minimize],
    &[NwaPass::PruneUnreachable, NwaPass::PushFinalWeights, NwaPass::Minimize, NwaPass::CompressTransitions, NwaPass::PruneDeadEnds],
    &[NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::Minimize, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds],
    &[NwaPass::PushFinalWeights, NwaPass::CompressTransitions, NwaPass::Minimize, NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds],
    &[NwaPass::PruneUnreachable, NwaPass::PruneDeadEnds, NwaPass::CompressTransitions, NwaPass::PushFinalWeights, NwaPass::PushWeightsToInitial, NwaPass::Minimize],
];

pub fn run_nwa_optimization_experiment(nwa: &mut NWA) {
    let initial_clone = nwa.clone();
    let initial_states = nwa.states.len();
    println!("[NWA Optimize] Starting experiment with {} states.", initial_states);

    let mut best_result: Option<(NWA, std::time::Duration, usize)> = None;

    for (i, &ordering) in NWA_PASS_ORDERINGS.iter().enumerate() {
        let mut current_nwa = initial_clone.clone();
        let start_time = std::time::Instant::now();
        let mut iterations = 0;
        let mut timed_out = false;
        let mut last_changing_passes: Vec<NwaPass> = Vec::new();

        loop {
            if iterations >= MAX_OPTIMIZE_ITERATIONS {
                timed_out = true;
                break;
            }
            iterations += 1;

            let mut changed_in_iteration = false;
            let mut current_changing_passes: Vec<NwaPass> = Vec::new();
            for &pass in ordering {
                let changed = match pass {
                    NwaPass::PruneUnreachable => current_nwa.prune_unreachable(),
                    NwaPass::PruneDeadEnds => current_nwa.prune_dead_ends(),
                    NwaPass::PushFinalWeights => current_nwa.push_final_weights_along_epsilons(),
                    NwaPass::PushWeightsToInitial => current_nwa.push_weights_to_initial(),
                    NwaPass::CompressTransitions => current_nwa.compress_transitions(),
                    NwaPass::Minimize => current_nwa.minimize_states(),
                    NwaPass::RmEpsilon => { current_nwa.rm_epsilon(); true },
                    NwaPass::MinimizeRustfst => current_nwa.minimize_with_rustfst_full(),
                };
                if changed {
                    current_changing_passes.push(pass);
                }
                changed_in_iteration |= changed;
            }
            if !changed_in_iteration {
                last_changing_passes.clear();
                break;
            } else {
                last_changing_passes = current_changing_passes;
            }
        }

        let elapsed = start_time.elapsed();
        let final_states = current_nwa.states.len();

        let ordering_str = format!("{:?}", ordering);
        let timeout_str = if timed_out {
            format!(" (TIMED OUT, changing: {:?})", last_changing_passes)
        } else {
            "".to_string()
        };
        println!("[NWA Optimize] Ordering #{}: {}, Time: {:.2?}, States: {}{}", i, ordering_str, elapsed, final_states, timeout_str);

        if !timed_out && best_result.as_ref().map_or(true, |(_, best_time, best_states)| {
            final_states < *best_states || (final_states == *best_states && elapsed < *best_time)
        }) {
            best_result = Some((current_nwa, elapsed, final_states));
        }
    }

    if let Some((best_nwa, _, _)) = best_result {
        *nwa = best_nwa;
    }
}

#[derive(Clone, Debug)]
pub struct DeterminizeAndMinimizeConfig {
    pub nwa_passes: Vec<NwaPass>,
    pub dwa_passes: Vec<DwaPass>,
}

impl NWA {
    pub fn determinize_and_minimize(mut self, context: &str) -> DWA {
        if self.states.len() > 1000 && optimize_debug() {
            return Self::run_determinize_and_minimize_experiment(self, context);
        }

        // Production configs based on experiments
        let config = match context {
            "TerminalDWA" => DeterminizeAndMinimizeConfig {
                // Full pipeline for Terminal DWA construction (precompute1)
                // Current best: NWA minimize_rustfst → compress → rm_epsilon → determinize → DWA minimize
                // Results: 14647 → 5904 → 5904 → 889 → 189 states
                nwa_passes: vec![NwaPass::MinimizeRustfst, NwaPass::CompressTransitions, NwaPass::RmEpsilon],
                dwa_passes: vec![DwaPass::Minimize],
            },
            "Precompute1" => DeterminizeAndMinimizeConfig {
                // OPTIMIZATION: Skip Minimize to save ~420ms - Precompute1 is just input to precompute4
                // The final DWA will be minimized, so intermediate minimization is redundant.
                nwa_passes: vec![NwaPass::PruneDeadEnds, NwaPass::PruneUnreachable, NwaPass::CompressTransitions],
                dwa_passes: vec![DwaPass::PruneDeadEnds],
            },
            "FinalDWA" => DeterminizeAndMinimizeConfig {
                // Full pipeline for Parser DWA (finalize_and_optimize_and_determinize)
                // Includes minimize to get optimal state count
                nwa_passes: vec![],
                dwa_passes: vec![DwaPass::PruneDeadEnds, DwaPass::Minimize],
            },
            "SuperDWA" => DeterminizeAndMinimizeConfig {
                // Fallback / Default for SuperDWA (was not large enough to trigger experiment in test)
                // Using a balanced approach
                nwa_passes: vec![NwaPass::CompressTransitions],
                dwa_passes: vec![DwaPass::PruneDeadEnds, DwaPass::Minimize],
            },
            _ => DeterminizeAndMinimizeConfig {
                // Default fallback
                nwa_passes: vec![NwaPass::CompressTransitions],
                dwa_passes: vec![
                    DwaPass::PruneDeadEnds,
                    DwaPass::Minimize,
                    DwaPass::PushWeights,
                    DwaPass::PushWeightsToInitial,
                    DwaPass::PruneUnreachable,
                ],
            }
        };
        Self::determinize_and_minimize_with_config(&mut self, config)
    }

    pub fn determinize_and_minimize_with_config(&mut self, config: DeterminizeAndMinimizeConfig) -> DWA {
        // Run NWA passes
        for pass in config.nwa_passes {
            match pass {
                NwaPass::PruneUnreachable => { self.prune_unreachable(); },
                NwaPass::PruneDeadEnds => { self.prune_dead_ends(); },
                NwaPass::PushFinalWeights => { self.push_final_weights_along_epsilons(); },
                NwaPass::PushWeightsToInitial => { self.push_weights_to_initial(); },
                NwaPass::CompressTransitions => { self.compress_transitions(); },
                NwaPass::Minimize => { self.minimize_states(); },
                NwaPass::RmEpsilon => { self.rm_epsilon(); },
                NwaPass::MinimizeRustfst => { self.minimize_with_rustfst_full(); },
            }
        }
        crate::debug!(5, "NWA minimization: {} states, {} transitions, {} ranges ({} interned)", 
            self.states.len(), self.states.num_transitions(), self.num_ranges(), self.num_ranges_interned());

        let det_start = std::time::Instant::now();
        let mut dwa = self.determinize();
        let det_time = det_start.elapsed();
        crate::debug!(5, "Determinization: {} states, {} transitions, {} ranges ({} interned) in {:.2?}", 
            dwa.states.len(), dwa.states.num_transitions(), dwa.num_ranges(), dwa.num_ranges_interned(), det_time);

        // Run DWA passes
        for pass in config.dwa_passes.clone() {
            let pass_start = std::time::Instant::now();
            match pass {
                DwaPass::PruneUnreachable => { dwa.prune_unreachable(); },
                DwaPass::PruneDeadEnds => { dwa.prune_dead_ends(); },
                DwaPass::PushWeights => { dwa.push_weights_into_transitions_and_finals(); },
                DwaPass::PushWeightsToInitial => { dwa.push_weights_to_initial(); },
                DwaPass::ResidualPush => { dwa.residuated_push(); },
                DwaPass::Minimize => { dwa.minimize_states(); },
            }
            let pass_time = pass_start.elapsed();
            if pass_time.as_millis() > 50 {
                crate::debug!(5, "DWA Pass {:?}: {} states, {} transitions, {} ranges ({} interned) in {:.2?}", 
                    pass, dwa.states.len(), dwa.states.num_transitions(), dwa.num_ranges(), dwa.num_ranges_interned(), pass_time);
            }
        }
        dwa
    }

    pub fn run_determinize_and_minimize_experiment(self, context: &str) -> DWA {
        let initial_states = self.states.len();
        println!("[Det&Min Experiment] [{}] Starting experiment with {} NWA states.", context, initial_states);

        // Define interesting NWA sequences
        let nwa_configs: Vec<Vec<NwaPass>> = vec![
            vec![], // Baseline: no NWA minimization
            vec![NwaPass::CompressTransitions],
            vec![NwaPass::PruneUnreachable, NwaPass::CompressTransitions],
            vec![NwaPass::PruneDeadEnds, NwaPass::PruneUnreachable, NwaPass::CompressTransitions],
            vec![NwaPass::CompressTransitions, NwaPass::Minimize],
            vec![NwaPass::PushFinalWeights, NwaPass::CompressTransitions],
            // Add more aggressive ones
            vec![NwaPass::PruneUnreachable, NwaPass::CompressTransitions, NwaPass::Minimize],
        ];

        // Define interesting DWA sequences
        let dwa_configs: Vec<Vec<DwaPass>> = vec![
            vec![DwaPass::PruneDeadEnds, DwaPass::Minimize, DwaPass::PushWeights, DwaPass::PushWeightsToInitial, DwaPass::PruneUnreachable], // Standard
            vec![DwaPass::Minimize],
            vec![DwaPass::PruneDeadEnds, DwaPass::Minimize],
            vec![DwaPass::PushWeights, DwaPass::Minimize],
            vec![DwaPass::Minimize, DwaPass::PushWeights],
            vec![DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::Minimize],
        ];

        let mut best_result: Option<(DWA, std::time::Duration, usize)> = None;
        let mut best_config_idx = (0, 0);

        let initial_nwa = self; // moved here

        for (n_idx, nwa_pass_seq) in nwa_configs.iter().enumerate() {
            for (d_idx, dwa_pass_seq) in dwa_configs.iter().enumerate() {
                let mut nwa_clone = initial_nwa.clone();

                let start_time = std::time::Instant::now();

                let config = DeterminizeAndMinimizeConfig {
                    nwa_passes: nwa_pass_seq.clone(),
                    dwa_passes: dwa_pass_seq.clone(),
                };

                let dwa = Self::determinize_and_minimize_with_config(&mut nwa_clone, config);

                let elapsed = start_time.elapsed();
                let final_states = dwa.states.len();

                println!("[Det&Min Experiment] [{}] Config N#{}-D#{}: NWA={:?} | DWA={:?} -> Time: {:.2?}, States: {}",
                         context, n_idx, d_idx, nwa_pass_seq, dwa_pass_seq, elapsed, final_states);

                if best_result.as_ref().map_or(true, |(_, best_time, best_states)| {
                    // Prefer fewer states, then faster time
                    final_states < *best_states || (final_states == *best_states && elapsed < *best_time)
                }) {
                    best_result = Some((dwa, elapsed, final_states));
                    best_config_idx = (n_idx, d_idx);
                }
            }
        }

        println!("[Det&Min Experiment] [{}] Winner: Config N#{}-D#{}", context, best_config_idx.0, best_config_idx.1);
        best_result.unwrap().0
    }
}