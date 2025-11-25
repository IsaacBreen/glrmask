#![allow(dead_code)]

use super::common::optimize_debug;
use super::dwa::DWA;
use super::nwa::NWA;
use super::simplification::{DwaPass, NwaPass, MAX_OPTIMIZE_ITERATIONS};
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
pub struct DeterminizeAndSimplifyConfig {
    pub nwa_passes: Vec<NwaPass>,
    pub dwa_passes: Vec<DwaPass>,
}

pub fn determinize_and_simplify(mut nwa: NWA, context: &str) -> DWA {
    if nwa.states.len() > 1000 && optimize_debug() {
        return run_determinize_and_simplify_experiment(nwa, context);
    }

    // Production configs based on experiments
    let config = match context {
        "Precompute1" => DeterminizeAndSimplifyConfig {
            // Best: NWA=[PruneDeadEnds, PruneUnreachable, CompressTransitions] | DWA=[Minimize]
            nwa_passes: vec![NwaPass::PruneDeadEnds, NwaPass::PruneUnreachable, NwaPass::CompressTransitions],
            dwa_passes: vec![DwaPass::Minimize],
        },
        "FinalDWA" => DeterminizeAndSimplifyConfig {
            // Best: NWA=[] | DWA=[PruneDeadEnds, Minimize]
            nwa_passes: vec![],
            dwa_passes: vec![DwaPass::PruneDeadEnds, DwaPass::Minimize],
        },
        "SuperDWA" => DeterminizeAndSimplifyConfig {
            // Fallback / Default for SuperDWA (was not large enough to trigger experiment in test)
            // Using a balanced approach
            nwa_passes: vec![NwaPass::CompressTransitions],
            dwa_passes: vec![DwaPass::PruneDeadEnds, DwaPass::Minimize],
        },
        _ => DeterminizeAndSimplifyConfig {
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
    determinize_and_simplify_with_config(&mut nwa, config)
}

pub fn determinize_and_simplify_with_config(nwa: &mut NWA, config: DeterminizeAndSimplifyConfig) -> DWA {
    // Run NWA passes
    for pass in config.nwa_passes {
        match pass {
            NwaPass::PruneUnreachable => { nwa.prune_unreachable(); },
            NwaPass::PruneDeadEnds => { nwa.prune_dead_ends(); },
            NwaPass::PushFinalWeights => { nwa.push_final_weights_along_epsilons(); },
            NwaPass::PushWeightsToInitial => { nwa.push_weights_to_initial(); },
            NwaPass::CompressTransitions => { nwa.compress_transitions(); },
            NwaPass::Minimize => { nwa.minimize_states(); },
        }
    }

    let mut dwa = nwa.determinize();

    // Run DWA passes
    for pass in config.dwa_passes {
        match pass {
            DwaPass::PruneUnreachable => { dwa.prune_unreachable(); },
            DwaPass::PruneDeadEnds => { dwa.prune_dead_ends(); },
            DwaPass::PushWeights => { dwa.push_weights_into_transitions_and_finals(); },
            DwaPass::PushWeightsToInitial => { dwa.push_weights_to_initial(); },
            DwaPass::Minimize => { dwa.minimize_states(); },
        }
    }
    dwa
}

pub fn run_determinize_and_simplify_experiment(nwa: NWA, context: &str) -> DWA {
    let initial_states = nwa.states.len();
    println!("[Det&Sim Experiment] [{}] Starting experiment with {} NWA states.", context, initial_states);

    // Define interesting NWA sequences
    let nwa_configs: Vec<Vec<NwaPass>> = vec![
        vec![], // Baseline: no NWA simplification
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

    let initial_nwa = nwa; // moved here

    for (n_idx, nwa_pass_seq) in nwa_configs.iter().enumerate() {
        for (d_idx, dwa_pass_seq) in dwa_configs.iter().enumerate() {
            let mut nwa_clone = initial_nwa.clone();

            let start_time = std::time::Instant::now();

            let config = DeterminizeAndSimplifyConfig {
                nwa_passes: nwa_pass_seq.clone(),
                dwa_passes: dwa_pass_seq.clone(),
            };

            let dwa = determinize_and_simplify_with_config(&mut nwa_clone, config);

            let elapsed = start_time.elapsed();
            let final_states = dwa.states.len();

            println!("[Det&Sim Experiment] [{}] Config N#{}-D#{}: NWA={:?} | DWA={:?} -> Time: {:.2?}, States: {}",
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

    println!("[Det&Sim Experiment] [{}] Winner: Config N#{}-D#{}", context, best_config_idx.0, best_config_idx.1);
    best_result.unwrap().0
}
