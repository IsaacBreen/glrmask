//! Minimization and determinization configurations for NWA/DWA.
//!
//! This module provides named configurations for different contexts where we need
//! to determinize and/or minimize automata. Each config specifies which NWA and DWA
//! optimization passes to run.
//!
//! Main configs:
//! - `Terminal`: Full pipeline for terminal/lexical DWA
//! - `Template`: Template DWAs built from terminal characterizations  
//! - `Parser`: Final Parser DWA after composition
//! - `Super`: Intermediate composition result
//! - `SpecializedSuper`: Specialized DWAs derived from Super
//!
//! Also includes experimental functions for testing different pass orderings.

#![allow(dead_code)]

use super::common::optimize_debug;
use super::dwa::DWA;
use super::nwa::NWA;
use profiler_macro::{time_it, timeit};
use super::minimization::{DwaPass, NwaPass, MAX_OPTIMIZE_ITERATIONS};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DwaOptimizeConfig {
    SpecializedSuper,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeterminizeAndMinimizeProfile {
    Terminal,
    Template,
    Super,
    SpecializedSuper,
    Parser,
}

impl DeterminizeAndMinimizeProfile {
    pub fn as_dwa_type(self) -> &'static str {
        match self {
            DeterminizeAndMinimizeProfile::Terminal => "terminal",
            DeterminizeAndMinimizeProfile::Template => "template",
            DeterminizeAndMinimizeProfile::Super => "super",
            DeterminizeAndMinimizeProfile::SpecializedSuper => "specialized_super",
            DeterminizeAndMinimizeProfile::Parser => "parser",
        }
    }
}

const DWA_PASS_ORDERINGS: &[&[DwaPass]] = &[
    &[DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::SatMinimize],
    &[DwaPass::SatMinimize, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::PushWeights],
    &[DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::SatMinimize],
    &[DwaPass::PushWeights, DwaPass::SatMinimize, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::SatMinimize, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::SatMinimize, DwaPass::PruneUnreachable],
    &[DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::SatMinimize, DwaPass::PushWeights],
    &[DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::SatMinimize],
    &[DwaPass::PruneUnreachable, DwaPass::SatMinimize, DwaPass::PruneDeadEnds, DwaPass::PushWeights],
    &[DwaPass::PruneUnreachable, DwaPass::SatMinimize, DwaPass::PushWeights, DwaPass::PruneDeadEnds],
    &[DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeights, DwaPass::SatMinimize],
    &[DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::SatMinimize, DwaPass::PushWeights],
    &[DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::SatMinimize],
    &[DwaPass::PruneDeadEnds, DwaPass::SatMinimize, DwaPass::PruneUnreachable, DwaPass::PushWeights],
    &[DwaPass::PruneDeadEnds, DwaPass::SatMinimize, DwaPass::PushWeights, DwaPass::PruneUnreachable],
    &[DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::SatMinimize],
    &[DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::SatMinimize, DwaPass::PruneUnreachable],
    &[DwaPass::PushWeights, DwaPass::SatMinimize, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable],
    &[DwaPass::SatMinimize, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeights],
    &[DwaPass::SatMinimize, DwaPass::PruneDeadEnds, DwaPass::PushWeights, DwaPass::PruneUnreachable],
    &[DwaPass::SatMinimize, DwaPass::PushWeights, DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds],
    &[DwaPass::SatMinimize, DwaPass::PushWeights, DwaPass::PruneDeadEnds, DwaPass::PruneUnreachable, DwaPass::PushWeightsToInitial],
];

pub fn run_dwa_optimization_experiment(dwa: &mut DWA) {
    let initial_clone = dwa.clone();
    let initial_stats = dwa.stats();
    println!("[DWA Optimize] Starting experiment with {}.", initial_stats);

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
                    DwaPass::SatMinimize => current_dwa.minimize_states_sat(),
                    DwaPass::CadicalMinimize => current_dwa.minimize_states_cadical(),
                    DwaPass::DsaturMinimize => current_dwa.minimize_states_dsatur(),
                    DwaPass::ColPackMinimize => current_dwa.minimize_states_colpack(),
                    DwaPass::ColPackVerifiedMinimize => current_dwa.minimize_states_colpack_verified(),
                    DwaPass::FastMinimize => current_dwa.minimize_states_fast(),
                    DwaPass::RustfstMinimize => current_dwa.minimize_with_rustfst_full(),
                    DwaPass::ConsolidateRanges => current_dwa.consolidate_ranges(),
                    DwaPass::TrimWeights => current_dwa.trim_weights(),
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
        let final_stats = current_dwa.stats();
        let final_states = final_stats.states;

        let ordering_str = format!("{:?}", ordering);
        let timeout_str = if timed_out {
            format!(" (TIMED OUT, changing: {:?})", last_changing_passes)
        } else {
            "".to_string()
        };
        println!("[DWA Optimize] Ordering #{}: {}, Time: {:.2?}, Stats: {}{}", i, ordering_str, elapsed, final_stats, timeout_str);

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
    let initial_stats = nwa.stats();
    println!("[NWA Optimize] Starting experiment with {}.", initial_stats);

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
        let final_stats = current_nwa.stats();
        let final_states = final_stats.states;

        let ordering_str = format!("{:?}", ordering);
        let timeout_str = if timed_out {
            format!(" (TIMED OUT, changing: {:?})", last_changing_passes)
        } else {
            "".to_string()
        };
        println!("[NWA Optimize] Ordering #{}: {}, Time: {:.2?}, Stats: {}{}", i, ordering_str, elapsed, final_stats, timeout_str);

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

impl DWA {
    /// Apply DWA optimization passes based on a named config.
    pub fn optimize(&mut self, config: DwaOptimizeConfig) {
        let passes = match config {
            // Full minimize - good quality but slow for large DWAs
            DwaOptimizeConfig::SpecializedSuper => vec![
                DwaPass::PruneDeadEnds,
                DwaPass::FastMinimize,
            ],
        };

        for pass in passes {
            // Check if pass is enabled (e.g., ConsolidateRanges is disabled in weight-heavy mode)
            if !pass.is_enabled() {
                continue;
            }
            match pass {
                DwaPass::PruneUnreachable => { self.prune_unreachable(); },
                DwaPass::PruneDeadEnds => { self.prune_dead_ends(); },
                DwaPass::PushWeights => { self.push_weights_into_transitions_and_finals(); },
                DwaPass::PushWeightsToInitial => { self.push_weights_to_initial(); },
                DwaPass::ResidualPush => { self.residuated_push(); },
                DwaPass::SatMinimize => { self.minimize_states_sat(); },
                DwaPass::CadicalMinimize => { self.minimize_states_cadical(); },
                DwaPass::DsaturMinimize => { self.minimize_states_dsatur(); },
                DwaPass::ColPackMinimize => { self.minimize_states_colpack(); },
                DwaPass::ColPackVerifiedMinimize => { self.minimize_states_colpack_verified(); },
                DwaPass::FastMinimize => { self.minimize_states_fast(); },
                DwaPass::RustfstMinimize => { self.minimize_with_rustfst_full(); },
                DwaPass::ConsolidateRanges => { self.consolidate_ranges(); },
                DwaPass::TrimWeights => { self.trim_weights(); },
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeterminizeAndMinimizeConfig {
    pub nwa_passes: Vec<NwaPass>,
    pub dwa_passes: Vec<DwaPass>,
    pub use_rustfst_determinize: bool,
}

impl DeterminizeAndMinimizeConfig {
    pub fn for_profile(profile: DeterminizeAndMinimizeProfile) -> Self {
        match profile {
            DeterminizeAndMinimizeProfile::Terminal => {
                // Full pipeline for Terminal DWA construction
                let nwa_passes = if std::env::var("SKIP_RUSTFST_MIN").map_or(false, |v| v == "1") {
                    vec![NwaPass::CompressTransitions]
                } else {
                    vec![
                        NwaPass::RmEpsilon,
                        NwaPass::CompressTransitions,
                    ]
                };

                let skip_minimize_before_suffix = std::env::var("SKIP_TERMINAL_DWA_MINIMIZE_BEFORE_SUFFIX")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);

                let mut dwa_passes = match std::env::var("TERMINAL_DWA_PASS")
                    .ok()
                    .map(|v| v.to_ascii_lowercase())
                    .as_deref()
                {
                    Some("fast") => vec![
                        DwaPass::FastMinimize,
                        DwaPass::ConsolidateRanges,
                        DwaPass::TrimWeights,
                    ],
                    Some("colpack") => vec![
                        DwaPass::ColPackMinimize,
                        DwaPass::ConsolidateRanges,
                        DwaPass::TrimWeights,
                    ],
                    Some("colpack_verified") | Some("colpack-verified") => vec![
                        DwaPass::ColPackVerifiedMinimize,
                        DwaPass::ConsolidateRanges,
                        DwaPass::TrimWeights,
                    ],
                    Some(other) => {
                        eprintln!(
                            "WARN: unknown TERMINAL_DWA_PASS='{}', using ColPackMinimize",
                            other
                        );
                        vec![
                            DwaPass::ColPackMinimize,
                            DwaPass::ConsolidateRanges,
                            DwaPass::TrimWeights,
                        ]
                    }
                    None => vec![
                        DwaPass::ColPackMinimize,
                        DwaPass::ConsolidateRanges,
                        DwaPass::TrimWeights,
                    ],
                };

                if skip_minimize_before_suffix {
                    dwa_passes = vec![DwaPass::TrimWeights];
                }

                DeterminizeAndMinimizeConfig {
                    nwa_passes,
                    // NOTE: SatMinimize is intentionally disabled by default due to
                    // flakiness/performance (SAT UNSAT timeouts on some graphs).
                    dwa_passes,
                    use_rustfst_determinize: false,
                }
            },
            DeterminizeAndMinimizeProfile::Template => DeterminizeAndMinimizeConfig {
                // Template DWAs are built from terminal characterization NWAs in template_dfa.rs.
                // Each terminal has a characterization NWA that encodes how it interacts with
                // the parse stack (shifts, reduces, reduction cascades). These are determinized
                // into DWAs and then instantiated in the Parser NWA during precompute4.
                // Minimize is worthwhile since templates are reused many times.
                nwa_passes: vec![],  // NWA already processed before determinization
                dwa_passes: vec![DwaPass::FastMinimize],
                use_rustfst_determinize: false,
            },
            DeterminizeAndMinimizeProfile::Super => DeterminizeAndMinimizeConfig {
                // Super is the "universal" DWA that gets specialized into many DWAs.
                // Full minimization here pays off because the smaller Super means
                // smaller specialized DWAs and smaller combined NWA.
                nwa_passes: vec![NwaPass::CompressTransitions, NwaPass::Minimize],
                // NOTE: SatMinimize is intentionally disabled by default due to
                // flakiness/performance (SAT UNSAT timeouts on some graphs).
                dwa_passes: vec![DwaPass::PruneDeadEnds, DwaPass::ColPackMinimize],
                use_rustfst_determinize: false,
            },
            DeterminizeAndMinimizeProfile::SpecializedSuper => {
                // Specialized DWAs derived from Super by weight mapping.
                // Det/min is valid after vocab-space instantiation.
                let config_choice = std::env::var("SPECSUPER_CONFIG")
                    .unwrap_or_else(|_| "baseline".to_string())
                    .to_lowercase();
                let (nwa_passes, dwa_passes) = match config_choice.as_str() {
                    "baseline" => (
                        vec![NwaPass::Minimize],
                        vec![
                            DwaPass::PruneUnreachable,
                            DwaPass::PruneDeadEnds,
                            DwaPass::FastMinimize,
                            DwaPass::ColPackMinimize,
                        ],
                    ),
                    "colpack-only" => (
                        vec![NwaPass::Minimize],
                        vec![DwaPass::ColPackMinimize],
                    ),
                    "fast-only" => (
                        vec![NwaPass::Minimize],
                        vec![DwaPass::FastMinimize],
                    ),
                    "no-dwa" | "nwa-only" => (vec![NwaPass::Minimize], vec![]),
                    "no-min" => (vec![], vec![]),
                    other => {
                        eprintln!(
                            "WARN: Unknown SPECSUPER_CONFIG='{}', using baseline",
                            other
                        );
                        (
                            vec![NwaPass::Minimize],
                            vec![
                                DwaPass::PruneUnreachable,
                                DwaPass::PruneDeadEnds,
                                DwaPass::FastMinimize,
                                DwaPass::ColPackMinimize,
                            ],
                        )
                    }
                };
                DeterminizeAndMinimizeConfig {
                    nwa_passes,
                    dwa_passes,
                    use_rustfst_determinize: false,
                }
            },
            DeterminizeAndMinimizeProfile::Parser => {
                // Full pipeline for Parser DWA (finalize_and_optimize_and_determinize)
                // Includes minimize to reduce state count
                // NOTE: NWA MinimizeRustfst can be memory-intensive for large NWAs (2M+ states).
                // RustFST minimization relies on division, which is invalid for set-based weights,
                // so it is disabled by default. Enable explicitly via ENABLE_RUSTFST_MIN=1.
                let enable_rustfst_min = std::env::var("ENABLE_RUSTFST_MIN")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                let mut nwa_passes = vec![
                    NwaPass::PruneDeadEnds,
                    NwaPass::PruneUnreachable,
                    NwaPass::RmEpsilon,
                ];
                if enable_rustfst_min {
                    nwa_passes.push(NwaPass::MinimizeRustfst);
                }
                nwa_passes.extend([
                    NwaPass::PushFinalWeights,
                    NwaPass::PushWeightsToInitial,
                ]);
                if enable_rustfst_min {
                    nwa_passes.push(NwaPass::MinimizeRustfst);
                }
                nwa_passes.push(NwaPass::Minimize);
                DeterminizeAndMinimizeConfig {
                    nwa_passes,
                    dwa_passes: vec![DwaPass::PruneDeadEnds, DwaPass::FastMinimize, DwaPass::ConsolidateRanges, DwaPass::TrimWeights],
                    use_rustfst_determinize: false,
                }
            },
        }
    }
}

impl NWA {
    #[time_it("NWA::determinize_and_minimize")]
    pub fn determinize_and_minimize(mut self, profile: DeterminizeAndMinimizeProfile) -> DWA {
        self.determinize_and_minimize_with_hook(profile, Option::<fn(&mut DWA)>::None)
    }

    pub fn determinize_and_minimize_with_hook<F>(
        mut self,
        profile: DeterminizeAndMinimizeProfile,
        pre_dwa_hook: Option<F>,
    ) -> DWA
    where
        F: FnOnce(&mut DWA),
    {
        let _dwa_type_guard = crate::dwa_i32::minimization::graph_coloring::set_current_dwa_type(
            Some(profile.as_dwa_type()),
        );
        if self.states.len() > 1000 && optimize_debug() {
            return Self::run_determinize_and_minimize_experiment(self, profile);
        }
        let config = DeterminizeAndMinimizeConfig::for_profile(profile);
        if matches!(profile, DeterminizeAndMinimizeProfile::Parser) {
            crate::dwa_i32::determinization::with_determinize_progress_enabled(true, || {
                self.determinize_and_minimize_with_config_and_hook(config, pre_dwa_hook)
            })
        } else {
            self.determinize_and_minimize_with_config_and_hook(config, pre_dwa_hook)
        }
    }

    #[time_it("NWA::determinize_and_minimize_with_config")]
    pub fn determinize_and_minimize_with_config(&mut self, config: DeterminizeAndMinimizeConfig) -> DWA {
        self.determinize_and_minimize_with_config_and_hook(config, Option::<fn(&mut DWA)>::None)
    }

    fn determinize_and_minimize_with_config_and_hook<F>(
        &mut self,
        config: DeterminizeAndMinimizeConfig,
        pre_dwa_hook: Option<F>,
    ) -> DWA
    where
        F: FnOnce(&mut DWA),
    {
        let total_start = std::time::Instant::now();
        let debug_path = crate::debug_path_weight::parse_debug_path_weight_env();
        let debug_ignore_final = crate::debug_path_weight::debug_path_weight_ignore_final();
        let debug_num_tsids = crate::datastructures::get_num_tsids();
        let mut log_path_weight = |stage: &str, weight: &crate::dwa_i32::Weight, token_id: usize, labels: &[crate::dwa_i32::Label]| {
            let contains = crate::debug_path_weight::weight_contains_token(weight, token_id, debug_num_tsids);
            eprintln!(
                "DEBUG_PATH_WEIGHT stage={} token={} labels={:?} contains={} weight_len={} ignore_final={}",
                stage,
                token_id,
                labels,
                contains,
                weight.len(),
                debug_ignore_final,
            );
        };
        crate::debug!(5, "Determinize and minimize initial stats: {}",
            self.stats());

        if let Some((token_id, labels)) = debug_path.as_ref() {
            let weight = if debug_ignore_final {
                crate::debug_path_weight::check_nwa_path_weight_no_final(self, labels)
            } else {
                crate::debug_path_weight::check_nwa_path_weight(self, labels)
            };
            log_path_weight("nwa_initial", &weight, *token_id, labels);
        }

        // Run NWA passes
        let nwa_passes_start = std::time::Instant::now();
        for pass in config.nwa_passes {
            if !pass.is_enabled() {
                continue;
            }
            let pass_name = format!("{:?}", pass);
            let pass_start = std::time::Instant::now();
            timeit!(format!("NWA pass {:?}", pass), {
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
            });
            eprintln!("TIMING: NWA pass {} {:?}", pass_name, pass_start.elapsed());
            let pass_stats = self.stats();
            eprintln!(
                "TIMING: NWA pass {} states={} transitions={}",
                pass_name,
                pass_stats.states,
                pass_stats.transitions
            );
        }
        eprintln!("TIMING: NWA passes total {:?}", nwa_passes_start.elapsed());
        crate::debug!(5, "NWA minimization: {}", 
            self.stats());

        if let Some((token_id, labels)) = debug_path.as_ref() {
            let weight = if debug_ignore_final {
                crate::debug_path_weight::check_nwa_path_weight_no_final(self, labels)
            } else {
                crate::debug_path_weight::check_nwa_path_weight(self, labels)
            };
            log_path_weight("nwa_post_passes", &weight, *token_id, labels);
        }

        crate::datastructures::hybrid_bitset::reset_profiling();
        crate::datastructures::rangemap_weight::reset_profiling();
        crate::datastructures::abstract_weight::reset_weight_op_profiling();

        let det_total_start = std::time::Instant::now();
        let mut dwa = timeit!("NWA::determinize", {
            let det_start = std::time::Instant::now();
            let mut dwa = if config.use_rustfst_determinize {
                self.determinize_to_dwa_with_rustfst()
            } else {
                self.determinize()
            };
            let det_time = det_start.elapsed();
            crate::debug!(5, "Determinization: {} in {:.2?}", 
                dwa.stats(), det_time);
            dwa
        });
        eprintln!("TIMING: NWA::determinize {:?}", det_total_start.elapsed());

        if let Some((token_id, labels)) = debug_path.as_ref() {
            let weight = if debug_ignore_final {
                crate::debug_path_weight::check_dwa_path_weight_no_final(&dwa, labels)
            } else {
                crate::debug_path_weight::check_dwa_path_weight(&dwa, labels)
            };
            log_path_weight("dwa_post_determinize", &weight, *token_id, labels);
        }
        let pre_min_stats = dwa.stats();
        eprintln!(
            "TIMING: DWA pre_minimize states={} transitions={}",
            pre_min_stats.states,
            pre_min_stats.transitions,
        );

        if let Some(hook) = pre_dwa_hook {
            hook(&mut dwa);
        }

        if let Some((token_id, labels)) = debug_path.as_ref() {
            let weight = if debug_ignore_final {
                crate::debug_path_weight::check_dwa_path_weight_no_final(&dwa, labels)
            } else {
                crate::debug_path_weight::check_dwa_path_weight(&dwa, labels)
            };
            log_path_weight("dwa_post_hook", &weight, *token_id, labels);
        }

        // Run DWA passes
        let dwa_passes_start = std::time::Instant::now();
        for pass in config.dwa_passes.clone() {
            // Check if pass is enabled (e.g., ConsolidateRanges is disabled in weight-heavy mode)
            if !pass.is_enabled() {
                continue;
            }
            let pass_name = format!("{:?}", pass);
            let pass_start = std::time::Instant::now();
            timeit!(format!("DWA pass {:?}", pass), {
                match pass {
                    DwaPass::PruneUnreachable => { dwa.prune_unreachable(); },
                    DwaPass::PruneDeadEnds => { dwa.prune_dead_ends(); },
                    DwaPass::PushWeights => { dwa.push_weights_into_transitions_and_finals(); },
                    DwaPass::PushWeightsToInitial => { dwa.push_weights_to_initial(); },
                    DwaPass::ResidualPush => { dwa.residuated_push(); },
                    DwaPass::SatMinimize => { dwa.minimize_states_sat(); },
                    DwaPass::CadicalMinimize => { dwa.minimize_states_cadical(); },
                    DwaPass::DsaturMinimize => { dwa.minimize_states_dsatur(); },
                    DwaPass::ColPackMinimize => { dwa.minimize_states_colpack(); },
                    DwaPass::ColPackVerifiedMinimize => { dwa.minimize_states_colpack_verified(); },
                    DwaPass::FastMinimize => { dwa.minimize_states_fast(); },
                    DwaPass::RustfstMinimize => { dwa.minimize_with_rustfst_full(); },
                    DwaPass::ConsolidateRanges => { dwa.consolidate_ranges(); },
                    DwaPass::TrimWeights => { dwa.trim_weights(); },
                }
            });
            eprintln!("TIMING: DWA pass {} {:?}", pass_name, pass_start.elapsed());
            let pass_stats = dwa.stats();
            eprintln!(
                "TIMING: DWA pass {} states={} transitions={}",
                pass_name,
                pass_stats.states,
                pass_stats.transitions
            );
        }
        eprintln!("TIMING: DWA passes total {:?}", dwa_passes_start.elapsed());
        crate::debug!(5, "DWA minimization: {}",
            dwa.stats());

        if let Some((token_id, labels)) = debug_path.as_ref() {
            let weight = if debug_ignore_final {
                crate::debug_path_weight::check_dwa_path_weight_no_final(&dwa, labels)
            } else {
                crate::debug_path_weight::check_dwa_path_weight(&dwa, labels)
            };
            log_path_weight("dwa_post_minimize", &weight, *token_id, labels);
        }
        let post_min_stats = dwa.stats();
        eprintln!(
            "TIMING: DWA post_minimize states={} transitions={}",
            post_min_stats.states,
            post_min_stats.transitions
        );
        eprintln!("TIMING: NWA::determinize_and_minimize_with_config total {:?}", total_start.elapsed());
        dwa
    }

    pub fn run_determinize_and_minimize_experiment(self, profile: DeterminizeAndMinimizeProfile) -> DWA {
        let initial_stats = self.stats();
        println!("[Det&Min Experiment] [{:?}] Starting experiment with {}.", profile, initial_stats);

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
            vec![DwaPass::PruneDeadEnds, DwaPass::SatMinimize, DwaPass::PushWeights, DwaPass::PushWeightsToInitial, DwaPass::PruneUnreachable], // Standard
            vec![DwaPass::SatMinimize],
            vec![DwaPass::PruneDeadEnds, DwaPass::SatMinimize],
            vec![DwaPass::PushWeights, DwaPass::SatMinimize],
            vec![DwaPass::SatMinimize, DwaPass::PushWeights],
            vec![DwaPass::PruneUnreachable, DwaPass::PruneDeadEnds, DwaPass::SatMinimize],
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
                    use_rustfst_determinize: false,
                };

                let dwa = Self::determinize_and_minimize_with_config(&mut nwa_clone, config);

                let elapsed = start_time.elapsed();
                let final_stats = dwa.stats();
                let final_states = final_stats.states;

                println!("[Det&Min Experiment] [{:?}] Config N#{}-D#{}: NWA={:?} | DWA={:?} -> Time: {:.2?}, Stats: {}",
                         profile, n_idx, d_idx, nwa_pass_seq, dwa_pass_seq, elapsed, final_stats);

                if best_result.as_ref().map_or(true, |(_, best_time, best_states)| {
                    // Prefer fewer states, then faster time
                    final_states < *best_states || (final_states == *best_states && elapsed < *best_time)
                }) {
                    best_result = Some((dwa, elapsed, final_states));
                    best_config_idx = (n_idx, d_idx);
                }
            }
        }

        println!("[Det&Min Experiment] [{:?}] Winner: Config N#{}-D#{}", profile, best_config_idx.0, best_config_idx.1);
        best_result.unwrap().0
    }
}