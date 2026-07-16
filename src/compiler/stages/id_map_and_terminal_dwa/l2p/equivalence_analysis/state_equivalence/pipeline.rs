use crate::automata::lexer::Lexer;
use std::time::Instant;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;

use super::super::compat::TokenizerView;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::identity_state_map;
use super::max_length::{self, MaxLengthMode};
use super::restricted_observation;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateEquivalencePassKind {
    RestrictedObservation,
    MaxLength,
}

impl StateEquivalencePassKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "restricted_observation" => Ok(Self::RestrictedObservation),
            "max_length" => Ok(Self::MaxLength),
            other => Err(format!(
                "unknown state-equivalence pass `{other}`; expected one of: restricted_observation, max_length"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateEquivalenceScope {
    Global,
    L2p,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StateEquivalencePipelineConfig {
    pub passes: Vec<StateEquivalencePassKind>,
}

#[derive(Debug, Clone)]
pub(crate) struct StateEquivalencePassProfile {
    pub kind: StateEquivalencePassKind,
    pub name: &'static str,
    pub elapsed_ms: f64,
    pub representative_count: usize,
    pub skipped: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct StateEquivalencePipelineProfile {
    pub pass_profiles: Vec<StateEquivalencePassProfile>,
    pub restricted_observation_state_equiv_ms: f64,
    pub restricted_observation_reps: usize,
    pub max_length_skipped: bool,
    pub max_length_state_equiv_ms: f64,
    pub max_length_reps: usize,
    pub max_length_congruence_certified: bool,
}

fn parse_passes(value: &str) -> Vec<StateEquivalencePassKind> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .map(StateEquivalencePassKind::parse)
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|err| panic!("Invalid GLRMASK_STATE_EQUIV_PASSES value: {err}"))
}

fn resolve_pipeline_config(
    scoped_env: &str,
    default_passes: &[StateEquivalencePassKind],
) -> StateEquivalencePipelineConfig {
    let passes = std::env::var(scoped_env)
        .ok()
        .map(|value| parse_passes(&value))
        .or_else(|| {
            std::env::var("GLRMASK_STATE_EQUIV_PASSES")
                .ok()
                .map(|value| parse_passes(&value))
        })
        .unwrap_or_else(|| default_passes.to_vec());
    StateEquivalencePipelineConfig { passes }
}

pub(crate) fn resolve_global_pipeline_config(
    default_include_max_length: bool,
) -> StateEquivalencePipelineConfig {
    let default_passes = if default_include_max_length {
        &[StateEquivalencePassKind::MaxLength][..]
    } else {
        &[][..]
    };
    let mut config = resolve_pipeline_config("GLRMASK_GLOBAL_STATE_EQUIV_PASSES", default_passes);
    // Restricted observation depends on a local L2P vocabulary partition and
    // is deliberately unavailable to the global pipeline.
    config
        .passes
        .retain(|kind| !matches!(kind, StateEquivalencePassKind::RestrictedObservation));
    config
}

pub(crate) fn resolve_l2p_pipeline_config(
    default_include_max_length: bool,
) -> StateEquivalencePipelineConfig {
    let default_passes = if default_include_max_length {
        &[
            StateEquivalencePassKind::RestrictedObservation,
            StateEquivalencePassKind::MaxLength,
        ][..]
    } else {
        &[StateEquivalencePassKind::RestrictedObservation][..]
    };
    let mut config = resolve_pipeline_config("GLRMASK_L2P_STATE_EQUIV_PASSES", default_passes);
    // This is the coordinate-preserving replacement for L2P tokenizer
    // simplification, so it is mandatory and always precedes every optional
    // pass, including an environment-selected max-length pass.
    config
        .passes
        .retain(|kind| !matches!(kind, StateEquivalencePassKind::RestrictedObservation));
    config
        .passes
        .insert(0, StateEquivalencePassKind::RestrictedObservation);
    config
}

pub(crate) fn run_state_equivalence_pipeline(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    initial_state_map: Option<&ManyToOneIdMap>,
    active_groups: Option<&[bool]>,
    scope: StateEquivalenceScope,
    config: &StateEquivalencePipelineConfig,
    prebuilt_nfa_refinement: Option<&super::nfa::PrebuiltSparsePowersetRefinement<'_>>,
    kbounded_tokenizer_view: Option<&TokenizerView>,
    kbounded_byte_to_class: Option<&[u8; 256]>,
) -> (ManyToOneIdMap, StateEquivalencePipelineProfile) {
    let mut current_state_map = initial_state_map
        .cloned()
        .unwrap_or_else(|| identity_state_map(tokenizer.num_states() as usize));
    let mut profile = StateEquivalencePipelineProfile {
        max_length_skipped: !config
            .passes
            .iter()
            .any(|kind| matches!(kind, StateEquivalencePassKind::MaxLength)),
        restricted_observation_reps: current_state_map.num_internal_ids() as usize,
        max_length_reps: current_state_map.num_internal_ids() as usize,
        ..StateEquivalencePipelineProfile::default()
    };
    let statistic = max_length::compute_statistic(vocab);

    for kind in &config.passes {
        if matches!(kind, StateEquivalencePassKind::MaxLength)
            && !tokenizer.has_epsilon_transitions()
            && matches!(scope, StateEquivalenceScope::L2p)
            && kbounded_tokenizer_view.is_some_and(|view| {
                view.is_relevant_byte_congruent(&current_state_map, statistic.relevant_bytes())
            })
        {
            // A visible-output right congruence is already safe for every
            // vocabulary suffix, hence for the bounded max-length observer.
            // Retaining it is at least as fine as the optional prepass; the
            // following exact token refinement still decides the final map.
            profile.max_length_skipped = false;
            profile.max_length_state_equiv_ms = 0.0;
            profile.max_length_reps = current_state_map.num_internal_ids() as usize;
            profile.max_length_congruence_certified = true;
            profile.pass_profiles.push(StateEquivalencePassProfile {
                kind: StateEquivalencePassKind::MaxLength,
                name: "max_length_congruence_certified",
                elapsed_ms: 0.0,
                representative_count: current_state_map.num_internal_ids() as usize,
                skipped: false,
            });
            continue;
        }

        match *kind {
            StateEquivalencePassKind::RestrictedObservation => {
                assert!(
                    matches!(scope, StateEquivalenceScope::L2p),
                    "restricted-observation state equivalence is L2P-only",
                );
                let started_at = Instant::now();
                current_state_map = if tokenizer.has_epsilon_transitions() {
                    if let Some(prebuilt) = prebuilt_nfa_refinement {
                        prebuilt.compute_state_map(
                            tokenizer,
                            Some(&current_state_map),
                            super::nfa::RefinementDepth::Stable,
                        )
                    } else {
                        super::nfa::compute_state_map(
                            tokenizer,
                            statistic.relevant_bytes(),
                            active_groups,
                            Some(&current_state_map),
                            super::nfa::RefinementDepth::Stable,
                        )
                    }
                } else {
                    let tokenizer_view = kbounded_tokenizer_view
                        .expect("L2P restricted observation requires the shared analysis view");
                    restricted_observation::compute_state_map(
                        tokenizer_view,
                        statistic.relevant_bytes(),
                        Some(&current_state_map),
                        kbounded_byte_to_class,
                        config
                            .passes
                            .iter()
                            .any(|kind| matches!(kind, StateEquivalencePassKind::MaxLength)),
                    )
                };
                record_restricted_observation_profile(
                    &mut profile,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                    current_state_map.num_internal_ids() as usize,
                );
            }
            StateEquivalencePassKind::MaxLength => {
                let mode = match scope {
                    StateEquivalenceScope::Global => MaxLengthMode::StableByteRestricted,
                    StateEquivalenceScope::L2p => MaxLengthMode::KBoundedByteRestricted,
                };
                let started_at = Instant::now();
                current_state_map = if tokenizer.has_epsilon_transitions()
                    && matches!(scope, StateEquivalenceScope::L2p)
                    && let Some(prebuilt) = prebuilt_nfa_refinement
                {
                    prebuilt.compute_state_map(
                        tokenizer,
                        Some(&current_state_map),
                        super::nfa::RefinementDepth::Bounded(statistic.max_token_len()),
                    )
                } else {
                    max_length::compute_state_map(
                        tokenizer,
                        &statistic,
                        Some(&current_state_map),
                        active_groups,
                        mode,
                        kbounded_tokenizer_view,
                        kbounded_byte_to_class,
                    )
                };
                record_max_length_profile(
                    &mut profile,
                    *kind,
                    mode,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                    current_state_map.num_internal_ids() as usize,
                );
            }
        }
    }

    (current_state_map, profile)
}

fn record_restricted_observation_profile(
    profile: &mut StateEquivalencePipelineProfile,
    elapsed_ms: f64,
    representative_count: usize,
) {
    profile.restricted_observation_state_equiv_ms = elapsed_ms;
    profile.restricted_observation_reps = representative_count;
    profile.pass_profiles.push(StateEquivalencePassProfile {
        kind: StateEquivalencePassKind::RestrictedObservation,
        name: "restricted_observation",
        elapsed_ms,
        representative_count,
        skipped: false,
    });
}

fn record_max_length_profile(
    profile: &mut StateEquivalencePipelineProfile,
    kind: StateEquivalencePassKind,
    mode: MaxLengthMode,
    elapsed_ms: f64,
    representative_count: usize,
) {
    profile.max_length_skipped = false;
    profile.max_length_state_equiv_ms = elapsed_ms;
    profile.max_length_reps = representative_count;
    profile.pass_profiles.push(StateEquivalencePassProfile {
        kind,
        name: mode.name(),
        elapsed_ms,
        representative_count,
        skipped: false,
    });
}
