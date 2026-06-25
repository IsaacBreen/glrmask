use crate::automata::lexer::Lexer;
use std::time::Instant;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;

use super::super::compat::TokenizerView;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::identity_state_map;
use super::max_length::{self, MaxLengthMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateEquivalencePassKind {
    MaxLength,
}

impl StateEquivalencePassKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            "max_length" => Ok(Self::MaxLength),
            other => Err(format!(
                "unknown state-equivalence pass `{other}`; expected one of: max_length"
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
    pub max_length_skipped: bool,
    pub max_length_state_equiv_ms: f64,
    pub max_length_reps: usize,
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
    resolve_pipeline_config("GLRMASK_GLOBAL_STATE_EQUIV_PASSES", default_passes)
}

pub(crate) fn resolve_l2p_pipeline_config(
    default_include_max_length: bool,
) -> StateEquivalencePipelineConfig {
    let default_passes = if default_include_max_length {
        &[StateEquivalencePassKind::MaxLength][..]
    } else {
        &[][..]
    };
    resolve_pipeline_config("GLRMASK_L2P_STATE_EQUIV_PASSES", default_passes)
}

pub(crate) fn run_state_equivalence_pipeline(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    initial_state_map: Option<&ManyToOneIdMap>,
    active_groups: Option<&[bool]>,
    scope: StateEquivalenceScope,
    config: &StateEquivalencePipelineConfig,
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
        max_length_reps: current_state_map.num_internal_ids() as usize,
        ..StateEquivalencePipelineProfile::default()
    };

    for kind in &config.passes {
        match *kind {
            StateEquivalencePassKind::MaxLength => {
                let mode = match scope {
                    StateEquivalenceScope::Global => MaxLengthMode::StableByteRestricted,
                    StateEquivalenceScope::L2p => MaxLengthMode::KBoundedByteRestricted,
                };
                let statistic = max_length::compute_statistic(vocab);
                let started_at = Instant::now();
                current_state_map = max_length::compute_state_map(
                    tokenizer,
                    &statistic,
                    Some(&current_state_map),
                    active_groups,
                    mode,
                    kbounded_tokenizer_view,
                    kbounded_byte_to_class,
                );
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
