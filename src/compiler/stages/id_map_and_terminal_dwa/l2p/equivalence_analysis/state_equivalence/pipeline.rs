use std::time::Instant;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;

use super::identity_state_map;
use super::max_length::MaxLengthPass;
use super::pass::{StateEquivalencePass, StateEquivalencePassKind, StateEquivalenceScope};
use super::vocab_trie_hash128::VocabTrieHash128Pass;

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
        match (*kind, scope) {
            (StateEquivalencePassKind::MaxLength, StateEquivalenceScope::Global) => {
                let pass = MaxLengthPass::stable_byte_restricted();
                let statistic = pass.compute_statistic(vocab);
                let started_at = Instant::now();
                current_state_map = pass.compute_state_map(
                    tokenizer,
                    &statistic,
                    Some(&current_state_map),
                    active_groups,
                );
                let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                profile.max_length_skipped = false;
                profile.max_length_state_equiv_ms = elapsed_ms;
                profile.max_length_reps = current_state_map.num_internal_ids() as usize;
                profile.pass_profiles.push(StateEquivalencePassProfile {
                    kind: *kind,
                    name: pass.name(),
                    elapsed_ms,
                    representative_count: current_state_map.num_internal_ids() as usize,
                    skipped: false,
                });
            }
            (StateEquivalencePassKind::MaxLength, StateEquivalenceScope::L2p) => {
                let pass = MaxLengthPass::kbounded_byte_restricted();
                let statistic = pass.compute_statistic(vocab);
                let started_at = Instant::now();
                current_state_map = pass.compute_state_map(
                    tokenizer,
                    &statistic,
                    Some(&current_state_map),
                    active_groups,
                );
                let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                profile.max_length_skipped = false;
                profile.max_length_state_equiv_ms = elapsed_ms;
                profile.max_length_reps = current_state_map.num_internal_ids() as usize;
                profile.pass_profiles.push(StateEquivalencePassProfile {
                    kind: *kind,
                    name: pass.name(),
                    elapsed_ms,
                    representative_count: current_state_map.num_internal_ids() as usize,
                    skipped: false,
                });
            }
            (StateEquivalencePassKind::VocabTrieHash128, _) => {
                let pass = VocabTrieHash128Pass;
                let statistic = pass.compute_statistic(vocab);
                let started_at = Instant::now();
                current_state_map = pass.compute_state_map(
                    tokenizer,
                    &statistic,
                    Some(&current_state_map),
                    active_groups,
                );
                let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
                profile.pass_profiles.push(StateEquivalencePassProfile {
                    kind: *kind,
                    name: pass.name(),
                    elapsed_ms,
                    representative_count: current_state_map.num_internal_ids() as usize,
                    skipped: false,
                });
            }
        }
    }

    (current_state_map, profile)
}