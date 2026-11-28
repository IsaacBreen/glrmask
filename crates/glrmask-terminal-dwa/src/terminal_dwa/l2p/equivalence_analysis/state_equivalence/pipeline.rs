use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::ManyToOneIdMap;
use crate::Vocab;
use std::time::Instant;

use super::super::compat::FlatDfa;
use super::{identity_state_map, nfa, restricted_observation};

/// The local quotient either preserves the incoming coordinate, observes
/// destinations only, or also observes the source. The latter establishes
/// a right congruence and therefore subsumes every bounded byte observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalObservation {
    Skip,
    Destinations,
    SourceAndDestinations,
}

#[derive(Debug, Clone, Default)]
pub struct StateEquivalencePipelineProfile {
    pub restricted_observation_state_equiv_ms: f64,
    pub restricted_observation_reps: usize,
    pub max_length_skipped: bool,
    pub max_length_state_equiv_ms: f64,
    pub max_length_reps: usize,
    pub relevant_byte_congruence_certified: bool,
}

/// Keep the existing private diagnostic switches, but interpret the list as
/// a selection of observations, not an executable sequence of passes.
pub fn includes_source_observation(scoped_env: &str, default: bool) -> bool {
    let Ok(value) = std::env::var(scoped_env)
        .or_else(|_| std::env::var("GLRMASK_STATE_EQUIV_PASSES"))
    else { return default; };
    if value.trim().is_empty() { return false; }
    let mut include = false;
    for item in value.split(',').map(str::trim) {
        match item {
            "restricted_observation" => {}
            "max_length" => include = true,
            other => panic!("Invalid GLRMASK_STATE_EQUIV_PASSES value: unknown state-equivalence pass `{other}`"),
        }
    }
    include
}

impl LocalObservation {
    pub fn resolve(include_source: bool) -> Self {
        if includes_source_observation("GLRMASK_L2P_STATE_EQUIV_PASSES", include_source) {
            Self::SourceAndDestinations
        } else {
            Self::Destinations
        }
    }
}

pub fn local_observation_map(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    initial: Option<&ManyToOneIdMap>,
    active_groups: Option<&[bool]>,
    observation: LocalObservation,
    prebuilt: Option<&nfa::PrebuiltSparsePowersetRefinement<'_>>,
    view: Option<&FlatDfa>,
    byte_to_class: Option<&[u8; 256]>,
    certified: bool,
) -> (ManyToOneIdMap, StateEquivalencePipelineProfile) {
    assert!(!certified || initial.is_some());
    let initial = initial.cloned()
        .unwrap_or_else(|| identity_state_map(tokenizer.num_states() as usize));
    let initial_reps = initial.num_internal_ids() as usize;
    let includes_source = observation == LocalObservation::SourceAndDestinations;
    let mut profile = StateEquivalencePipelineProfile {
        restricted_observation_reps: initial_reps,
        max_length_reps: initial_reps,
        max_length_skipped: !includes_source,
        relevant_byte_congruence_certified: certified,
        ..Default::default()
    };
    if observation == LocalObservation::Skip || certified {
        return (initial, profile);
    }
    let started = Instant::now();
    let relevant_bytes = vocab.relevant_byte_mask();
    let epsilon = tokenizer.has_epsilon_transitions();
    let map = if epsilon {
        if let Some(prebuilt) = prebuilt {
            prebuilt.compute_state_map(tokenizer, Some(&initial), nfa::RefinementDepth::Stable)
        } else {
            nfa::compute_state_map(
                tokenizer, &relevant_bytes, active_groups,
                Some(&initial), nfa::RefinementDepth::Stable,
            )
        }
    } else {
        restricted_observation::compute_state_map(
            view.expect("L2P restricted observation requires the shared analysis view"),
            &relevant_bytes, Some(&initial), byte_to_class, includes_source,
        )
    };
    profile.restricted_observation_state_equiv_ms = started.elapsed().as_secs_f64() * 1000.0;
    profile.restricted_observation_reps = map.num_internal_ids() as usize;
    profile.relevant_byte_congruence_certified = epsilon || includes_source;
    if includes_source {
        profile.max_length_reps = profile.restricted_observation_reps;
    }
    (map, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

    #[test]
    fn deterministic_source_observation_is_a_right_congruence() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer()
            .try_full_determinization(1_024, 100_000)
            .expect("small test tokenizer determinizes").tokenizer;
        assert!(!tokenizer.has_epsilon_transitions());
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()), (1, b"b".to_vec()),
            (2, b"ab".to_vec()), (3, b"ba".to_vec()),
        ]);
        let view = FlatDfa::new(&tokenizer);
        let (map, profile) = local_observation_map(
            &tokenizer, &vocab, None, None, LocalObservation::SourceAndDestinations,
            None, Some(&view), None, false,
        );
        assert!(profile.relevant_byte_congruence_certified);
        assert!(view.is_relevant_byte_congruent(&map, &vocab.relevant_byte_mask()));
    }

    #[test]
    fn epsilon_observation_includes_source_even_when_not_requested() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()), (1, b"b".to_vec()),
            (2, b"ab".to_vec()), (3, b"ba".to_vec()),
        ]);
        let (map, profile) = local_observation_map(
            &tokenizer, &vocab, None, None, LocalObservation::Destinations,
            None, None, None, false,
        );
        let (other, _) = local_observation_map(
            &tokenizer, &vocab, None, None, LocalObservation::SourceAndDestinations,
            None, None, None, false,
        );
        assert!(profile.relevant_byte_congruence_certified);
        assert_eq!(map.original_to_internal, other.original_to_internal);
    }
}
