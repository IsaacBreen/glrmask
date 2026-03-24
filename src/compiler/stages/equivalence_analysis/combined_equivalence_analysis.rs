//! Combined state and vocab equivalence analysis.
//!
//! State representatives are computed first, then vocab equivalence runs only
//! on the surviving representative set.

use std::collections::{BTreeMap, BTreeSet};

use super::compat::TokenizerView;
use crate::ds::bitset::BitSet;

use super::state::fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab::fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};
#[cfg(test)]
use super::vocab::slow::partition_is_at_least_as_fine;

const SKIP_MAX_LENGTH_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_MAX_LENGTH_STATE_EQUIV";
const SKIP_TOKEN_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_TOKEN_STATE_EQUIV";

/// Result of combined equivalence analysis.
pub struct CombinedEquivalenceResult {
    /// Vocab equivalence classes: sets of token indices that behave identically.
    pub vocab_classes: VocabEquivalenceResult,

    /// State equivalence classes: sets of state IDs that behave identically.
    pub state_classes: StateEquivalenceResult,
}

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

#[cfg(test)]
fn verify_state_partition_reference(
    fast_state_classes: &StateEquivalenceResult,
    reference_state_classes: &StateEquivalenceResult,
) {
    let fast_state_classes: BTreeSet<Vec<_>> = fast_state_classes
        .iter()
        .map(|class| class.iter().copied().collect())
        .collect();
    let reference_state_classes: BTreeSet<Vec<_>> = reference_state_classes
        .iter()
        .map(|class| class.iter().copied().collect())
        .collect();
    assert!(
        partition_is_at_least_as_fine(&fast_state_classes, &reference_state_classes),
        "Fast state equivalence merged tokens that reference kept separate!\n\
         Fast classes: {}\n\
         Reference classes: {}",
        fast_state_classes.len(),
        reference_state_classes.len(),
    );
}

fn collect_representative_states(states: &[usize]) -> Vec<usize> {
    states.iter().copied().collect::<BTreeSet<_>>().into_iter().collect()
}

/// Compute combined state and vocab equivalence analysis.
///
/// This function:
/// 1. Computes state equivalence classes to find representative states
/// 2. Runs vocab equivalence analysis only on representative states
///
/// # Arguments
/// * `tokenizer` - The tokenizer DFA
/// * `tokens` - Vocabulary tokens to analyze
/// * `initial_states` - Initial tokenizer state IDs to consider
///
/// # Returns
/// Combined result containing vocab classes and state classes.
pub fn compute_combined_equivalence<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    tokens: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    _ignore_terminal: Option<u32>,
) -> CombinedEquivalenceResult {
    let skip_max_length = env_flag_enabled(SKIP_MAX_LENGTH_STATE_EQUIV_ENV);
    let skip_token_state = env_flag_enabled(SKIP_TOKEN_STATE_EQUIV_ENV);

    let pre_state_reps = if skip_max_length {
        initial_states.to_vec()
    } else {
        super::state::max_length::find_state_equivalence_classes(tokenizer, tokens, initial_states)
    };

    let pre_reduced_states = collect_representative_states(&pre_state_reps);
    let representative_states = if skip_token_state {
        pre_state_reps
    } else {
        let reduced_state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            tokenizer,
            tokens,
            &pre_reduced_states,
        );
        let rep_to_final: BTreeMap<usize, usize> = pre_reduced_states
            .iter()
            .copied()
            .zip(reduced_state_reps)
            .collect();
        pre_state_reps
            .iter()
            .map(|pre_rep| rep_to_final[pre_rep])
            .collect()
    };

    let reduced_states = collect_representative_states(&representative_states);
    let state_classes = state_equivalence_analysis::mapping_to_equivalence_classes(
        initial_states,
        &representative_states,
    );

    let vocab_classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_follow(
        tokenizer,
        tokens,
        &reduced_states,
        disallowed_follows,
    );

    CombinedEquivalenceResult {
        vocab_classes,
        state_classes,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::automata::lexer::ast::{bytes, star};
    use crate::compiler::compile::build_tokenizer_from_exprs;
    use crate::compiler::stages::equivalence_analysis::compat::TokenizerView;
    use crate::compiler::stages::equivalence_analysis::reference::find_equivalence_classes;
    use crate::compiler::stages::equivalence_analysis::state::fast as fast_state_equivalence;
    use crate::ds::bitset::BitSet;

    use super::verify_state_partition_reference;

    #[test]
    fn unrestricted_state_partition_refines_disallowed_follow_reference() {
        let exprs = [bytes(b"a"), star(bytes(b"b")), bytes(b"c")];
        let tokenizer = build_tokenizer_from_exprs(&exprs);
        let tokenizer_view = TokenizerView::new(&tokenizer);

        let tokens: Vec<Vec<u8>> = vec![
            b"c".to_vec(),
            b"ca".to_vec(),
            b"cba".to_vec(),
            b"bb".to_vec(),
        ];
        let states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

        let mut disallowed = BTreeMap::new();
        let mut bits = BitSet::new(3);
        bits.set(1);
        disallowed.insert(2u32, bits);

        let fast_mapping =
            fast_state_equivalence::find_state_equivalence_classes(&tokenizer_view, &tokens, &states);
        let fast_classes =
            fast_state_equivalence::mapping_to_equivalence_classes(&states, &fast_mapping);
        let reference = find_equivalence_classes(&tokenizer_view, &tokens, &states, &disallowed, None);

        verify_state_partition_reference(&fast_classes, &reference.state_classes);
    }
}
