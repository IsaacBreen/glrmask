use std::collections::BTreeSet;

use super::{BuildInput, Implementation, LocalIdMapTerminalDwa};
use crate::automata::lexer::Lexer;
use crate::ds::weight::Weight;

fn original_tokens(
    artifact: &LocalIdMapTerminalDwa,
    weight: &Weight,
    raw_state: u32,
) -> BTreeSet<u32> {
    let tsid = artifact.id_map.tokenizer_states.original_to_internal[raw_state as usize];
    let Some(tokens) = weight.token_set_for_tsid_ref(tsid) else {
        return BTreeSet::new();
    };
    let mut originals = BTreeSet::new();
    for internal in tokens.iter() {
        if let Some(singletons) = artifact.id_map.deferred_vocab_singleton_original_ids.as_ref() {
            originals.insert(singletons[internal as usize]);
        } else {
            originals.extend(
                artifact.id_map.vocab_tokens.internal_to_originals[internal as usize]
                    .iter()
                    .copied(),
            );
        }
    }
    originals
}

pub(super) fn assert_equivalent(
    input: BuildInput<'_>,
    actual_name: Implementation,
    actual: Option<&LocalIdMapTerminalDwa>,
    expected_name: Implementation,
    expected: Option<&LocalIdMapTerminalDwa>,
) {
    match (actual, expected) {
        (None, None) => return,
        (Some(_), None) | (None, Some(_)) => panic!(
            "L1 implementation mismatch in partition {}: {:?} returned {}, {:?} returned {}",
            input.partition_label,
            actual_name,
            if actual.is_some() { "a DWA" } else { "None" },
            expected_name,
            if expected.is_some() { "a DWA" } else { "None" },
        ),
        (Some(actual), Some(expected)) => {
            for terminal in 0..input.grammar.num_terminals {
                let actual_weight = actual.dwa.eval_word(&[terminal as i32]);
                let expected_weight = expected.dwa.eval_word(&[terminal as i32]);
                for state in 0..input.tokenizer.num_states() {
                    let actual_tokens = original_tokens(actual, &actual_weight, state);
                    let expected_tokens = original_tokens(expected, &expected_weight, state);
                    assert_eq!(
                        actual_tokens, expected_tokens,
                        "L1 mismatch partition={} terminal={} raw_state={} actual={:?} expected={:?}",
                        input.partition_label, terminal, state, actual_name, expected_name,
                    );
                }
            }
        }
    }
}
