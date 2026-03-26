//! Combined state and vocab equivalence analysis.
//!
//! State representatives are computed first, then vocab equivalence runs only
//! on the surviving representative set.

use std::collections::{BTreeMap, BTreeSet};

use hashbrown::HashMap;

use super::compat::TokenizerView;
use crate::ds::bitset::BitSet;

use super::state::fast::{self as state_equivalence_analysis, StateEquivalenceResult};
use super::vocab::fast::{self as vocab_equivalence_analysis, VocabEquivalenceResult};
use super::vocab::slow::partition_is_at_least_as_fine;

const REFERENCE_EQUIV_VERIFICATION_ENV: &str = "REFERENCE_EQUIV_VERIFICATION";
const SKIP_MAX_LENGTH_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_MAX_LENGTH_STATE_EQUIV";
const SKIP_TOKEN_STATE_EQUIV_ENV: &str = "GLRMASK_SKIP_TOKEN_STATE_EQUIV";
const USE_SLOW_VOCAB_EQUIV_ENV: &str = "GLRMASK_USE_SLOW_VOCAB_EQUIV";

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

fn compile_profile_enabled() -> bool {
    env_flag_enabled("GLRMASK_PROFILE_COMPILE") || env_flag_enabled("GLRMASK_PROFILE_COMPILE_SUMMARY")
}

fn elapsed_ms(started_at: std::time::Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

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
        "Fast state equivalence merged states that reference kept separate!\n\
         Fast classes: {}\n\
         Reference classes: {}",
        fast_state_classes.len(),
        reference_state_classes.len(),
    );
}

fn verify_vocab_partition_reference(
    fast_vocab_classes: &VocabEquivalenceResult,
    reference_vocab_classes: &VocabEquivalenceResult,
) {
    assert!(
        partition_is_at_least_as_fine(fast_vocab_classes, reference_vocab_classes),
        "Fast vocab equivalence merged tokens that reference kept separate!\n\
         Fast classes: {}\n\
         Reference classes: {}",
        fast_vocab_classes.len(),
        reference_vocab_classes.len(),
    );
}

fn collect_representative_states(states: &[usize]) -> Vec<usize> {
    states.iter().copied().collect::<BTreeSet<_>>().into_iter().collect()
}

/// Compute byte equivalence classes from the tokenizer DFA.
///
/// Bytes with identical transitions across all DFA states are merged into
/// the same class. This is used to deduplicate tokens before equivalence
/// analysis: tokens whose byte-class sequences are identical will always
/// produce the same DFA behavior from any starting state.
/// Token deduplication result.
struct TokenDedup<'a> {
    /// Byte slices for representative tokens (references into the original array).
    representative_token_bytes: Vec<&'a [u8]>,
    /// For each original token index, the index of its representative.
    original_to_repr: Vec<usize>,
    /// For each representative index, the list of original token indices it represents.
    repr_to_originals: Vec<Vec<usize>>,
}

/// Hash a token's byte-class sequence into a u128 for dedup.
/// Collision probability is ~n²/2^128 ≈ 0 for any practical n.
#[inline]
fn hash_byte_class_seq(bytes: &[u8], byte_to_class: &[u8; 256]) -> u128 {
    // Length-prefixed hash with a good mixing function.
    let mut h: u128 = 0xFF51_AFD7_ED55_8CCD;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53).wrapping_add(bytes.len() as u128);
    for &b in bytes {
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(byte_to_class[b as usize] as u128);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^= h >> 29;
    h
}

/// Deduplicate tokens by their byte-class sequence.
///
/// Tokens whose bytes map to the same sequence of byte classes under the
/// tokenizer DFA will always produce identical DFA trajectories from any
/// starting state. We only need to analyze one representative per group.
fn deduplicate_tokens_by_byte_class<'a, S: AsRef<[u8]>>(
    tokens: &'a [S],
    byte_to_class: &[u8; 256],
) -> TokenDedup<'a> {
    let mut hash_to_repr: HashMap<u128, usize> = HashMap::with_capacity(tokens.len() / 2);
    let mut representative_token_bytes: Vec<&'a [u8]> = Vec::new();
    let mut original_to_repr: Vec<usize> = Vec::with_capacity(tokens.len());
    let mut repr_to_originals: Vec<Vec<usize>> = Vec::new();

    for (orig_idx, token) in tokens.iter().enumerate() {
        let bytes = token.as_ref();
        let h = hash_byte_class_seq(bytes, byte_to_class);
        let repr_idx = *hash_to_repr.entry(h).or_insert_with(|| {
            let idx = representative_token_bytes.len();
            representative_token_bytes.push(bytes);
            repr_to_originals.push(Vec::new());
            idx
        });
        original_to_repr.push(repr_idx);
        repr_to_originals[repr_idx].push(orig_idx);
    }

    TokenDedup {
        representative_token_bytes,
        original_to_repr,
        repr_to_originals,
    }
}

/// Expand vocab equivalence classes from representative indices back to
/// original token indices.
fn expand_vocab_classes(
    dedup_classes: VocabEquivalenceResult,
    repr_to_originals: &[Vec<usize>],
) -> VocabEquivalenceResult {
    dedup_classes
        .into_iter()
        .map(|dedup_class| {
            let mut original_class: Vec<usize> = Vec::new();
            for dedup_idx in dedup_class {
                original_class.extend(repr_to_originals[dedup_idx].iter().copied());
            }
            original_class.sort_unstable();
            original_class
        })
        .collect()
}

fn representative_tokens_for_vocab_classes<'a>(
    dedup_vocab_classes: &VocabEquivalenceResult,
    representative_token_bytes: &'a [&'a [u8]],
) -> Vec<&'a [u8]> {
    dedup_vocab_classes
        .iter()
        .map(|dedup_class| representative_token_bytes[dedup_class[0]])
        .collect()
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
    ignore_terminal: Option<u32>,
) -> CombinedEquivalenceResult {
    let skip_max_length = env_flag_enabled(SKIP_MAX_LENGTH_STATE_EQUIV_ENV);
    let skip_token_state = env_flag_enabled(SKIP_TOKEN_STATE_EQUIV_ENV);
    let profile_compile = compile_profile_enabled();
    let combined_started_at = std::time::Instant::now();

    // Deduplicate tokens by byte-class sequence. Tokens whose bytes map
    // to the same DFA byte-class sequence behave identically from every
    // starting state, so we only need to analyze one representative.
    let dedup_started_at = std::time::Instant::now();
    let byte_to_class = super::compat::compute_byte_classes(tokenizer.dfa());
    let dedup = deduplicate_tokens_by_byte_class(tokens, &byte_to_class);
    let dedup_ms = elapsed_ms(dedup_started_at);

    let max_length_started_at = std::time::Instant::now();
    let pre_state_reps = if skip_max_length {
        initial_states.to_vec()
    } else {
        super::state::max_length::find_state_equivalence_classes(
            tokenizer,
            &dedup.representative_token_bytes,
            initial_states,
        )
    };
    let max_length_ms = elapsed_ms(max_length_started_at);

    let pre_reduced_states = collect_representative_states(&pre_state_reps);

    let use_slow_vocab = env_flag_enabled(USE_SLOW_VOCAB_EQUIV_ENV);
    let vocab_started_at = std::time::Instant::now();
    let dedup_vocab_classes = if use_slow_vocab {
        super::vocab::slow::find_vocab_equivalence_classes_with_follow(
            tokenizer,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            disallowed_follows,
        )
    } else {
        vocab_equivalence_analysis::find_vocab_equivalence_classes_with_follow_and_byte_classes(
            tokenizer,
            &dedup.representative_token_bytes,
            &pre_reduced_states,
            disallowed_follows,
            Some(&byte_to_class),
        )
    };
    let vocab_ms = elapsed_ms(vocab_started_at);

    // Running vocab first shrinks the token set before token_state refinement.
    // Tokens in the same vocab class are behaviorally identical across the
    // surviving states, so one representative token per class is sufficient
    // for the state refinement pass.
    let token_state_started_at = std::time::Instant::now();
    let representative_states = if skip_token_state {
        pre_state_reps.clone()
    } else {
        let vocab_representative_tokens = representative_tokens_for_vocab_classes(
            &dedup_vocab_classes,
            &dedup.representative_token_bytes,
        );
        let reduced_state_reps = state_equivalence_analysis::find_state_equivalence_classes(
            tokenizer,
            &vocab_representative_tokens,
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
    let token_state_ms = elapsed_ms(token_state_started_at);

    // Expand dedup vocab classes back to original token indices.
    let vocab_classes = expand_vocab_classes(dedup_vocab_classes, &dedup.repr_to_originals);

    let reduced_states = collect_representative_states(&representative_states);
    let state_classes = state_equivalence_analysis::mapping_to_equivalence_classes(
        initial_states,
        &representative_states,
    );

    if env_flag_enabled(REFERENCE_EQUIV_VERIFICATION_ENV) {
        let reference = super::reference::find_equivalence_classes(
            tokenizer,
            tokens,
            initial_states,
            disallowed_follows,
            ignore_terminal.map(|terminal| terminal as usize),
        );
        verify_state_partition_reference(&state_classes, &reference.state_classes);
        verify_vocab_partition_reference(&vocab_classes, &reference.vocab_classes);
    }

    if profile_compile {
        eprintln!(
            "[glrmask/profile][equiv] dedup_ms={:.3} tokens={}->{} max_length_ms={:.3} pre_states={} pre_reduced_states={} token_state_ms={:.3} reduced_states={} vocab_ms={:.3} state_classes={} vocab_classes={} total_ms={:.3}",
            dedup_ms,
            tokens.len(),
            dedup.representative_token_bytes.len(),
            max_length_ms,
            initial_states.len(),
            pre_reduced_states.len(),
            token_state_ms,
            reduced_states.len(),
            vocab_ms,
            state_classes.len(),
            vocab_classes.len(),
            elapsed_ms(combined_started_at),
        );
    }

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
