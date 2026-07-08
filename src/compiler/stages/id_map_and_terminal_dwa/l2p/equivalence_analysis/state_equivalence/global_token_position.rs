//! Exact global scanner-state equivalence induced by token positions.
//!
//! The seed is built from three token-position observations:
//!
//! * every first-byte destination is retained exactly;
//! * every state reachable at byte position three or later is singleton; and
//! * all frozen finalizer/future-finalizer observations remain equal.
//!
//! The seed is then closed by stable refinement over every byte used by the
//! vocabulary partition. That final closure is necessary before the quotient
//! can be supplied to byte-level terminal interchangeability discovery.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{GlobalScannerStateQuotient, ManyToOneIdMap};
use crate::ds::bitset::BitSet;

use super::max_length::stable_refinement_from_initial_blocks;

#[derive(Debug, Clone, Default)]
pub(crate) struct GlobalTokenPositionEquivalenceProfile {
    pub(crate) first_byte_count: usize,
    pub(crate) remaining_byte_count: usize,
    pub(crate) second_byte_count: usize,
    pub(crate) second_state_count: usize,
    pub(crate) third_plus_state_count: usize,
    pub(crate) frozen_output_class_count: usize,
    pub(crate) first_destination_class_count: usize,
    pub(crate) position_seed_class_count: usize,
    pub(crate) seed_class_count: usize,
    pub(crate) exact_class_count: usize,
    pub(crate) seed_ms: f64,
    pub(crate) closure_ms: f64,
    pub(crate) total_ms: f64,
}

/// Total raw-state partition that is exact at nonempty vocabulary-token
/// boundaries. It is not a raw-byte DFA quotient and must not be supplied to
/// byte-level TI directly.
#[derive(Debug, Clone)]
pub(crate) struct GlobalTokenBoundaryStatePartition {
    map: ManyToOneIdMap,
}

impl GlobalTokenBoundaryStatePartition {
    #[inline]
    pub(crate) fn as_many_to_one(&self) -> &ManyToOneIdMap {
        &self.map
    }
}

/// The requested global token-position partition C.
///
/// A state is singleton once it can occur from the second byte onward. All
/// remaining states are grouped by their exact destinations on every possible
/// token-first byte. This is a token-boundary relation: it is intentionally
/// not a raw-byte DFA quotient, because a state class is only entered before
/// the first byte of a vocabulary token.
#[derive(Debug, Clone)]
pub(crate) struct GlobalTokenPositionStatePartition {
    map: ManyToOneIdMap,
}

impl GlobalTokenPositionStatePartition {
    #[inline]
    pub(crate) fn as_many_to_one(&self) -> &ManyToOneIdMap {
        &self.map
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct TokenPositionSeedKey {
    frozen_output: u32,
    third_plus_singleton: u32,
    first_destinations: Box<[u32]>,
}

#[inline]
fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn fingerprint_bitset(bits: &BitSet) -> u64 {
    bits.words().iter().fold(
        mix_u64(bits.len() as u64 ^ 0x9e37_79b9_7f4a_7c15),
        |hash, &word| mix_u64(hash ^ word.wrapping_add(0xa24b_aed4_963e_e407)),
    )
}

/// Intern exact frozen `(finalizers, future-finalizers)` labels without
/// cloning terminal bitsets. Fingerprints only select a collision bucket; the
/// final equality check is on the complete bitsets.
fn frozen_output_labels(tokenizer: &Tokenizer) -> (Vec<u32>, usize) {
    let state_count = tokenizer.num_states() as usize;
    let mut buckets = FxHashMap::<(u64, u64), Vec<(u32, u32)>>::default();
    let mut labels = vec![u32::MAX; state_count];
    let mut label_count = 0u32;

    for state in 0..state_count {
        let finalizers = tokenizer.matched_terminal_bitset(state as u32);
        let futures = tokenizer.possible_future_terminals(state as u32);
        let key = (fingerprint_bitset(finalizers), fingerprint_bitset(futures));
        let existing = buckets.get(&key).and_then(|candidates| {
            candidates.iter().find_map(|&(representative, label)| {
                (tokenizer.matched_terminal_bitset(representative) == finalizers
                    && tokenizer.possible_future_terminals(representative) == futures)
                    .then_some(label)
            })
        });
        let label = existing.unwrap_or_else(|| {
            let label = label_count;
            label_count += 1;
            buckets.entry(key).or_default().push((state as u32, label));
            label
        });
        labels[state] = label;
    }

    (labels, label_count as usize)
}

fn token_position_byte_sets(vocab: &Vocab) -> ([bool; 256], [bool; 256], [bool; 256]) {
    let mut first = [false; 256];
    let mut remaining = [false; 256];
    let mut second = [false; 256];
    for bytes in vocab.entries.values() {
        let Some((&first_byte, tail)) = bytes.split_first() else {
            continue;
        };
        first[first_byte as usize] = true;
        for &byte in tail {
            remaining[byte as usize] = true;
        }
        if let Some(&second_byte) = tail.first() {
            second[second_byte as usize] = true;
        }
    }
    (first, remaining, second)
}

fn selected_bytes(bytes: &[bool; 256]) -> Vec<u8> {
    (0..=u8::MAX)
        .filter(|&byte| bytes[byte as usize])
        .collect()
}

/// States that can occur immediately after a first token byte. A finalizer in
/// this set makes the lexer initial state relevant to the second-byte frontier,
/// exactly as specified by the token-position construction.
fn second_states(
    tokenizer: &Tokenizer,
    first_bytes: &[u8],
) -> Vec<bool> {
    let state_count = tokenizer.num_states() as usize;
    let mut states = vec![false; state_count];
    for source in 0..state_count {
        for &byte in first_bytes {
            if let Some(destination) = tokenizer.step(source as u32, byte) {
                states[destination as usize] = true;
            }
        }
    }
    if states.iter().enumerate().any(|(state, &included)| {
        included && !tokenizer.matched_terminal_bitset(state as u32).is_empty()
    }) {
        states[tokenizer.initial_state_id() as usize] = true;
    }
    states
}

/// States reachable at byte position three or later: one transition on a
/// possible second token byte from a second-position state, followed by any
/// number of ordinary lexer transitions.
fn third_plus_states(
    tokenizer: &Tokenizer,
    second_states: &[bool],
    second_bytes: &[u8],
) -> Vec<bool> {
    let state_count = tokenizer.num_states() as usize;
    let mut reached = vec![false; state_count];
    let mut worklist = VecDeque::new();

    for (state, &is_second_state) in second_states.iter().enumerate() {
        if !is_second_state {
            continue;
        }
        for &byte in second_bytes {
            if let Some(destination) = tokenizer.step(state as u32, byte) {
                let destination = destination as usize;
                if !reached[destination] {
                    reached[destination] = true;
                    worklist.push_back(destination as u32);
                }
            }
        }
    }

    while let Some(state) = worklist.pop_front() {
        for (_, destination) in tokenizer.transitions_from(state) {
            let destination = destination as usize;
            if !reached[destination] {
                reached[destination] = true;
                worklist.push_back(destination as u32);
            }
        }
    }

    reached
}

fn seed_partition(
    tokenizer: &Tokenizer,
    first_bytes: &[u8],
    frozen_outputs: &[u32],
    third_plus: &[bool],
) -> (Vec<u32>, usize) {
    let state_count = tokenizer.num_states() as usize;
    let mut key_to_class = FxHashMap::<TokenPositionSeedKey, u32>::default();
    let mut blocks = vec![u32::MAX; state_count];
    let mut class_count = 0u32;

    for state in 0..state_count {
        let first_destinations = first_bytes
            .iter()
            .map(|&byte| tokenizer.get_transition(state as u32, byte))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let key = TokenPositionSeedKey {
            frozen_output: frozen_outputs[state],
            third_plus_singleton: third_plus[state].then_some(state as u32).unwrap_or(u32::MAX),
            first_destinations,
        };
        let next = class_count;
        let class = *key_to_class.entry(key).or_insert_with(|| {
            class_count += 1;
            next
        });
        blocks[state] = class;
    }

    (blocks, class_count as usize)
}

fn token_position_partition(
    tokenizer: &Tokenizer,
    first_bytes: &[u8],
    third_plus: &[bool],
) -> ManyToOneIdMap {
    let state_count = tokenizer.num_states() as usize;
    let mut key_to_class = FxHashMap::<TokenPositionSeedKey, u32>::default();
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for state in 0..state_count {
        let key = TokenPositionSeedKey {
            frozen_output: 0,
            third_plus_singleton: third_plus[state].then_some(state as u32).unwrap_or(u32::MAX),
            first_destinations: first_bytes
                .iter()
                .map(|&byte| tokenizer.get_transition(state as u32, byte))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let next = internal_to_originals.len() as u32;
        let class = *key_to_class.entry(key).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(state as u32);
            next
        });
        original_to_internal[state] = class;
        internal_to_originals[class as usize].push(state as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn first_destination_partition_class_count(
    tokenizer: &Tokenizer,
    first_bytes: &[u8],
    third_plus: Option<&[bool]>,
    frozen_outputs: Option<&[u32]>,
) -> usize {
    let mut classes = FxHashMap::<TokenPositionSeedKey, u32>::default();
    for state in 0..tokenizer.num_states() as usize {
        let key = TokenPositionSeedKey {
            frozen_output: frozen_outputs.map_or(0, |labels| labels[state]),
            third_plus_singleton: third_plus
                .is_some_and(|states| states[state])
                .then_some(state as u32)
                .unwrap_or(u32::MAX),
            first_destinations: first_bytes
                .iter()
                .map(|&byte| tokenizer.get_transition(state as u32, byte))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };
        let next = classes.len() as u32;
        classes.entry(key).or_insert(next);
    }
    classes.len()
}

fn first_destination_partition(
    tokenizer: &Tokenizer,
    first_bytes: &[u8],
) -> ManyToOneIdMap {
    let state_count = tokenizer.num_states() as usize;
    let mut key_to_class = FxHashMap::<Box<[u32]>, u32>::default();
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for state in 0..state_count {
        let destinations = first_bytes
            .iter()
            .map(|&byte| tokenizer.get_transition(state as u32, byte))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let next = internal_to_originals.len() as u32;
        let class = *key_to_class.entry(destinations).or_insert_with(|| {
            internal_to_originals.push(Vec::new());
            representative_original_ids.push(state as u32);
            next
        });
        original_to_internal[state] = class;
        internal_to_originals[class as usize].push(state as u32);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

/// Build the exact token-boundary partition induced by the first bytes of all
/// nonempty tokens. If two states agree here, every token begins from the same
/// raw successor, and deterministic continuation makes the rest of that token
/// identical byte-for-byte.
pub(crate) fn compute_global_token_boundary_state_partition(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> Option<GlobalTokenBoundaryStatePartition> {
    if vocab.entries.values().any(Vec::is_empty) {
        return None;
    }
    let (first_set, _, _) = token_position_byte_sets(vocab);
    let first_bytes = selected_bytes(&first_set);
    (!first_bytes.is_empty()).then(|| GlobalTokenBoundaryStatePartition {
        map: first_destination_partition(tokenizer, &first_bytes),
    })
}

/// Build the exact global token-position partition C from A and the set of B
/// first-byte destination partitions. Length-one tokens contribute their first
/// byte normally and simply contribute no second byte.
pub(crate) fn compute_global_token_position_state_partition(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> Option<GlobalTokenPositionStatePartition> {
    if vocab.entries.values().any(Vec::is_empty) {
        return None;
    }
    let (first_set, _, second_set) = token_position_byte_sets(vocab);
    let first_bytes = selected_bytes(&first_set);
    if first_bytes.is_empty() {
        return None;
    }
    let second_bytes = selected_bytes(&second_set);
    let second_states = second_states(tokenizer, &first_bytes);
    let third_plus = third_plus_states(tokenizer, &second_states, &second_bytes);
    let (frozen_outputs, _) = frozen_output_labels(tokenizer);
    let (blocks, class_count) = seed_partition(tokenizer, &first_bytes, &frozen_outputs, &third_plus);
    Some(GlobalTokenPositionStatePartition {
        map: map_from_blocks(blocks, class_count),
    })
}

fn map_from_blocks(blocks: Vec<u32>, class_count: usize) -> ManyToOneIdMap {
    let mut internal_to_originals = vec![Vec::<u32>::new(); class_count];
    let mut representative_original_ids = vec![u32::MAX; class_count];
    for (state, &class) in blocks.iter().enumerate() {
        let class = class as usize;
        let originals = &mut internal_to_originals[class];
        if originals.is_empty() {
            representative_original_ids[class] = state as u32;
        }
        originals.push(state as u32);
    }
    ManyToOneIdMap {
        original_to_internal: blocks,
        internal_to_originals,
        representative_original_ids,
    }
}

/// Build an exact total scanner-state quotient before TI. `seed` is the
/// requested A∧B construction, strengthened by frozen-output equality; the
/// final stable closure makes it valid for byte-level TI as well as later
/// token-level equivalence.
pub(crate) fn compute_global_token_position_state_quotient(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> (GlobalScannerStateQuotient, GlobalTokenPositionEquivalenceProfile) {
    let started_at = Instant::now();
    let (first_set, remaining_set, second_set) = token_position_byte_sets(vocab);
    let first_bytes = selected_bytes(&first_set);
    let remaining_bytes = selected_bytes(&remaining_set);
    let second_bytes = selected_bytes(&second_set);

    let seed_started_at = Instant::now();
    let second_states = second_states(tokenizer, &first_bytes);
    let third_plus = third_plus_states(tokenizer, &second_states, &second_bytes);
    let (frozen_outputs, frozen_output_class_count) = frozen_output_labels(tokenizer);
    let first_destination_class_count =
        first_destination_partition_class_count(tokenizer, &first_bytes, None, None);
    let position_seed_class_count = first_destination_partition_class_count(
        tokenizer,
        &first_bytes,
        Some(&third_plus),
        None,
    );
    let (seed_blocks, seed_class_count) =
        seed_partition(tokenizer, &first_bytes, &frozen_outputs, &third_plus);
    let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

    // TI's characterizer can traverse every byte that occurs anywhere in the
    // partition vocabulary, not merely token first bytes. Completing the seed
    // to a stable right congruence is what makes representative-only TI exact.
    let closure_started_at = Instant::now();
    let active_bytes = selected_bytes(&{
        let mut all = first_set;
        for byte in 0..256 {
            all[byte] |= remaining_set[byte];
        }
        all
    });
    let exact_blocks = stable_refinement_from_initial_blocks(
        tokenizer,
        &active_bytes,
        &seed_blocks,
        seed_class_count,
    );
    let exact_class_count = exact_blocks
        .iter()
        .copied()
        .max()
        .map_or(0, |class| class as usize + 1);
    let closure_ms = closure_started_at.elapsed().as_secs_f64() * 1000.0;

    let state_count = tokenizer.num_states() as usize;
    let quotient = GlobalScannerStateQuotient::from_total_raw_state_map(
        map_from_blocks(exact_blocks, exact_class_count),
        state_count,
    );
    let profile = GlobalTokenPositionEquivalenceProfile {
        first_byte_count: first_bytes.len(),
        remaining_byte_count: remaining_bytes.len(),
        second_byte_count: second_bytes.len(),
        second_state_count: second_states.iter().filter(|&&present| present).count(),
        third_plus_state_count: third_plus.iter().filter(|&&present| present).count(),
        frozen_output_class_count,
        first_destination_class_count,
        position_seed_class_count,
        seed_class_count,
        exact_class_count,
        seed_ms,
        closure_ms,
        total_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    };
    (quotient, profile)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;

    fn tokenizer(expressions: Vec<Expr>) -> Tokenizer {
        let terminal_count = expressions.len() as u32;
        build_regex(&expressions).into_tokenizer(
            terminal_count,
            Some(Arc::from(expressions.into_boxed_slice())),
        )
    }

    fn vocab(entries: &[(u32, &[u8])]) -> Vocab {
        Vocab::new(
            entries
                .iter()
                .map(|&(token, bytes)| (token, bytes.to_vec()))
                .collect(),
            None,
        )
    }

    #[test]
    fn one_byte_token_finalizers_remain_distinct() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"a".to_vec()),
            Expr::U8Seq(b"b".to_vec()),
        ]);
        let (quotient, _) = compute_global_token_position_state_quotient(
            &tokenizer,
            &vocab(&[(0, b"a"), (1, b"b")]),
        );
        let state_a = tokenizer.step(tokenizer.initial_state_id(), b'a').unwrap() as usize;
        let state_b = tokenizer.step(tokenizer.initial_state_id(), b'b').unwrap() as usize;
        assert_ne!(
            quotient.as_many_to_one().original_to_internal[state_a],
            quotient.as_many_to_one().original_to_internal[state_b],
            "different frozen finalizers must not merge for one-byte tokens",
        );
    }

    #[test]
    fn token_boundary_classes_share_every_nonempty_token_trajectory() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"d".to_vec()),
            Expr::U8Seq(b"db".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"ab"), (1, b"ac"), (2, b"d"), (3, b"db")]);
        let partition = compute_global_token_boundary_state_partition(&tokenizer, &vocab)
            .expect("all fixture tokens are nonempty");
        let map = partition.as_many_to_one();

        for members in &map.internal_to_originals {
            let representative = *members.first().expect("total quotient class") as usize;
            for &state in members {
                for token in vocab.entries.values() {
                    let mut left = Some(state);
                    let mut right = Some(representative as u32);
                    for &byte in token {
                        left = left.and_then(|current| tokenizer.step(current, byte));
                        right = right.and_then(|current| tokenizer.step(current, byte));
                        assert_eq!(left, right, "boundary states diverged on token {:?}", token);
                    }
                }
            }
        }
    }
}
