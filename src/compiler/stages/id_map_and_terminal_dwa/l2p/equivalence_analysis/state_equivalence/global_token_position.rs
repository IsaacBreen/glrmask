//! Exact global scanner-state equivalence induced by token positions.
//!
//! This is the token-position partition C, built from two token-position
//! observations:
//!
//! * every first-byte destination is retained exactly (partition A); and
//! * every state reachable at byte position three or later is singleton
//!   (partition B).
//!
//! It is a token-boundary relation: a state class is only entered before the
//! first byte of a vocabulary token. It is intentionally not strengthened by
//! frozen-output equality nor closed to a raw-byte congruence.

use std::collections::VecDeque;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{GlobalScannerStateQuotient, ManyToOneIdMap};

#[derive(Debug, Clone, Default)]
pub(crate) struct GlobalTokenPositionEquivalenceProfile {
    pub(crate) first_byte_count: usize,
    pub(crate) remaining_byte_count: usize,
    pub(crate) second_byte_count: usize,
    pub(crate) second_state_count: usize,
    pub(crate) third_plus_state_count: usize,
    pub(crate) first_destination_class_count: usize,
    pub(crate) seed_class_count: usize,
    pub(crate) seed_ms: f64,
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
    third_plus_singleton: u32,
    first_destinations: Box<[u32]>,
}

struct NfaTokenPositionView<'a> {
    tokenizer: &'a Tokenizer,
    config_ids: FxHashMap<Vec<u32>, u32>,
    configs: Vec<Box<[u32]>>,
    raw_start_configs: Vec<u32>,
    transitions: FxHashMap<(u32, u8), u32>,
}

impl<'a> NfaTokenPositionView<'a> {
    fn new(tokenizer: &'a Tokenizer) -> Self {
        let mut view = Self {
            tokenizer,
            config_ids: FxHashMap::default(),
            configs: Vec::new(),
            raw_start_configs: Vec::with_capacity(tokenizer.num_states() as usize),
            transitions: FxHashMap::default(),
        };
        for raw_state in 0..tokenizer.num_states() {
            let closure = tokenizer
                .execute_from_state_end_only(&[], raw_state)
                .to_vec();
            let config = view.intern_config(closure);
            view.raw_start_configs.push(config);
        }
        view
    }

    fn intern_config(&mut self, states: Vec<u32>) -> u32 {
        if let Some(&config) = self.config_ids.get(&states) {
            return config;
        }
        let config = self.configs.len() as u32;
        self.config_ids.insert(states.clone(), config);
        self.configs.push(states.into_boxed_slice());
        config
    }

    #[inline]
    fn raw_start_config(&self, raw_state: usize) -> u32 {
        self.raw_start_configs[raw_state]
    }

    fn step(&mut self, config: u32, byte: u8) -> u32 {
        if let Some(&target) = self.transitions.get(&(config, byte)) {
            return target;
        }
        let source = self.configs[config as usize].to_vec();
        let targets = self.tokenizer.step_all(&source, byte);
        let target = if targets.is_empty() {
            u32::MAX
        } else {
            self.intern_config(targets.to_vec())
        };
        self.transitions.insert((config, byte), target);
        target
    }

    #[inline]
    fn states(&self, config: u32) -> &[u32] {
        &self.configs[config as usize]
    }

    fn has_finalizer(&self, config: u32) -> bool {
        self.states(config)
            .iter()
            .any(|&state| !self.tokenizer.matched_terminal_bitset(state).is_empty())
    }

    fn outgoing_bytes(&self, config: u32, bytes: &mut Vec<u8>) {
        let mut seen = [false; 256];
        bytes.clear();
        for &state in self.states(config) {
            for (byte, _) in self.tokenizer.transitions_from(state) {
                if !seen[byte as usize] {
                    seen[byte as usize] = true;
                    bytes.push(byte);
                }
            }
        }
    }
}

fn first_destination_rows(tokenizer: &Tokenizer, first_bytes: &[u8]) -> Vec<Box<[u32]>> {
    if !tokenizer.has_epsilon_transitions() {
        return (0..tokenizer.num_states())
            .map(|state| {
                first_bytes
                    .iter()
                    .map(|&byte| tokenizer.get_transition(state, byte))
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect();
    }

    let mut view = NfaTokenPositionView::new(tokenizer);
    (0..tokenizer.num_states() as usize)
        .map(|state| {
            let source = view.raw_start_config(state);
            first_bytes
                .iter()
                .map(|&byte| view.step(source, byte))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect()
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
    if tokenizer.has_epsilon_transitions() {
        let mut view = NfaTokenPositionView::new(tokenizer);
        let mut states = vec![false; state_count];
        let mut reset_needed = false;
        for source in 0..state_count {
            let source_config = view.raw_start_config(source);
            for &byte in first_bytes {
                let destination = view.step(source_config, byte);
                if destination == u32::MAX {
                    continue;
                }
                reset_needed |= view.has_finalizer(destination);
                for &state in view.states(destination) {
                    states[state as usize] = true;
                }
            }
        }
        if reset_needed {
            let reset = view.raw_start_config(tokenizer.initial_state_id() as usize);
            for &state in view.states(reset) {
                states[state as usize] = true;
            }
        }
        return states;
    }
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
    if tokenizer.has_epsilon_transitions() {
        let mut view = NfaTokenPositionView::new(tokenizer);
        let mut reached = vec![false; state_count];
        let mut worklist = VecDeque::new();

        for (state, &is_second_state) in second_states.iter().enumerate() {
            if !is_second_state {
                continue;
            }
            let source = view.raw_start_config(state);
            for &byte in second_bytes {
                let destination = view.step(source, byte);
                if destination == u32::MAX {
                    continue;
                }
                for &target in view.states(destination) {
                    if !reached[target as usize] {
                        reached[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
            }
        }

        let mut bytes = Vec::<u8>::new();
        while let Some(state) = worklist.pop_front() {
            let source = view.raw_start_config(state as usize);
            view.outgoing_bytes(source, &mut bytes);
            for byte in bytes.clone() {
                let destination = view.step(source, byte);
                if destination == u32::MAX {
                    continue;
                }
                for &target in view.states(destination) {
                    if !reached[target as usize] {
                        reached[target as usize] = true;
                        worklist.push_back(target);
                    }
                }
            }
        }
        return reached;
    }
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
    third_plus: &[bool],
) -> (Vec<u32>, usize) {
    let state_count = tokenizer.num_states() as usize;
    let first_destinations = first_destination_rows(tokenizer, first_bytes);
    // Group states by their exact A∧B seed key (first-byte destinations and a
    // position->=3 singleton marker) via a 128-bit fingerprint. Position->=3
    // states fold their own index into the hash so they remain singletons. A
    // false collision between distinct keys is ~states^2/2^128 -- negligible --
    // and avoids allocating a Box<[u32]> destination key per state. Exactness
    // is backstopped by the strict-reference validator.
    let mut key_to_class = FxHashMap::<(u64, u64), u32>::default();
    key_to_class.reserve(state_count);
    let mut blocks = vec![u32::MAX; state_count];
    let mut class_count = 0u32;

    for state in 0..state_count {
        let mut hash_a = 0x9e37_79b9_7f4a_7c15u64;
        let mut hash_b = 0xd1b5_4a32_d192_ed03u64;
        for &destination in first_destinations[state].iter() {
            let destination = destination as u64;
            hash_a = hash_a
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(destination.wrapping_add(1));
            hash_b = hash_b
                .wrapping_mul(0x2545_f491_4f6c_dd1d)
                .wrapping_add(destination.rotate_left(23) ^ 0xa24b_aed4_963e_e407);
        }
        if third_plus[state] {
            // Distinguish this state from every other so it stays a singleton.
            let marker = (state as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
            hash_a = hash_a.wrapping_mul(0x517c_c1b7_2722_0a95).wrapping_add(marker);
            hash_b = hash_b
                .wrapping_mul(0x2545_f491_4f6c_dd1d)
                .wrapping_add(marker.rotate_left(31) ^ 0xbf58_476d_1ce4_e5b9);
        }
        let next = class_count;
        let class = *key_to_class.entry((hash_a, hash_b)).or_insert_with(|| {
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
    let first_destinations = first_destination_rows(tokenizer, first_bytes);
    let mut key_to_class = FxHashMap::<TokenPositionSeedKey, u32>::default();
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for state in 0..state_count {
        let key = TokenPositionSeedKey {
            third_plus_singleton: third_plus[state].then_some(state as u32).unwrap_or(u32::MAX),
            first_destinations: first_destinations[state].clone(),
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
) -> usize {
    let first_destinations = first_destination_rows(tokenizer, first_bytes);
    let mut classes = FxHashMap::<TokenPositionSeedKey, u32>::default();
    for state in 0..tokenizer.num_states() as usize {
        let key = TokenPositionSeedKey {
            third_plus_singleton: third_plus
                .is_some_and(|states| states[state])
                .then_some(state as u32)
                .unwrap_or(u32::MAX),
            first_destinations: first_destinations[state].clone(),
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
    let first_destinations = first_destination_rows(tokenizer, first_bytes);
    let mut key_to_class = FxHashMap::<Box<[u32]>, u32>::default();
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for state in 0..state_count {
        let destinations = first_destinations[state].clone();
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
    let (blocks, class_count) = seed_partition(tokenizer, &first_bytes, &third_plus);
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

/// Build the total scanner-state quotient before TI. This is exactly the
/// requested A∧B token-position partition C: first-byte destinations plus
/// position->=3 singletons, with no frozen-output strengthening and no
/// raw-byte congruence closure.
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
    let first_destination_class_count =
        first_destination_partition_class_count(tokenizer, &first_bytes, None);
    let (seed_blocks, seed_class_count) =
        seed_partition(tokenizer, &first_bytes, &third_plus);
    let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;

    let state_count = tokenizer.num_states() as usize;
    let quotient = GlobalScannerStateQuotient::from_total_raw_state_map(
        map_from_blocks(seed_blocks, seed_class_count),
        state_count,
    );
    let profile = GlobalTokenPositionEquivalenceProfile {
        first_byte_count: first_bytes.len(),
        remaining_byte_count: remaining_bytes.len(),
        second_byte_count: second_bytes.len(),
        second_state_count: second_states.iter().filter(|&&present| present).count(),
        third_plus_state_count: third_plus.iter().filter(|&&present| present).count(),
        first_destination_class_count,
        seed_class_count,
        seed_ms,
        total_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    };
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        let first_bytes_preview: String = first_bytes
            .iter()
            .take(40)
            .map(|&b| {
                if b.is_ascii_graphic() || b == b' ' {
                    format!("{}", b as char)
                } else {
                    format!("\\x{:02x}", b)
                }
            })
            .collect::<Vec<_>>()
            .join("");
        eprintln!(
            "[glrmask/profile][global_token_position_quotient] raw_states={} first_byte_count={} second_byte_count={} remaining_byte_count={} first_destination_classes={} seed_classes={} seed_ms={:.3} total_ms={:.3} first_bytes[<=40]=\"{}\"",
            state_count,
            profile.first_byte_count,
            profile.second_byte_count,
            profile.remaining_byte_count,
            profile.first_destination_class_count,
            profile.seed_class_count,
            profile.seed_ms,
            profile.total_ms,
            first_bytes_preview,
        );
    }
    (quotient, profile)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::automata::lexer::ast::Expr;
    use crate::automata::lexer::compile::build_regex;
    use crate::automata::lexer::tokenizer::arbitrary_epsilon_l1_test_tokenizer;

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

    #[test]
    fn epsilon_token_boundary_classes_share_exact_state_set_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let vocab = vocab(&[(0, b"a"), (1, b"b"), (2, b"aa"), (3, b"ba")]);
        let partition = compute_global_token_boundary_state_partition(&tokenizer, &vocab)
            .expect("all fixture tokens are nonempty");
        let map = partition.as_many_to_one();

        for members in &map.internal_to_originals {
            let representative = *members.first().expect("total quotient class");
            for &state in members {
                for token in vocab.entries.values() {
                    let mut left = tokenizer.execute_from_state_end_only(&[], state);
                    let mut right = tokenizer.execute_from_state_end_only(&[], representative);
                    for &byte in token {
                        left = tokenizer.step_all(&left, byte);
                        right = tokenizer.step_all(&right, byte);
                        assert_eq!(
                            left, right,
                            "epsilon boundary states diverged on token {token:?}",
                        );
                    }
                }
            }
        }

        let (quotient, _) = compute_global_token_position_state_quotient(&tokenizer, &vocab);
        assert_eq!(quotient.raw_state_count(), tokenizer.num_states() as usize);
    }
}
