//! Exact global scanner-state equivalence at vocabulary-token boundaries.
//!
//! Partition C groups two raw scanner states exactly when every byte that can
//! begin a nonempty token in this vocabulary sends them to the same scanner
//! state (or, for an epsilon lexer, the same scanner-state configuration).
//!
//! Therefore every complete vocabulary token has the same scanner trajectory
//! from either member after its first byte. This is a token-boundary relation,
//! not a raw-byte DFA quotient: current outputs and transitions on bytes that
//! cannot begin a vocabulary token need not agree.

use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equiv_types::{GlobalScannerStateQuotient, ManyToOneIdMap};

#[derive(Debug, Clone, Default)]
pub(crate) struct GlobalTokenPositionEquivalenceProfile {
    pub(crate) first_byte_count: usize,
    pub(crate) class_count: usize,
    pub(crate) build_ms: f64,
    pub(crate) total_ms: f64,
}

/// Global token-position partition C.
///
/// States are grouped by their exact destinations on every possible token-first
/// byte. A class is only substituted at a vocabulary-token boundary.
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

/// Determinized state-set view used only to define the first-byte destination
/// row for a lexer with epsilon transitions.
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
}

fn token_first_bytes(vocab: &Vocab) -> Vec<u8> {
    let mut first = [false; 256];
    for bytes in vocab.entries.values() {
        if let Some(&byte) = bytes.first() {
            first[byte as usize] = true;
        }
    }
    (0..=u8::MAX)
        .filter(|&byte| first[byte as usize])
        .collect()
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

fn first_destination_partition(tokenizer: &Tokenizer, first_bytes: &[u8]) -> ManyToOneIdMap {
    let state_count = tokenizer.num_states() as usize;
    let first_destinations = first_destination_rows(tokenizer, first_bytes);
    let mut key_to_class = FxHashMap::<Box<[u32]>, u32>::default();
    let mut original_to_internal = vec![u32::MAX; state_count];
    let mut internal_to_originals = Vec::<Vec<u32>>::new();
    let mut representative_original_ids = Vec::<u32>::new();

    for (state, destinations) in first_destinations.into_iter().enumerate() {
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

/// Build partition C. For every nonempty vocabulary token, equivalent starts
/// reach the same scanner state after byte one and therefore have identical
/// scanner behavior for the entire remaining token suffix.
pub(crate) fn compute_global_token_position_state_partition(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> Option<GlobalTokenPositionStatePartition> {
    if vocab.entries.values().any(Vec::is_empty) {
        return None;
    }
    let first_bytes = token_first_bytes(vocab);
    (!first_bytes.is_empty()).then(|| GlobalTokenPositionStatePartition {
        map: first_destination_partition(tokenizer, &first_bytes),
    })
}

/// Wrap the same partition C as a total raw-state quotient. The relation itself
/// remains token-boundary-specific; consumers must not assume raw-byte
/// congruence or current-output equality.
pub(crate) fn compute_global_token_position_state_quotient(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
) -> (GlobalScannerStateQuotient, GlobalTokenPositionEquivalenceProfile) {
    let started_at = Instant::now();
    let first_bytes = token_first_bytes(vocab);

    let build_started_at = Instant::now();
    let map = first_destination_partition(tokenizer, &first_bytes);
    let class_count = map.representative_original_ids.len();
    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;

    let state_count = tokenizer.num_states() as usize;
    let quotient = GlobalScannerStateQuotient::from_total_raw_state_map(map, state_count);
    let profile = GlobalTokenPositionEquivalenceProfile {
        first_byte_count: first_bytes.len(),
        class_count,
        build_ms,
        total_ms: started_at.elapsed().as_secs_f64() * 1000.0,
    };
    if std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some() {
        let first_bytes_preview = first_bytes
            .iter()
            .take(40)
            .map(|&byte| {
                if byte.is_ascii_graphic() || byte == b' ' {
                    format!("{}", byte as char)
                } else {
                    format!("\\x{byte:02x}")
                }
            })
            .collect::<Vec<_>>()
            .join("");
        eprintln!(
            "[glrmask/profile][global_token_position_quotient] raw_states={} first_byte_count={} classes={} build_ms={:.3} total_ms={:.3} first_bytes[<=40]=\"{}\"",
            state_count,
            profile.first_byte_count,
            profile.class_count,
            profile.build_ms,
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
    fn c_classes_share_every_nonempty_token_trajectory() {
        let tokenizer = tokenizer(vec![
            Expr::U8Seq(b"ab".to_vec()),
            Expr::U8Seq(b"ac".to_vec()),
            Expr::U8Seq(b"d".to_vec()),
            Expr::U8Seq(b"db".to_vec()),
        ]);
        let vocab = vocab(&[(0, b"ab"), (1, b"ac"), (2, b"d"), (3, b"db")]);
        let partition = compute_global_token_position_state_partition(&tokenizer, &vocab)
            .expect("all fixture tokens are nonempty");
        let map = partition.as_many_to_one();

        for members in &map.internal_to_originals {
            let representative = *members.first().expect("total quotient class");
            for &state in members {
                for token in vocab.entries.values() {
                    let mut left = Some(state);
                    let mut right = Some(representative);
                    for &byte in token {
                        left = left.and_then(|current| tokenizer.step(current, byte));
                        right = right.and_then(|current| tokenizer.step(current, byte));
                        assert_eq!(left, right, "C states diverged on token {token:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn epsilon_c_classes_share_exact_state_set_trajectories() {
        let tokenizer = arbitrary_epsilon_l1_test_tokenizer();
        let vocab = vocab(&[(0, b"a"), (1, b"b"), (2, b"aa"), (3, b"ba")]);
        let partition = compute_global_token_position_state_partition(&tokenizer, &vocab)
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
                        assert_eq!(left, right, "epsilon C states diverged on token {token:?}");
                    }
                }
            }
        }

        let (quotient, _) = compute_global_token_position_state_quotient(&tokenizer, &vocab);
        assert_eq!(quotient.raw_state_count(), tokenizer.num_states() as usize);
        assert_eq!(
            quotient.as_many_to_one().original_to_internal,
            map.original_to_internal,
            "partition and quotient wrappers must expose the same C relation",
        );
    }
}
