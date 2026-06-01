//! Vocabulary quotienting by CanMatch signatures.
//!
//! Two vocabulary tokens may be merged for the scan relation only when, for
//! every lexer state, scanning their bytes exposes the same set of terminals
//! that could be completed.  This quotient is intentionally independent from
//! Terminal-DWA equivalence: completed-token behavior and partial-completion
//! behavior are different mathematical relations.

use super::prelude::*;
use super::ordered_vocab::OrderedVocab;

pub(super) fn scan_relation_vocab_equiv_enabled() -> bool {
    std::env::var("GLRMASK_SCAN_RELATION_VOCAB_EQUIV")
        .map(|value| {
            let trimmed = value.trim();
            trimmed.is_empty() || (trimmed != "0" && !trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
}

#[derive(Clone, Copy)]
struct CanMatchTokenOutcome {
    terminals: u128,
    end_state: u32,
}

#[inline]
fn mix_can_match_signature_word(hash: u64, word: u64) -> u64 {
    let mixed = word.wrapping_add(0x9e37_79b9_7f4a_7c15);
    hash ^ mixed
        .wrapping_add(hash << 6)
        .wrapping_add(hash >> 2)
        .wrapping_mul(0x517c_c1b7_2722_0a95)
}

fn can_match_signature_hash(outcomes: &[CanMatchTokenOutcome]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for outcome in outcomes {
        hash = mix_can_match_signature_word(hash, outcome.terminals as u64);
        hash = mix_can_match_signature_word(hash, (outcome.terminals >> 64) as u64);
    }
    hash
}

fn can_match_signature_matches(signature: &[u128], outcomes: &[CanMatchTokenOutcome]) -> bool {
    signature.len() == outcomes.len()
        && signature
            .iter()
            .zip(outcomes.iter())
            .all(|(&left, right)| left == right.terminals)
}

fn intern_can_match_token_signature(
    outcomes: &[CanMatchTokenOutcome],
    buckets: &mut FxHashMap<u64, Vec<u32>>,
    signatures: &mut Vec<Vec<u128>>,
) -> u32 {
    let hash = can_match_signature_hash(outcomes);
    if let Some(bucket) = buckets.get(&hash) {
        for &signature_id in bucket {
            if can_match_signature_matches(&signatures[signature_id as usize], outcomes) {
                return signature_id;
            }
        }
    }

    let signature_id = signatures.len() as u32;
    let signature = outcomes
        .iter()
        .map(|outcome| outcome.terminals)
        .collect::<Vec<_>>();
    signatures.push(signature);
    buckets.entry(hash).or_default().push(signature_id);
    signature_id
}

fn advance_can_match_token_outcomes(
    parent: &[CanMatchTokenOutcome],
    segment: &[u8],
    byte_transitions: &[Vec<u32>],
    matched_terminal_masks: &[u128],
) -> Vec<CanMatchTokenOutcome> {
    let mut child = Vec::with_capacity(parent.len());
    for &outcome in parent {
        let mut terminals = outcome.terminals;
        let mut current_state = outcome.end_state;
        if current_state != u32::MAX {
            for &byte in segment {
                let next_state = byte_transitions[byte as usize][current_state as usize];
                if next_state == u32::MAX {
                    current_state = u32::MAX;
                    break;
                }
                current_state = next_state;
                terminals |= matched_terminal_masks[current_state as usize];
            }
        }
        child.push(CanMatchTokenOutcome {
            terminals,
            end_state: current_state,
        });
    }
    child
}

struct CanMatchVocabEquivBuilder<'a> {
    ordered_vocab: &'a OrderedVocab,
    byte_transitions: &'a [Vec<u32>],
    matched_terminal_masks: &'a [u128],
    signature_buckets: FxHashMap<u64, Vec<u32>>,
    signatures: Vec<Vec<u128>>,
    original_to_internal: Vec<u32>,
    internal_to_originals: Vec<Vec<u32>>,
    representative_original_ids: Vec<u32>,
}

impl<'a> CanMatchVocabEquivBuilder<'a> {
    fn new(
        ordered_vocab: &'a OrderedVocab,
        byte_transitions: &'a [Vec<u32>],
        matched_terminal_masks: &'a [u128],
    ) -> Self {
        Self {
            ordered_vocab,
            byte_transitions,
            matched_terminal_masks,
            signature_buckets: FxHashMap::default(),
            signatures: Vec::new(),
            original_to_internal: vec![u32::MAX; ordered_vocab.original_slot_count],
            internal_to_originals: Vec::new(),
            representative_original_ids: Vec::new(),
        }
    }

    fn record_token(&mut self, ordered_token_id: usize, outcomes: &[CanMatchTokenOutcome]) {
        let class_id = intern_can_match_token_signature(
            outcomes,
            &mut self.signature_buckets,
            &mut self.signatures,
        );
        let class_idx = class_id as usize;
        while self.internal_to_originals.len() <= class_idx {
            self.internal_to_originals.push(Vec::new());
            self.representative_original_ids.push(u32::MAX);
        }
        let Some(originals) = self.ordered_vocab.ordered_to_originals.get(ordered_token_id) else {
            return;
        };
        for &original in originals {
            if let Some(slot) = self.original_to_internal.get_mut(original as usize) {
                *slot = class_id;
            }
            if self.representative_original_ids[class_idx] == u32::MAX {
                self.representative_original_ids[class_idx] = original;
            }
            self.internal_to_originals[class_idx].push(original);
        }
    }

    fn visit(&mut self, node: &VocabPrefixTreeNode, outcomes: &[CanMatchTokenOutcome]) {
        if node.has_token() {
            self.record_token(node.token_id(), outcomes);
        }
        for (segment, child) in node.iter_children() {
            let child_outcomes = advance_can_match_token_outcomes(
                outcomes,
                segment,
                self.byte_transitions,
                self.matched_terminal_masks,
            );
            self.visit(child, &child_outcomes);
        }
    }

    fn finish(mut self) -> ManyToOneIdMap {
        for originals in &mut self.internal_to_originals {
            originals.sort_unstable();
            originals.dedup();
        }
        ManyToOneIdMap {
            original_to_internal: self.original_to_internal,
            internal_to_originals: self.internal_to_originals,
            representative_original_ids: self.representative_original_ids,
        }
    }
}

pub(super) fn compute_scan_relation_vocab_equivalence_map(
    tokenizer: &Tokenizer,
    ordered_vocab: &OrderedVocab,
    trie: &VocabPrefixTree,
) -> ManyToOneIdMap {
    let num_states = tokenizer.num_states() as usize;
    let mut byte_transitions = vec![vec![u32::MAX; num_states]; 256];
    for (state_idx, dfa_state) in tokenizer.dfa.states().iter().enumerate() {
        for (byte, &target) in dfa_state.transitions.iter() {
            byte_transitions[byte as usize][state_idx] = target;
        }
    }

    let mut matched_terminal_masks = Vec::with_capacity(num_states);
    for state in 0..tokenizer.num_states() {
        let mut mask = 0u128;
        for terminal in tokenizer.matched_terminals_iter(state) {
            if terminal < 128 {
                mask |= 1u128 << terminal;
            }
        }
        matched_terminal_masks.push(mask);
    }

    let root_outcomes = (0..tokenizer.num_states())
        .map(|state| CanMatchTokenOutcome {
            terminals: 0,
            end_state: state,
        })
        .collect::<Vec<_>>();
    let mut builder = CanMatchVocabEquivBuilder::new(
        ordered_vocab,
        &byte_transitions,
        &matched_terminal_masks,
    );
    builder.visit(&trie.root, &root_outcomes);
    builder.finish()
}

pub(super) fn compute_scan_relation_vocab_equivalence_map_fast(
    tokenizer: &Tokenizer,
    ordered_vocab: &OrderedVocab,
) -> ManyToOneIdMap {
    let dfa_states = tokenizer.dfa.states();
    let num_states = dfa_states.len();
    let mut transitions = vec![u32::MAX; num_states * 256];
    let states = dfa_states
        .iter()
        .enumerate()
        .map(|(state_idx, state)| {
            let base = state_idx * 256;
            for (byte, &target) in state.transitions.iter() {
                transitions[base + byte as usize] = target;
            }
            FlatDfaState {
                finalizers: state.finalizers.iter().collect(),
                // CanMatch token equivalence only cares which terminals can be
                // reached while consuming the token bytes. Terminal-DWA
                // future-group metadata is deliberately not part of this CanMatch
                // owned equivalence relation.
                possible_future_group_ids: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    let tokenizer_view = TokenizerView {
        flat_dfa: FlatDfa {
            states,
            start_state: tokenizer.start_state() as usize,
            transitions: Arc::from(transitions),
        },
    };
    let strings = ordered_vocab
        .ordered_token_bytes
        .iter()
        .map(|bytes| bytes.as_slice())
        .collect::<Vec<_>>();
    let initial_states = (0..tokenizer.num_states() as usize).collect::<Vec<_>>();
    let disallowed_follows = BTreeMap::<u32, BitSet>::new();
    let classes = vocab_equivalence_analysis::find_vocab_equivalence_classes_with_group_filter(
        &tokenizer_view,
        &strings,
        &initial_states,
        &disallowed_follows,
        None,
        None,
        None,
    );

    let mut original_to_internal = vec![u32::MAX; ordered_vocab.original_slot_count];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();
    for class in classes {
        let internal = internal_to_originals.len() as u32;
        let mut originals = Vec::new();
        for ordered_id in class {
            if let Some(ordered_originals) = ordered_vocab.ordered_to_originals.get(ordered_id) {
                for &original in ordered_originals {
                    if let Some(slot) = original_to_internal.get_mut(original as usize) {
                        *slot = internal;
                    }
                    originals.push(original);
                }
            }
        }
        originals.sort_unstable();
        originals.dedup();
        let representative = originals.first().copied().unwrap_or(u32::MAX);
        internal_to_originals.push(originals);
        representative_original_ids.push(representative);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}
