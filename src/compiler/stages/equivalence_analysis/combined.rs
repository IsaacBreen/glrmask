use std::collections::{BTreeMap, BTreeSet};

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::equivalence_analysis::compat::TokenizerView;
use crate::compiler::stages::equivalence_analysis::combined_equivalence_analysis;
use crate::ds::bitset::BitSet;

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

fn adjust_disallowed_follows(
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) -> Option<BTreeMap<u32, BitSet>> {
    let ignore_terminal = ignore_terminal?;
    let mut adjusted = disallowed_follows.clone();
    adjusted.remove(&ignore_terminal);
    for bits in adjusted.values_mut() {
        if (ignore_terminal as usize) < bits.len() {
            bits.clear(ignore_terminal as usize);
        }
    }
    adjusted.retain(|_, bits| !bits.is_zero());
    Some(adjusted)
}

fn build_state_map(
    state_classes: &BTreeSet<BTreeSet<usize>>,
    num_dfa_states: usize,
) -> ManyToOneIdMap {
    let mut original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut internal_to_originals = Vec::new();
    let mut representative_original_ids = Vec::new();

    for class in state_classes {
        let internal_id = internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&state| state as u32).collect();
        for &state in &originals {
            original_to_internal[state as usize] = internal_id;
        }
        representative_original_ids
            .push(*originals.first().expect("state class must be non-empty"));
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn build_vocab_map(
    vocab_classes: &BTreeSet<Vec<usize>>,
    token_bytes: &[&[u8]],
    token_ids: &[u32],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let mut ordered_vocab_classes: Vec<(usize, Vec<u32>)> = vocab_classes
        .iter()
        .map(|class| {
            let mut indices: Vec<usize> = class.iter().copied().collect();
            indices.sort_unstable_by(|&left, &right| {
                token_bytes[left]
                    .cmp(&token_bytes[right])
                    .then_with(|| token_ids[left].cmp(&token_ids[right]))
            });
            let representative_idx = *indices.first().expect("vocab class must be non-empty");
            let originals = indices.into_iter().map(|idx| token_ids[idx]).collect();
            (representative_idx, originals)
        })
        .collect();
    ordered_vocab_classes.sort_unstable_by(|(left_rep_idx, _), (right_rep_idx, _)| {
        token_bytes[*left_rep_idx]
            .cmp(&token_bytes[*right_rep_idx])
            .then_with(|| token_ids[*left_rep_idx].cmp(&token_ids[*right_rep_idx]))
    });

    let mut original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut internal_to_originals = Vec::with_capacity(ordered_vocab_classes.len());
    let mut representative_original_ids = Vec::with_capacity(ordered_vocab_classes.len());

    for (internal_id, (_, originals)) in ordered_vocab_classes.into_iter().enumerate() {
        let representative = *originals.first().expect("vocab class must be non-empty");
        for &token_id in &originals {
            original_to_internal[token_id as usize] = internal_id as u32;
        }
        representative_original_ids.push(representative);
        internal_to_originals.push(originals);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

pub(crate) fn analyze_equivalences(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) -> InternalIdMap {
    analyze_equivalences_impl(tokenizer, vocab, disallowed_follows, ignore_terminal)
}

/// Combined equivalence analysis over a flattened tokenizer DFA.
///
/// Uses state equivalence (k-step hashing plus token-based refinement) followed
/// by vocab equivalence (parallel batched with byte-class compression).
fn analyze_equivalences_impl(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) -> InternalIdMap {
    let profile_compile = compile_profile_enabled();
    let total_started_at = std::time::Instant::now();

    let adjust_started_at = std::time::Instant::now();
    let adjusted_disallowed = adjust_disallowed_follows(disallowed_follows, ignore_terminal);
    let effective_disallowed = adjusted_disallowed.as_ref().unwrap_or(disallowed_follows);
    let adjust_ms = elapsed_ms(adjust_started_at);

    let tokenizer_view_started_at = std::time::Instant::now();
    let tokenizer_view = TokenizerView::new(tokenizer);
    let tokenizer_view_ms = elapsed_ms(tokenizer_view_started_at);

    // Extract vocab tokens as byte slices, ordered by token ID.
    let vocab_extract_started_at = std::time::Instant::now();
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }
    let vocab_extract_ms = elapsed_ms(vocab_extract_started_at);

    // All DFA states as initial states
    let initial_states_started_at = std::time::Instant::now();
    let initial_states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();
    let initial_states_ms = elapsed_ms(initial_states_started_at);

    let combined_started_at = std::time::Instant::now();
    let result = combined_equivalence_analysis::compute_combined_equivalence(
        &tokenizer_view,
        &token_bytes,
        &initial_states,
        effective_disallowed,
        ignore_terminal,
    );
    let combined_ms = elapsed_ms(combined_started_at);

    let state_map_started_at = std::time::Instant::now();
    let num_dfa_states = tokenizer.num_states() as usize;
    let state_map = build_state_map(&result.state_classes, num_dfa_states);
    let state_map_ms = elapsed_ms(state_map_started_at);

    let vocab_map_started_at = std::time::Instant::now();
    let vocab_map = build_vocab_map(
        &result.vocab_classes,
        &token_bytes,
        &token_ids,
        max_token_id,
    );
    let vocab_map_ms = elapsed_ms(vocab_map_started_at);

    if profile_compile {
        eprintln!(
            "[glrmask/profile][id_map] adjust_disallowed_ms={:.3} tokenizer_view_ms={:.3} vocab_extract_ms={:.3} initial_states_ms={:.3} combined_equiv_ms={:.3} build_state_map_ms={:.3} build_vocab_map_ms={:.3} total_ms={:.3}",
            adjust_ms,
            tokenizer_view_ms,
            vocab_extract_ms,
            initial_states_ms,
            combined_ms,
            state_map_ms,
            vocab_map_ms,
            elapsed_ms(total_started_at),
        );
    }

    InternalIdMap {
        tokenizer_states: state_map,
        vocab_tokens: vocab_map,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile::build_tokenizer;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    #[test]
    fn test_internal_id_map_shape() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..Default::default()
        };
        let tok = build_tokenizer(&gdef);
        let vocab = Vocab::new(
            vec![
                (0, b"a".to_vec()),
                (1, b"a".to_vec()),
                (2, b"b".to_vec()),
            ],
            None,
        );
        let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);

        assert!(id_map.num_tsids() >= 1);
        assert_eq!(id_map.max_token_id(), 2);
    }

    #[test]
    fn test_json_schema_equivalence_classes() {
        use crate::import::json_schema::json_schema_to_grammar;

        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        let grammar = json_schema_to_grammar(schema).expect("Schema should convert");
        let tok = build_tokenizer(&grammar);
        let vocab_strs = vec![
            "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g",
            "{\"", "\":",
        ];
        let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, s.as_bytes().to_vec()))
            .collect();
        let vocab = Vocab::new(vocab_entries, None);
        let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);
        let classes = &id_map.vocab_tokens.internal_to_originals;
        let expected: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![3],
            vec![4],
            vec![5],
            vec![6],
            vec![7],
            vec![8],
            vec![9],
            vec![10],
            vec![11],
            vec![12, 13],
            vec![14],
            vec![15],
        ];
        let mut expected_sorted: Vec<Vec<usize>> = expected
            .iter()
            .map(|class| {
                let mut sorted = class.clone();
                sorted.sort();
                sorted
            })
            .collect();
        expected_sorted.sort();
        let mut actual_sorted: Vec<Vec<usize>> = classes
            .iter()
            .map(|class| {
                let mut sorted: Vec<usize> = class.iter().map(|&id| id as usize).collect();
                sorted.sort();
                sorted
            })
            .collect();
        actual_sorted.sort();
        assert_eq!(
            actual_sorted,
            expected_sorted,
            "Equivalence classes don't match expected!\nExpected: {:?}\nActual:   {:?}",
            expected_sorted,
            actual_sorted,
        );
    }

    #[test]
    fn test_json_schema_equivalence_classes_simpler() {
        let grammar = crate::import::ebnf::parse_ebnf("root ::= '{' '}'")
            .expect("Grammar should build");
        let tok = build_tokenizer(&grammar);
        let vocab_entries = vec![(0, b"{".to_vec()), (1, b"}".to_vec())];
        let vocab = Vocab::new(vocab_entries, None);
        let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);
        let classes = &id_map.vocab_tokens.internal_to_originals;
        let expected = vec![vec![0], vec![1]];
        let mut expected_sorted: Vec<Vec<usize>> = expected
            .into_iter()
            .map(|mut class| {
                class.sort();
                class
            })
            .collect();
        expected_sorted.sort();
        let mut actual_sorted: Vec<Vec<usize>> = classes
            .iter()
            .map(|class| {
                let mut sorted: Vec<usize> = class.iter().map(|&id| id as usize).collect();
                sorted.sort();
                sorted
            })
            .collect();
        actual_sorted.sort();
        assert_eq!(
            actual_sorted,
            expected_sorted,
            "Equivalence classes don't match expected!\nExpected: {:?}\nActual:   {:?}",
            expected_sorted,
            actual_sorted,
        );
    }
}

