
use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::equivalence_analysis::compat::TokenizerView;
use crate::compiler::stages::equivalence_analysis::combined_equivalence_analysis;
use crate::ds::bitset::BitSet;

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
    // Adjust disallowed_follows for the ignore terminal:
    // - The ignore terminal can be followed by anything (remove its entry)
    // - Any terminal can be followed by the ignore terminal (clear it from all sets)
    let adjusted_disallowed;
    let effective_disallowed = if let Some(ign) = ignore_terminal {
        let mut adj = disallowed_follows.clone();
        adj.remove(&ign);
        for (_tid, bits) in adj.iter_mut() {
            if (ign as usize) < bits.len() {
                bits.clear(ign as usize);
            }
        }
        // Remove entries that became empty after clearing
        adj.retain(|_, bits| !bits.is_zero());
        adjusted_disallowed = adj;
        &adjusted_disallowed
    } else {
        disallowed_follows
    };

    let tokenizer_view = TokenizerView::new(tokenizer);

    // Extract vocab tokens as byte slices, ordered by token ID.
    // Vocab entries is a BTreeMap<u32, Vec<u8>>, so we need to handle sparse IDs.
    let max_token_id = vocab.max_token_id();
    let mut token_bytes: Vec<&[u8]> = Vec::with_capacity(vocab.len());
    let mut token_ids: Vec<u32> = Vec::with_capacity(vocab.len());
    for (&tid, bytes) in &vocab.entries {
        token_ids.push(tid);
        token_bytes.push(bytes.as_slice());
    }

    // All DFA states as initial states
    let initial_states: Vec<usize> = (0..tokenizer.num_states() as usize).collect();

    let result = combined_equivalence_analysis::compute_combined_equivalence(
        &tokenizer_view,
        &token_bytes,
        &initial_states,
        effective_disallowed,
        ignore_terminal,
    );

    // Convert state equivalence classes to ManyToOneIdMap
    let num_dfa_states = tokenizer.num_states() as usize;
    let mut state_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut state_internal_to_originals: Vec<RangeSetBlaze<u32>> = Vec::new();
    let mut state_representative_original_ids = Vec::new();

    for class in &result.state_classes {
        let internal_id = state_internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&s| s as u32).collect();
        for &s in &originals {
            state_original_to_internal[s as usize] = internal_id;
        }
        state_representative_original_ids.push(*originals.first().expect("state class must be non-empty"));
        state_internal_to_originals.push(RangeSetBlaze::from_iter(originals.iter().copied()));
    }

    let state_map = ManyToOneIdMap {
        original_to_internal: state_original_to_internal,
        internal_to_originals: state_internal_to_originals,
        representative_original_ids: state_representative_original_ids,
    };

    // Convert vocab equivalence classes to ManyToOneIdMap
    // The equivalence result uses indices into our token_bytes array, but we need
    // to map back to original token IDs.
    let mut ordered_vocab_classes: Vec<(usize, Vec<u32>)> = result
        .vocab_classes
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

    let mut vocab_original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut vocab_internal_to_originals: Vec<RangeSetBlaze<u32>> =
        Vec::with_capacity(ordered_vocab_classes.len());
    let mut vocab_representative_original_ids = Vec::with_capacity(ordered_vocab_classes.len());

    for (internal_id, (_, originals)) in ordered_vocab_classes.into_iter().enumerate() {
        let representative = *originals.first().expect("vocab class must be non-empty");
        for &tid in &originals {
            vocab_original_to_internal[tid as usize] = internal_id as u32;
        }
        vocab_representative_original_ids.push(representative);
        vocab_internal_to_originals.push(RangeSetBlaze::from_iter(originals.into_iter()));
    }

    let vocab_map = ManyToOneIdMap {
        original_to_internal: vocab_original_to_internal,
        internal_to_originals: vocab_internal_to_originals,
        representative_original_ids: vocab_representative_original_ids,
    };

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
            // Schema: {"type": "object", "properties": {"name": {"type": "string"}}}
            let schema = r#"{
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }"#;
            let grammar = json_schema_to_grammar(schema).expect("Schema should convert");
            let tok = build_tokenizer(&grammar);
            let vocab_strs = vec![
                "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g", "{\"", "\":"
            ];
            let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs.iter().enumerate().map(|(i, s)| (i as u32, s.as_bytes().to_vec())).collect();
            let vocab = Vocab::new(vocab_entries, None);
            let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);
            let classes = &id_map.vocab_tokens.internal_to_originals;
            // Combined state+vocab equivalence analysis groups tokens by their
            // behavior across all tokenizer states. Tokens "i" and "g" behave
            // identically (both are single-char string characters with the same
            // DFA transitions from every state), so they are merged into one class.
            let expected: Vec<Vec<usize>> = vec![
                vec![0],           // "{"
                vec![1],           // "}"
                vec![2],           // "\""
                vec![3],           // ":"
                vec![4],           // ","
                vec![5],           // "n"
                vec![6],           // "a"
                vec![7],           // "m"
                vec![8],           // "e"
                vec![9],           // "s"
                vec![10],          // "t"
                vec![11],          // "r"
                vec![12, 13],      // "i", "g" - equivalent string chars
                vec![14],          // "{\""
                vec![15],          // "\":"
            ];
            let mut expected_sorted: Vec<Vec<usize>> = expected.iter().map(|c| { let mut v = c.clone(); v.sort(); v }).collect();
            expected_sorted.sort();
            let mut actual_sorted: Vec<Vec<usize>> = classes.iter().map(|c| { let mut v: Vec<usize> = c.iter().map(|id| id as usize).collect(); v.sort(); v }).collect();
            actual_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted,
                "Equivalence classes don't match expected!\n\
                 Expected: {:?}\n\
                 Actual:   {:?}",
                expected_sorted, actual_sorted);
        }

        #[test]
        fn test_json_schema_equivalence_classes_simpler() {
            // Simple EBNF: root ::= '{' '}'
            let ebnf = "root ::= '{' '}'";
            let grammar = crate::import::ebnf::parse_ebnf(ebnf).expect("Grammar should build");
            let tok = build_tokenizer(&grammar);
            let vocab_strs = vec!["{", "}"];
            let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs.iter().enumerate().map(|(i, s)| (i as u32, s.as_bytes().to_vec())).collect();
            let vocab = Vocab::new(vocab_entries, None);
            let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);
            let classes = &id_map.vocab_tokens.internal_to_originals;
            let expected: Vec<Vec<usize>> = vec![vec![0], vec![1]];
            let mut expected_sorted: Vec<Vec<usize>> = expected.iter().map(|c| { let mut v = c.clone(); v.sort(); v }).collect();
            expected_sorted.sort();
            let mut actual_sorted: Vec<Vec<usize>> = classes.iter().map(|c| { let mut v: Vec<usize> = c.iter().map(|id| id as usize).collect(); v.sort(); v }).collect();
            actual_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted,
                "Equivalence classes don't match expected!\n\
                 Expected: {:?}\n\
                 Actual:   {:?}",
                expected_sorted, actual_sorted);
        }
    }

