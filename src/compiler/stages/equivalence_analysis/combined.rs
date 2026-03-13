
use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::equivalence_analysis::compat::Sep1Tokenizer;
use crate::compiler::stages::equivalence_analysis::combined_equivalence_analysis;
use crate::ds::bitset::BitSet;

pub(crate) fn analyze_equivalences(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<u32>,
) -> InternalIdMap {
    analyze_equivalences_sep1(tokenizer, vocab, disallowed_follows, ignore_terminal)
}

/// Sep1-derived combined equivalence analysis.
///
/// Uses the ported sep1 pipeline: state equivalence (k-step hashing + token-based
/// refinement) followed by vocab equivalence (parallel batched with byte-class
/// compression). Cross-validates against simple and flat implementations in tests.
fn analyze_equivalences_sep1(
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

    let sep1_tok = Sep1Tokenizer::new(tokenizer);

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
        &sep1_tok,
        &token_bytes,
        &initial_states,
        effective_disallowed,
        ignore_terminal,
    );

    // Convert state equivalence classes to ManyToOneIdMap
    let num_dfa_states = tokenizer.num_states() as usize;
    let mut state_original_to_internal = vec![u32::MAX; num_dfa_states];
    let mut state_internal_to_originals: Vec<Vec<u32>> = Vec::new();

    for class in &result.state_classes {
        let internal_id = state_internal_to_originals.len() as u32;
        let originals: Vec<u32> = class.iter().map(|&s| s as u32).collect();
        for &s in &originals {
            state_original_to_internal[s as usize] = internal_id;
        }
        state_internal_to_originals.push(originals);
    }

    let state_map = ManyToOneIdMap {
        original_to_internal: state_original_to_internal,
        internal_to_originals: state_internal_to_originals,
    };

    // Convert vocab equivalence classes to ManyToOneIdMap
    // The sep1 result uses indices into our token_bytes array, but we need
    // to map back to original token IDs.
    let mut vocab_original_to_internal = vec![u32::MAX; (max_token_id + 1) as usize];
    let mut vocab_internal_to_originals: Vec<Vec<u32>> = Vec::new();

    for class in &result.vocab_classes {
        let internal_id = vocab_internal_to_originals.len() as u32;
        let mut originals: Vec<u32> = class.iter().map(|&idx| token_ids[idx]).collect();
        // Sort so the shortest token (by byte length) comes first.
        // This makes it the representative, which downstream code picks via .first().
        originals.sort_by_key(|&tid| {
            vocab.entries.get(&tid).map_or(usize::MAX, |b| b.len())
        });
        for &tid in &originals {
            vocab_original_to_internal[tid as usize] = internal_id;
        }
        vocab_internal_to_originals.push(originals);
    }

    let vocab_map = ManyToOneIdMap {
        original_to_internal: vocab_original_to_internal,
        internal_to_originals: vocab_internal_to_originals,
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
        fn test_json_schema_equivalence_classes_port() {
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
            // Print for debugging
            for (i, class) in classes.iter().enumerate() {
                let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx as usize]).collect();
                println!("  Class {}: {:?}", i, content);
            }
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
            let mut actual_sorted: Vec<Vec<usize>> = classes.iter().map(|c| { let mut v: Vec<usize> = c.iter().map(|&id| id as usize).collect(); v.sort(); v }).collect();
            actual_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted,
                "Equivalence classes don't match expected!\n\
                 Expected: {:?}\n\
                 Actual:   {:?}",
                expected_sorted, actual_sorted);
        }

        #[test]
        fn test_json_schema_equivalence_classes_simpler_port() {
            // Simple EBNF: root ::= '{' '}'
            let ebnf = "root ::= '{' '}'";
            let grammar = crate::import::ebnf::parse_ebnf(ebnf).expect("Grammar should build");
            let tok = build_tokenizer(&grammar);
            let vocab_strs = vec!["{", "}"];
            let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs.iter().enumerate().map(|(i, s)| (i as u32, s.as_bytes().to_vec())).collect();
            let vocab = Vocab::new(vocab_entries, None);
            let id_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);
            let classes = &id_map.vocab_tokens.internal_to_originals;
            for (i, class) in classes.iter().enumerate() {
                let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx as usize]).collect();
                println!("  Class {}: {:?}", i, content);
            }
            let expected: Vec<Vec<usize>> = vec![vec![0], vec![1]];
            let mut expected_sorted: Vec<Vec<usize>> = expected.iter().map(|c| { let mut v = c.clone(); v.sort(); v }).collect();
            expected_sorted.sort();
            let mut actual_sorted: Vec<Vec<usize>> = classes.iter().map(|c| { let mut v: Vec<usize> = c.iter().map(|&id| id as usize).collect(); v.sort(); v }).collect();
            actual_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted,
                "Equivalence classes don't match expected!\n\
                 Expected: {:?}\n\
                 Actual:   {:?}",
                expected_sorted, actual_sorted);
        }

        /// Diagnostic test: measures equivalence analysis effectiveness.
        /// Run with: cargo test --lib measure_equivalence_effectiveness -- --nocapture
        #[test]
        fn measure_equivalence_effectiveness() {
            use crate::import::json_schema::json_schema_to_grammar;

            let schema = r#"{
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }"#;
            let grammar = json_schema_to_grammar(schema).expect("Schema should convert");
            let tok = build_tokenizer(&grammar);

            // Build a realistic-ish vocab with 16 entries
            let vocab_strs = vec![
                "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g", "{\"", "\":"
            ];
            let vocab_entries: Vec<(u32, Vec<u8>)> = vocab_strs.iter().enumerate()
                .map(|(i, s)| (i as u32, s.as_bytes().to_vec())).collect();
            let vocab = Vocab::new(vocab_entries, None);

            // Full combined analysis (sep1 pipeline)
            let full_map = analyze_equivalences(&tok, &vocab, &BTreeMap::new(), None);
            let num_original_states = tok.num_states();
            let num_combined_state_classes = full_map.num_tsids();
            let num_combined_vocab_classes = full_map.num_internal_tokens();

            println!("\n=== Equivalence Analysis Effectiveness (sep1 pipeline) ===");
            println!("Grammar: JSON schema (object with 'name' string property)");
            println!("Vocab: {} entries", vocab_strs.len());
            println!();
            println!("STATE EQUIVALENCE:");
            println!("  Original DFA states:     {}", num_original_states);
            println!("  State equiv classes:     {}", num_combined_state_classes);
            println!("  Compression ratio:       {:.1}x", num_original_states as f64 / num_combined_state_classes as f64);
            println!();
            println!("VOCAB EQUIVALENCE:");
            println!("  Original vocab entries:  {}", vocab_strs.len());
            println!("  Vocab classes:           {}", num_combined_vocab_classes);

            // Show classes
            println!();
            println!("Vocab classes:");
            for (i, class) in full_map.vocab_tokens.internal_to_originals.iter().enumerate() {
                let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx as usize]).collect();
                println!("  Class {}: {:?}", i, content);
            }
        }
}
