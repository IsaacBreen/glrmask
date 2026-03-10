#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::equivalence_analysis::state_analysis::analyze_state_equivalences;
use crate::compiler::stages::equivalence_analysis::vocab_trellis::analyze_vocab_equivalences_trellis;

pub(crate) fn analyze_equivalences(tokenizer: &Tokenizer, vocab: &Vocab) -> InternalIdMap {
    let state_map = analyze_state_equivalences(tokenizer);

    // Collect representative states for trellis-based vocab analysis.
    let representative_states: Vec<u32> = state_map
        .internal_to_originals
        .iter()
        .filter_map(|originals| originals.first().copied())
        .collect();

    let vocab_map = analyze_vocab_equivalences_trellis(tokenizer, vocab, &representative_states);
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
        let id_map = analyze_equivalences(&tok, &vocab);

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
            let id_map = analyze_equivalences(&tok, &vocab);
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
            let id_map = analyze_equivalences(&tok, &vocab);
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
}
