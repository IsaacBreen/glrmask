#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::equivalence_analysis::state_analysis::analyze_state_equivalences;
use crate::compiler::stages::equivalence_analysis::vocab_analysis::analyze_vocab_equivalences;

pub(crate) fn analyze_equivalences(tokenizer: &Tokenizer, vocab: &Vocab) -> InternalIdMap {
    InternalIdMap {
        tokenizer_states: analyze_state_equivalences(tokenizer),
        vocab_tokens: analyze_vocab_equivalences(vocab),
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
            // Current equivalence analysis groups by byte identity only.
            // All 16 tokens have distinct bytes, so each gets its own class.
            //
            // GOAL: When combined state+vocab analysis is effective (like sep1),
            // tokens "i" and "g" should be grouped together as they behave
            // identically across all tokenizer states. That would give 15 classes
            // with vec![12, 13] merged.
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
                vec![12],          // "i"
                vec![13],          // "g"
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
