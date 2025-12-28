// Tests for vocab equivalence analysis
// This module tests that the equivalence analysis correctly groups tokens

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use crate::finite_automata::{Regex, eat_u8, eat_u8_seq, rep, rep1, Expr, QuantifierType};
use crate::equivalence_analysis::compute_combined_equivalence;
use crate::tokenizer::LLMTokenID;
use crate::datastructures::u8set::U8Set;
use crate::interface::{GrammarDefinition, CompiledGrammar};
use crate::json_schema::json_schema_to_ebnf;
use crate::{choice, groups, seq};

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use super::*;

    /// Test equivalence analysis with the JSON schema tokenizer.
    /// This uses the same schema and vocab as test_small_vocab_only_brace_valid_at_start
    /// from test_json.rs to verify equivalence classes match expectations.
    ///
    /// Expected equivalence classes:
    /// - "{" solo
    /// - "}" solo
    /// - "\"" solo
    /// - ":" solo
    /// - "," solo
    /// - "n" solo
    /// - "a" solo
    /// - "m" solo
    /// - "e" solo
    /// - "{\"" solo
    /// - "\":" solo
    /// - "s", "t", "r", "i", "g" together
    #[test]
    fn test_json_schema_equivalence_classes() {
        // Same schema as test_small_vocab_only_brace_valid_at_start
        let schema = r#"{
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }"#;
        
        let ebnf = json_schema_to_ebnf(schema).expect("Schema should convert");
        let gd = GrammarDefinition::from_ebnf(&ebnf).expect("Grammar should build");
        
        // Build the tokenizer from the grammar
        let compiled = CompiledGrammar::from_definition(Arc::new(gd));
        let tokenizer = &compiled.tokenizer;
        
        // Same vocab as test_small_vocab_only_brace_valid_at_start
        let vocab_strs = vec![
            "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g", "{\"", "\":"
        ];
        let tokens: Vec<Vec<u8>> = vocab_strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        
        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        println!("Tokenizer has {} states", states.len());
        
        let start = std::time::Instant::now();
        let result = compute_combined_equivalence(tokenizer, &tokens, &states);
        println!("Analysis took {:?}", start.elapsed());
        
        let classes = result.vocab_classes;
        println!("Found {} vocab classes for {} tokens", classes.len(), tokens.len());
        
        for (i, class) in classes.iter().enumerate() {
            let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx]).collect();
            println!("  Class {}: {:?}", i, content);
        }
        
        // Build expected classes as sets of token indices
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
            vec![14],          // "{\""
            vec![15],          // "\":"
            vec![9, 10, 11, 12, 13], // "s", "t", "r", "i", "g"
        ];
        
        // Convert both to sorted format for comparison
        let mut expected_sorted: Vec<Vec<usize>> = expected.iter()
            .map(|c| { let mut v = c.clone(); v.sort(); v })
            .collect();
        expected_sorted.sort();
        
        let mut actual_sorted: Vec<Vec<usize>> = classes.iter()
            .map(|c| { let mut v = c.clone(); v.sort(); v })
            .collect();
        actual_sorted.sort();
        
        assert_eq!(actual_sorted, expected_sorted,
            "Equivalence classes don't match expected!\n\
             Expected: {:?}\n\
             Actual:   {:?}",
            expected_sorted, actual_sorted);
    }

    #[test]
    fn test_json_schema_equivalence_classes_simpler() {
        let ebnf = indoc! {r#"
            root ::= '{'  '}' ;
            #![ignore(WS)]
            WS ::= ' '* ;
        "#}.to_string();
        let gd = GrammarDefinition::from_ebnf(&ebnf).expect("Grammar should build");
        println!("Grammar definition: {}", gd);

        // Build the tokenizer from the grammar
        let compiled = CompiledGrammar::from_definition(Arc::new(gd));
        let tokenizer = &compiled.tokenizer;
        println!("Compiled grammar: {}", compiled);

        // Same vocab as test_small_vocab_only_brace_valid_at_start
        let vocab_strs = vec![
            "{", "}",
        ];
        let tokens: Vec<Vec<u8>> = vocab_strs.iter().map(|s| s.as_bytes().to_vec()).collect();

        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        println!("Tokenizer has {} states", states.len());

        let start = std::time::Instant::now();
        let result = compute_combined_equivalence(tokenizer, &tokens, &states);
        println!("Analysis took {:?}", start.elapsed());

        let classes = result.vocab_classes;
        println!("Found {} vocab classes for {} tokens", classes.len(), tokens.len());

        for (i, class) in classes.iter().enumerate() {
            let content: Vec<&str> = class.iter().map(|&idx| vocab_strs[idx]).collect();
            println!("  Class {}: {:?}", i, content);
        }

        // Build expected classes as sets of token indices
        let expected: Vec<Vec<usize>> = vec![
            vec![0],
            vec![1],
        ];

        // Convert both to sorted format for comparison
        let mut expected_sorted: Vec<Vec<usize>> = expected.iter()
            .map(|c| { let mut v = c.clone(); v.sort(); v })
            .collect();
        expected_sorted.sort();

        let mut actual_sorted: Vec<Vec<usize>> = classes.iter()
            .map(|c| { let mut v = c.clone(); v.sort(); v })
            .collect();
        actual_sorted.sort();

        assert_eq!(actual_sorted, expected_sorted,
            "Equivalence classes don't match expected!\n\
             Expected: {:?}\n\
             Actual:   {:?}",
            expected_sorted, actual_sorted);
    }
}
