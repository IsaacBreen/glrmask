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
    use super::*;
    
    /// Test with a multi-group tokenizer where each single-byte token has its own group.
    /// This is the NON-optimized grammar case.
    /// 
    /// EXPECTED: Each token should be in its own equivalence class because they
    /// finalize different groups.
    #[test]
    fn test_multi_group_tokenizer_separates_tokens() {
        // Create a tokenizer with SEPARATE groups for different tokens
        // Each group has a different ID (0, 1, 2, 3, 4)
        let tokenizer = groups![
            eat_u8(b'{'),   // group 0
            eat_u8(b'}'),   // group 1
            eat_u8(b':'),   // group 2
            eat_u8(b','),   // group 3
            eat_u8(b'"')    // group 4
        ].build();
        
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'{'],
            vec![b'}'],
            vec![b':'],
            vec![b','],
            vec![b'"'],
        ];
        
        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        
        let classes = compute_combined_equivalence(&tokenizer, &tokens, &states).vocab_classes;
        
        println!("Multi-group tokenizer equivalence classes:");
        for (i, class) in classes.iter().enumerate() {
            println!("  Class {}: {:?}", i, class);
        }
        
        // With separate groups, each token should finalize a different group,
        // so they should all be in separate equivalence classes
        assert_eq!(classes.len(), 5, 
            "With separate groups, each token should be in its own class");
    }

    /// Test equivalence analysis with the JSON schema tokenizer.
    /// This uses the same schema and vocab as test_small_vocab_only_brace_valid_at_start
    /// from test_json.rs to verify equivalence classes match expectations.
    ///
    /// Expected equivalence classes:
    /// - internal 0 -> [2, 15] = '"', '":'
    /// - internal 1 -> [0, 1, 3, 4] = '{', '}', ':', ','
    /// - internal 2 -> [6, 8] = 'a', 'e'
    /// - internal 3 -> [7, 9, 12, 13] = 'm', 's', 'i', 'g'
    /// - internal 4 -> [5, 10, 11] = 'n', 't', 'r'
    /// - internal 5 -> [14] = '{"'
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
        
        // Expected equivalence classes (from test_small_vocab_only_brace_valid_at_start):
        // - internal 0 -> [2, 15] = '"', '":'
        // - internal 1 -> [0, 1, 3, 4] = '{', '}', ':', ','
        // - internal 2 -> [6, 8] = 'a', 'e'
        // - internal 3 -> [7, 9, 12, 13] = 'm', 's', 'i', 'g'
        // - internal 4 -> [5, 10, 11] = 'n', 't', 'r'
        // - internal 5 -> [14] = '{"'
        
        // Build expected classes as sets of token indices
        let expected: Vec<Vec<usize>> = vec![
            vec![2, 15],       // '"', '":'
            vec![0, 1, 3, 4],  // '{', '}', ':', ','
            vec![6, 8],        // 'a', 'e'
            vec![7, 9, 12, 13], // 'm', 's', 'i', 'g'
            vec![5, 10, 11],   // 'n', 't', 'r'
            vec![14],          // '{"'
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
