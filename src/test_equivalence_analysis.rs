// Tests for vocab equivalence analysis
// This module tests that the equivalence analysis correctly groups tokens

use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::{Regex, eat_u8, rep1, Expr};
use crate::equivalence_analysis::vocab_equivalence_analysis_fast::find_vocab_equivalence_classes;
use crate::tokenizer::LLMTokenID;
use crate::datastructures::u8set::U8Set;
use crate::{choice, groups};

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
        
        let classes = find_vocab_equivalence_classes(&tokenizer, &tokens, &states);
        
        println!("Multi-group tokenizer equivalence classes:");
        for (i, class) in classes.iter().enumerate() {
            println!("  Class {}: {:?}", i, class);
        }
        
        // With separate groups, each token should finalize a different group,
        // so they should all be in separate equivalence classes
        assert_eq!(classes.len(), 5, 
            "With separate groups, each token should be in its own class");
    }
    
    /// Test with a single-group tokenizer where all patterns share one group ID.
    /// This simulates the optimized grammar case where all terminals are collapsed
    /// into a single `__optimized_terminal__` regex.
    ///
    /// CURRENT BEHAVIOR: All tokens get merged into one class because they all
    /// finalize the same group (0) from the same tokenizer state.
    ///
    /// This is the ROOT CAUSE of the bug where invalid tokens are marked as valid.
    /// The equivalence analysis is "correct" from a tokenizer perspective, but
    /// grammar constraints aren't being properly applied downstream.
    #[test]
    fn test_single_group_tokenizer_should_separate_tokens() {
        // Create a tokenizer with ONE group that matches several single characters
        // This simulates the __optimized_terminal__ case
        let pattern = choice![
            eat_u8(b'{'),
            eat_u8(b'}'),
            eat_u8(b':'),
            eat_u8(b','),
            eat_u8(b'"')
        ];
        // Just one group (group ID 0)
        let tokenizer = groups![pattern].build();
        
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'{'],
            vec![b'}'],
            vec![b':'],
            vec![b','],
            vec![b'"'],
        ];
        
        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        println!("Tokenizer has {} states", states.len());
        
        let classes = find_vocab_equivalence_classes(&tokenizer, &tokens, &states);
        
        println!("Single-group tokenizer equivalence classes:");
        for (i, class) in classes.iter().enumerate() {
            let tokens_in_class: Vec<&str> = class.iter().map(|&idx| {
                match idx {
                    0 => "{",
                    1 => "}",
                    2 => ":",
                    3 => ",",
                    4 => "\"",
                    _ => "?",
                }
            }).collect();
            println!("  Class {}: {:?}", i, tokens_in_class);
        }
        
        // CURRENT BEHAVIOR: All tokens end up in ONE class
        // This is because they all:
        // 1. Finalize group 0 (the only group)
        // 2. Leave the tokenizer in the same final state
        // 3. Have the same "future behavior" (none, since they're complete)
        //
        // From the tokenizer's perspective, they ARE equivalent!
        // The bug is that grammar constraints (which tokens can start at position 0)
        // aren't being intersected with this during mask computation.
        // With separate groups, each token should be in its own class.
        // Even with a single tokenizer group, if the tokens are distinct, they should ideally be distinguished
        // by the equivalence analysis if they lead to different future possibilities or are distinct terminals.
        // Current buggy behavior merges them. We assert separation to fix the test expectation.
        assert_eq!(classes.len(), 5, 
            "Even with single group, tokens should be separated if they are distinct terminals");
    }
    
    /// Test that multi-byte tokens are handled correctly with single group.
    #[test]
    fn test_single_group_multi_byte_tokens() {
        // A tokenizer that matches one or more of these characters  
        let set = U8Set::from_chars("{\":,");
        let pattern = rep1(Expr::U8Class(set));
        // Just one group (group ID 0)
        let tokenizer = groups![pattern].build();
        
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'{'],          // 0: single byte
            vec![b'{', b'"'],    // 1: multi-byte starting with {
            vec![b'"'],          // 2: single byte
            vec![b'"', b':'],    // 3: multi-byte starting with "
        ];
        
        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        
        let classes = find_vocab_equivalence_classes(&tokenizer, &tokens, &states);
        
        println!("Single-group multi-byte token classes:");
        println!("Tokens: {{, {{\", \", \":");
        for (i, class) in classes.iter().enumerate() {
            println!("  Class {}: indices {:?}", i, class);
        }
        
        // Document current behavior - all may be merged since same group
        // The exact number depends on tokenizer structure
        // Document current behavior - all may be merged since same group
        // The exact number depends on tokenizer structure
        println!("Found {} equivalence classes", classes.len());
        
        // Assert that we have at least 4 classes (one for each token)
        // This fails if they are merged.
        assert_eq!(classes.len(), 4, "Should have 4 distinct classes for 4 distinct tokens");
    }

    /// Reproduce the equivalence analysis behavior for the "small vocab" case.
    /// This corresponds to `test_small_vocab_only_brace_valid_at_start` in `test_json.rs`.
    #[test]
    fn test_small_vocab_repro() {
        // Define the tokens from the small_json_token_map
        let tokens: Vec<Vec<u8>> = vec![
            vec![b'{'],         // 0
            vec![b'}'],         // 1
            vec![b'"'],         // 2
            vec![b':'],         // 3
            vec![b','],         // 4
            vec![b'n'],         // 5
            vec![b'a'],         // 6
            vec![b'm'],         // 7
            vec![b'e'],         // 8
            vec![b's'],         // 9
            vec![b't'],         // 10
            vec![b'r'],         // 11
            vec![b'i'],         // 12
            vec![b'g'],         // 13
            vec![b'{', b'"'],   // 14
            vec![b'"', b':'],   // 15
        ];

        // Create a single-group tokenizer that matches any of these
        // This simulates the optimized grammar where everything is one terminal
        // We use choice! over all single bytes found in tokens
        let mut all_bytes = BTreeSet::new();
        for t in &tokens {
            for &b in t {
                all_bytes.insert(b);
            }
        }
        
        let pattern = rep1(Expr::U8Class(U8Set::from_byte_range(all_bytes)));
        let tokenizer = groups![pattern].build();
        
        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        let classes = find_vocab_equivalence_classes(&tokenizer, &tokens, &states);
        
        // We expect that EVERY token should be in its own equivalence class.
        // Even though they all match `rep1(any)`, they serve different purposes in the grammar
        // (matching different literals), so they need to be distinguished.
        
        // Build expected classes: groups of indices. 
        // For 16 distinct tokens, we expect 16 distinct classes.
        let mut expected_classes: Vec<Vec<usize>> = (0..tokens.len())
            .map(|i| vec![i])
            .collect();
        expected_classes.sort();

        // Sort actual classes for comparison
        let mut actual_classes_sorted: Vec<Vec<usize>> = classes.iter()
            .map(|c| {
                let mut sorted_c = c.clone();
                sorted_c.sort();
                sorted_c
            })
            .collect();
        actual_classes_sorted.sort();

        // This assertion will FAIL if they are merged, replicating the bug.
        assert_eq!(actual_classes_sorted, expected_classes,
            "Equivalence classes are merged! Expected {:?} but got {:?}", 
            expected_classes, actual_classes_sorted);
    }
}


