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
    fn test_single_group_tokenizer_merges_tokens_into_one_class() {
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
        assert_eq!(classes.len(), 1, 
            "With single group, all tokens currently get merged into one class");
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
        println!("Found {} equivalence classes", classes.len());
    }
}

