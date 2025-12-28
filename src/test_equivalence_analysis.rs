// Tests for vocab equivalence analysis
// This module tests that the equivalence analysis correctly groups tokens

use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::{Regex, eat_u8, rep1, Expr};
use crate::equivalence_analysis::compute_combined_equivalence;
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
}


