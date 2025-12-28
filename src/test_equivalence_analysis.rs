// Tests for vocab equivalence analysis
// This module tests that the equivalence analysis correctly groups tokens

use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::{Regex, eat_u8, eat_u8_seq, rep, rep1, Expr, QuantifierType};
use crate::equivalence_analysis::compute_combined_equivalence;
use crate::tokenizer::LLMTokenID;
use crate::datastructures::u8set::U8Set;
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

    /// Pathological JSON test case from user.
    /// Pattern:
    /// [\t\n\r ]** "{" [\t\n\r ]** "\"name\"" [\t\n\r ]** ":" [\t\n\r ]** "\"" (("\\" (("u" [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] | ["/\\bfnrt])) | [ !#-[\]-\xff]))* "\"" [\t\n\r ]** [\t\n\r ]** "," [\t\n\r ]** "\"name\"" [\t\n\r ]** ":" [\t\n\r ]** "\"" (("\\" (("u" [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] [0-9A-Fa-f] | ["/\\bfnrt])) | [ !#-[\]-\xff]))* "\""*? [\t\n\r ]** "}" [\t\n\r ]**
    ///
    /// Vocab:
    /// '{' '}' '"' ':' ',' 'n' 'a' 'm' 'e' 's' 't' 'r' 'i' 'g' '{"' '":'
    #[test]
    fn test_pathological_json_equivalence() {
        fn ws() -> Expr {
            rep(Expr::U8Class(U8Set::from_chars("\t\n\r ")))
        }

        fn hex() -> Expr {
            Expr::U8Class(U8Set::from_chars("0123456789abcdefABCDEF"))
        }

        fn string_content_loop() -> Expr {
            // (("\\" (("u" [hex]{4}) | ["/\\bfnrt])) | [ !#-[\]-\xff])*
            let unicode_escape = seq![
                eat_u8(b'u'),
                hex(), hex(), hex(), hex()
            ];
            let simple_escape = Expr::U8Class(U8Set::from_chars("\"/\\bfnrt"));
            
            let escape = seq![
                eat_u8(b'\\'),
                choice![unicode_escape, simple_escape]
            ];
            
            // [ !#-[\]-\xff]  -> All bytes >= 0x20 except '"' (0x22) and '\' (0x5c)
            // 0x20 is ' ', 0x21 is '!', 0x23 is '#'
            let mut normal_chars = U8Set::from_byte_range(0x20..=0xFF);
            normal_chars.remove(b'"');
            normal_chars.remove(b'\\');
            
            let char_choice = choice![
                escape,
                Expr::U8Class(normal_chars)
            ];
            
            rep(char_choice)
        }

        // Construct the regex components
        let part1 = seq![
            ws(),
            eat_u8(b'{'),
            ws(),
            eat_u8_seq(b"\"name\"".to_vec()),
            ws(),
            eat_u8(b':'),
            ws(),
            eat_u8(b'"'),
            string_content_loop(),
            eat_u8(b'"'),
            ws(),
            ws(), // Redundant ws from pattern
            eat_u8(b','),
            ws(),
            eat_u8_seq(b"\"name\"".to_vec()),
            ws(),
            eat_u8(b':'),
            ws(),
            eat_u8(b'"'),
            string_content_loop()
        ];
        
        // Special tail: "\""*? [\t\n\r ]** "}" [\t\n\r ]**
        // User regex has `\""*?`. Interpreting as non-greedy repeat of quote.
        // If engine doesn't support non-greedy, we use rep.
        let tail_quotes = rep(eat_u8(b'"')); 
        // Note: Actual non-greedy support requires ExprGroup with is_non_greedy=true, 
        // but simple `rep` produces Expr::Quantifier which is usually greedy.
        // For correctness verification of *vocab equivalence* structure, the greediness 
        // might not affect the DFA transitions for single bytes, just the matching priority.
        
        let part2 = seq![
            tail_quotes,
            ws(),
            eat_u8(b'}'),
            ws()
        ];

        let pattern = seq![part1, part2];
        
        // Create tokenizer with this single pattern
        let tokenizer = groups![pattern].build();
        
        // Define vocab: '{' '}' '"' ':' ',' 'n' 'a' 'm' 'e' 's' 't' 'r' 'i' 'g' '{"' '":'
        let vocab_strs = vec![
            "{", "}", "\"", ":", ",", "n", "a", "m", "e", "s", "t", "r", "i", "g", "{\"", "\":"
        ];
        let tokens: Vec<Vec<u8>> = vocab_strs.iter().map(|s| s.as_bytes().to_vec()).collect();
        
        let states: Vec<usize> = tokenizer.iter_states().map(|s| s.0).collect();
        println!("Tokenizer has {} states", states.len());
        
        let start = std::time::Instant::now();
        let result = compute_combined_equivalence(&tokenizer, &tokens, &states);
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
