use crate::interface::IncrementalParser;

#[cfg(test)]
mod tests {
    use crate::constraint::{GrammarConstraint, GrammarConstraintState};
    use crate::finite_automata::{eat_u8};
    use crate::interface::tokenizer_combinators::{eat_u8_fast, eat_u8_range_fast, repeat1_fast};
    use crate::tokenizer::LLMTokenID;
    use crate::interface::{choice, sequence, regex, Grammar, IncrementalParser};
    use crate::tokenizer::TokenizerStateID; // Import necessary types

    #[test]
    fn test_incremental_parser_simple() {
        // Grammar: S -> 'a' 'b' | 'a' 'c'
        let exprs = vec![
            (
                "S".to_string(),
                choice(vec![
                    sequence(vec![regex(eat_u8(b'a')), regex(eat_u8(b'b'))]),
                    sequence(vec![regex(eat_u8(b'a')), regex(eat_u8(b'c'))]),
                ]),
            ),
        ];
        let grammar = Grammar::from_exprs(exprs);
        let mut parser = IncrementalParser::new(&grammar);

        assert!(parser.is_valid()); // Initial state is valid

        parser.feed(b"a");
        assert!(parser.is_valid()); // After 'a', still valid (expecting 'b' or 'c')
        // Check internal state (optional): should have one GLR state in tokenizer state 0
        // The tokenizer state after matching 'a' should reset to 0.
        assert_eq!(parser.state.len(), 1, "Expected 1 state after feeding 'a'");
        assert!(parser.state.contains_key(&TokenizerStateID(0)), "Expected tokenizer state 0 after 'a'");

        parser.feed(b"b");
        assert!(parser.is_valid()); // After 'ab', it's a valid complete parse

        // Reset and try the other path
        parser = IncrementalParser::new(&grammar);
        parser.feed(b"ac");
        assert!(parser.is_valid()); // After 'ac', also valid

        // Try invalid sequence
        parser = IncrementalParser::new(&grammar);
        parser.feed(b"ad");
        dbg!(&parser.state.keys().collect::<Vec<_>>());
        assert!(!parser.is_valid()); // After 'ad', invalid
    }

    #[test]
    fn test_minimal_python_example() {
        // Grammar: S -> NUM '+' NUM '+' NUM
        //          NUM -> digit+
        let digit_regex = eat_u8_range_fast(b'0', b'9');
        let number_regex = repeat1_fast(digit_regex);
        let plus_regex = eat_u8_fast(b'+');

        let exprs = vec![
            (
                "S".to_string(), // Start rule implicitly added by from_exprs
                sequence(vec![
                    regex(number_regex.clone()), // Represent NUM directly for simplicity here
                    regex(plus_regex.clone()),
                    regex(number_regex.clone()),
                    regex(plus_regex.clone()),
                    regex(number_regex.clone()),
                ]),
            ),
        ];

        println!("Building grammar...");
        let grammar = Grammar::from_exprs(exprs);
        // grammar.glr_parser().print(); // Optional: Print parser table

        // LLM Tokens: '0', '1', ..., '9', '+'
        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut llm_tokens: Vec<Vec<u8>> = Vec::new();
        for i in 0..=9 {
            let digit_byte = b'0' + i;
            let token = vec![digit_byte];
            llm_token_map.insert(token.clone(), LLMTokenID(i as usize));
            llm_tokens.push(token);
        }
        let plus_token = vec![b'+'];
        let plus_token_id = 10usize;
        llm_token_map.insert(plus_token.clone(), LLMTokenID(plus_token_id));
        llm_tokens.push(plus_token);

        let max_llm_token_id = plus_token_id;
        let eof_llm_token_id = max_llm_token_id + 1; // Dummy EOF ID

        println!("Creating constraint...");
        let grammar_constraint = GrammarConstraint::from_grammar(
            grammar,
            llm_token_map.clone(),
            eof_llm_token_id,
            max_llm_token_id,
        );

        println!("Initializing state...");
        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens(); // Initial step

        // Input: "123+456+" -> Tokens: 1, 2, 3, 10, 4, 5, 6, 10
        let input_token_ids = vec![
            LLMTokenID(1), LLMTokenID(2), LLMTokenID(3), LLMTokenID(10), // "123+"
            LLMTokenID(4), LLMTokenID(5), LLMTokenID(6), LLMTokenID(10), // "456+"
        ];

        println!("Committing tokens...");
        for token_id in input_token_ids {
            println!("Ensuring token ID {} is in mask...", token_id.0);
            assert!(state.get_mask()[token_id.0], "Token ID {} not in mask", token_id.0);
            println!("Committing token ID: {}", token_id.0);
            state.commit(token_id);
            state.step_with_all_llm_tokens(); // Step after commit
        }

        println!("Getting final mask...");
        let final_mask = state.get_mask();

        // After "123+456+", the grammar expects NUM (digits '0'-'9')
        for i in 0..=9 {
            assert!(final_mask[i], "Expected digit '{}' (ID {}) to be allowed", (b'0' + i as u8) as char, i);
        }
        assert!(!final_mask[plus_token_id], "Expected '+' (ID {}) to be disallowed", plus_token_id);
        if final_mask.len() > eof_llm_token_id {
             assert!(!final_mask[eof_llm_token_id], "Expected EOF (ID {}) to be disallowed", eof_llm_token_id);
        }
        println!("Final mask check passed.");
    }
}
