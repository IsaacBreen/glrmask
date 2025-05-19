
#[cfg(test)]
mod tests {
    use crate::constraint::{GrammarConstraint, GrammarConstraintState};
    use crate::finite_automata::eat_u8;
    use crate::interface::tokenizer_combinators::{
        eat_u8_fast, eat_u8_range_fast, repeat1_fast,
    };
    use crate::tokenizer::LLMTokenID;
    // Import new structs and renamed items
    use crate::interface::{choice, sequence, regex, CompiledGrammar, IncrementalParser, GrammarExpr};
    use crate::tokenizer::TokenizerStateID;
    use crate::datastructures::hybrid_bitset::HybridBitset; // For test assertions

    #[test]
    fn test_incremental_parser_simple() {
        // Grammar: S -> 'a' 'b' | 'a' 'c'
        let exprs = vec![(
            "S".to_string(),
            choice(vec![
                sequence(vec![regex(eat_u8(b'a')), regex(eat_u8(b'b'))]),
                sequence(vec![regex(eat_u8(b'a')), regex(eat_u8(b'c'))]),
            ]),
        )];
        let grammar = CompiledGrammar::from_exprs(exprs).unwrap();
        let mut parser = IncrementalParser::new(&grammar);

        assert!(parser.is_valid());

        parser.feed(b"a");
        assert!(parser.is_valid());
        assert_eq!(parser.state.len(), 1, "Expected 1 state after feeding 'a'");
        // After a full token match ('a'), tokenizer should reset.
        // The key in `parser.state` should be the initial tokenizer state ID.
        assert!(parser.state.contains_key(&grammar.tokenizer().initial_state_id()), "Expected tokenizer initial state after 'a'");


        let mut parser_ab = parser.clone();
        parser_ab.feed(b"b");
        assert!(parser_ab.is_valid());

        // Reset and try the other path 'ac'
        let mut parser_ac = IncrementalParser::new(&grammar); // Start fresh for 'ac'
        parser_ac.feed(b"a"); // Feed 'a'
        parser_ac.feed(b"c"); // Then 'c'
        assert!(parser_ac.is_valid());


        // Try invalid sequence 'ad'
        let mut parser_ad = IncrementalParser::new(&grammar); // Start fresh
        parser_ad.feed(b"a");
        parser_ad.feed(b"d"); // 'd' is not 'b' or 'c'
        // dbg!(&parser_ad.state.keys().collect::<Vec<_>>());
        assert!(!parser_ad.is_valid());
    }

    #[test]
    fn test_minimal_python_example_with_compiled_grammar() {
        let digit_regex = eat_u8_range_fast(b'0', b'9');
        let number_expr = regex(repeat1_fast(digit_regex)); // This is a GrammarExpr
        let plus_expr = regex(eat_u8_fast(b'+'));   // This is a GrammarExpr

        let exprs = vec![(
            "S".to_string(),
            sequence(vec![
                number_expr.clone(),
                plus_expr.clone(),
                number_expr.clone(),
                plus_expr.clone(),
                number_expr.clone(),
            ]),
        )];

        println!("Building grammar...");
        let compiled_grammar = CompiledGrammar::from_exprs(exprs).unwrap();
        // compiled_grammar.glr_parser().print(); // GLRParser might be large

        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut llm_tokens_vec: Vec<Vec<u8>> = Vec::new(); // For consistency if needed later
        for i in 0..=9 {
            let digit_byte = b'0' + i;
            let token = vec![digit_byte];
            llm_token_map.insert(token.clone(), LLMTokenID(i as usize));
            llm_tokens_vec.push(token);
        }
        let plus_token = vec![b'+'];
        let plus_token_id = 10usize;
        llm_token_map.insert(plus_token.clone(), LLMTokenID(plus_token_id));
        llm_tokens_vec.push(plus_token);

        let max_llm_token_id = plus_token_id +1; // Max ID is 10, so capacity for bitset is 11 (0-10)
                                                 // If EOF is separate, then max_llm_token_id = 11 (for capacity 12)
        let eof_llm_token_id = max_llm_token_id; // EOF is the next ID after all actual tokens

        println!("Creating constraint...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar, // compiled_grammar is moved
            llm_token_map.clone(),
            eof_llm_token_id,
            max_llm_token_id, // This is the capacity for the bitset (num_tokens including EOF)
        );

        println!("Initializing state...");
        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens();

        let input_token_ids = vec![
            LLMTokenID(1), LLMTokenID(2), LLMTokenID(3), LLMTokenID(10), // "123+"
            LLMTokenID(4), LLMTokenID(5), LLMTokenID(6), LLMTokenID(10), // "456+"
        ];

        println!("Committing tokens...");
        for token_id in input_token_ids {
            // println!("Current mask: {:?}", state.get_mask().iter_ones().collect::<Vec<_>>());
            assert!(
                state.get_mask().contains(token_id.0),
                "Token ID {} not in mask. Mask: {:?}", token_id.0, state.get_mask().iter_ones().collect::<Vec<_>>()
            );
            // println!("Committing token ID: {}", token_id.0);
            state.commit(token_id);
            state.step_with_all_llm_tokens();
        }

        println!("Getting final mask...");
        let final_mask = state.get_mask();
        // println!("Final mask: {:?}", final_mask.iter_ones().collect::<Vec<_>>());


        // After "123+456+", the grammar expects NUM (digits '0'-'9')
        for i in 0..=9 { // LLM Token IDs for '0' through '9'
            assert!(
                final_mask.contains(i),
                "Expected digit '{}' (LLM Token ID {}) to be allowed. Mask: {:?}",
                (b'0' + i as u8) as char, i, final_mask.iter_ones().collect::<Vec<_>>()
            );
        }
        assert!(
            !final_mask.contains(plus_token_id), // LLM Token ID for '+'
            "Expected '+' (LLM Token ID {}) to be disallowed. Mask: {:?}",
            plus_token_id, final_mask.iter_ones().collect::<Vec<_>>()
        );
        // EOF is not explicitly checked here unless it's part of the grammar logic for completion.
        // The current grammar S -> NUM + NUM + NUM does not explicitly end.
        // If the grammar was S -> NUM + NUM + NUM EOF_SYMBOL, then EOF would be expected.
        // For now, we only check allowed continuations based on the grammar.
        // If the sequence is complete according to the grammar, then EOF should be allowed by GrammarConstraint.
        // The grammar S -> NUM + NUM + NUM is complete after the third NUM.
        // The input "123+456+" means we are expecting the third NUM.
        // So EOF should NOT be allowed yet.
        if final_mask.len() > eof_llm_token_id { // Check if eof_llm_token_id is a valid index
             assert!(!final_mask.contains(eof_llm_token_id), "Expected EOF (ID {}) to be disallowed at this stage", eof_llm_token_id);
        }

        println!("Final mask check passed.");
    }
}
