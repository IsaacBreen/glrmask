#[cfg(test)]
mod tests {
    use crate::constraint::{GrammarConstraint, GrammarConstraintState};
    use crate::finite_automata::{eat_u8, rep, Expr as RegexExpr};
    use crate::interface::{choice, sequence, regex, literal, CompiledGrammar, IncrementalParser, GrammarExpr};
    use crate::tokenizer::LLMTokenID;
    use crate::datastructures::hybrid_bitset::HybridBitset;
    use bimap::BiBTreeMap; // Add this line

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
        let digit_regex = crate::interface::tokenizer_combinators::eat_u8_range_fast(b'0', b'9');
        let number_expr = regex(crate::interface::tokenizer_combinators::repeat1_fast(digit_regex)); // This is a GrammarExpr
        let plus_expr = regex(crate::interface::tokenizer_combinators::eat_u8_fast(b'+'));   // This is a GrammarExpr

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
            // println!("Current mask: {:?}", state.get_mask().iter_bits().collect::<Vec<_>>());
            assert!(
                state.get_mask().contains(token_id.0),
                "Token ID {} not in mask. Mask: {:?}", token_id.0, state.get_mask().iter_bits().collect::<Vec<_>>()
            );
            // println!("Committing token ID: {}", token_id.0);
            state.commit(token_id);
            state.step_with_all_llm_tokens();
        }

        println!("Getting final mask...");
        let final_mask = state.get_mask();
        // println!("Final mask: {:?}", final_mask.iter_bits().collect::<Vec<_>>());


        // After "123+456+", the grammar expects NUM (digits '0'-'9')
        for i in 0..=9 { // LLM Token IDs for '0' through '9'
            assert!(
                final_mask.contains(i),
                "Expected digit '{}' (LLM Token ID {}) to be allowed. Mask: {:?}",
                (b'0' + i as u8) as char, i, final_mask.iter_bits().collect::<Vec<_>>()
            );
        }
        assert!(
            !final_mask.contains(plus_token_id), // LLM Token ID for '+'
            "Expected '+' (LLM Token ID {}) to be disallowed. Mask: {:?}",
            plus_token_id, final_mask.iter_bits().collect::<Vec<_>>()
        );
        // EOF is not explicitly checked here unless it's part of the grammar logic for completion.
        // The current grammar S -> NUM + NUM + NUM does not explicitly end.
        // The input "123+456+" means we are expecting the third NUM.
        // So EOF should NOT be allowed yet.
        if final_mask.len() > eof_llm_token_id { // Check if eof_llm_token_id is a valid index
             assert!(!final_mask.contains(eof_llm_token_id), "Expected EOF (ID {}) to be disallowed at this stage", eof_llm_token_id);
        }

        println!("Final mask check passed.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt() {
        // Helper to create GrammarExpr::Literal from string
        let lit = |s: &str| crate::interface::literal(s.as_bytes().to_vec());

        // Define GrammarExprs for non-terminals
        let expr_A = choice(vec![
            lit("a"),
            lit("the"),
            lit("apple"),
            lit("banana"),
            lit("person"),
        ]);

        let expr_IGNORE = lit(" ");

        let expr_B = choice(vec![
            lit("eats"),
            lit("likes"),
            lit("is"),
            lit("tasty"),
            lit("red"),
            lit("happy"),
            lit("."),
            lit("and"),
        ]);

        let expr_start = sequence(vec![
            crate::interface::r#ref("A"),
            crate::interface::r#ref("IGNORE"),
            crate::interface::r#ref("B"),
        ]);

        // Grammar definition for CompiledGrammar
        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("IGNORE".to_string(), expr_IGNORE),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let compiled_grammar = CompiledGrammar::from_exprs(grammar_exprs).expect("Failed to compile sentence grammar");
        println!("{}", compiled_grammar); // For debugging grammar structure

        // Setup LLMTokenMap
        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut next_llm_id_val = 0;

        // Helper closure to add tokens to the map and return their ID
        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            // Ensure no duplicate token strings mapping to different IDs for this test
            if let Some(existing_id) = llm_token_map.get_by_left(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        // Tokens for rule A
        let tok_a = add_token("a");
        let tok_the = add_token("the");
        let tok_apple = add_token("apple");
        let tok_banana = add_token("banana");
        let tok_person = add_token("person");

        // Token for rule IGNORE
        let tok_space = add_token(" ");

        // Tokens for rule B
        let tok_eats = add_token("eats");
        let tok_likes = add_token("likes");
        let tok_is = add_token("is");
        let tok_tasty = add_token("tasty");
        let tok_red = add_token("red");
        let tok_happy = add_token("happy");
        let tok_dot = add_token(".");
        let tok_and = add_token("and");

        let tok_e = add_token("e");
        let tok_eth = add_token("eth");

        // Determine max_original_llm_token_id for GrammarConstraint
        // If next_llm_id_val is N, actual IDs are 0 to N-1.
        let max_original_llm_token_id = if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };

        // Define a conceptual EOF token ID (not in llm_token_map for precomputation)
        let eof_llm_token_id = LLMTokenID(next_llm_id_val);


        // Helper to create expected HybridBitset mask
        let ids_to_mask = |ids: &[LLMTokenID]| -> HybridBitset {
            let mut bs = HybridBitset::new();
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id.0, // Pass the usize value for the old eof_llm_token_id param
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens();

        // 1. Initial mask: Expect tokens for rule A
        let mut expected_A_tokens = vec![tok_a, tok_the, tok_apple, tok_banana, tok_person];
        let mut current_mask = state.get_mask();
        assert_eq!(current_mask, ids_to_mask(&expected_A_tokens), "Initial mask should allow tokens for A");

        // 2. Commit "apple" (tok_apple)
        state.commit(tok_apple);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        let expected_IGNORE_tokens = vec![tok_space];
        assert_eq!(current_mask, ids_to_mask(&expected_IGNORE_tokens), "Mask after 'apple' should allow token for IGNORE (' ')");

        // 3. Commit " " (tok_space)
        state.commit(tok_space);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        let mut expected_B_tokens = vec![tok_a, tok_eats, tok_likes, tok_is, tok_tasty, tok_red, tok_happy, tok_dot, tok_and, tok_e];
        assert_eq!(current_mask, ids_to_mask(&expected_B_tokens), "Mask after 'apple ' should allow tokens for B");

        // 4. Commit "eats" (tok_eats)
        state.commit(tok_eats);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        // After "apple eats", the rule "start -> A IGNORE B" is complete.
        // The augmented rule "start' -> start" is also complete.
        // So, we expect EOF to be allowed.
        let mut expected_eof_mask = HybridBitset::new();
        assert_eq!(current_mask, expected_eof_mask);

        println!("Sentence grammar test completed successfully.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt_simplified() {
        // Helper to create GrammarExpr::Literal from string
        let lit = |s: &str| crate::interface::literal(s.as_bytes().to_vec());

        // Define GrammarExprs for non-terminals
        let expr_A = lit("ab");
        let expr_B = lit("bc");

        let expr_start = sequence(vec![
            crate::interface::r#ref("A"),
            crate::interface::r#ref("B"),
        ]);

        // Grammar definition for CompiledGrammar
        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let compiled_grammar = CompiledGrammar::from_exprs(grammar_exprs).expect("Failed to compile sentence grammar");
        println!("{}", compiled_grammar); // For debugging grammar structure

        // Setup LLMTokenMap
        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut next_llm_id_val = 0;

        // Helper closure to add tokens to the map and return their ID
        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            // Ensure no duplicate token strings mapping to different IDs for this test
            if let Some(existing_id) = llm_token_map.get_by_left(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        // Tokens
        let tok_b = add_token("b");

        // Determine max_original_llm_token_id for GrammarConstraint
        // If next_llm_id_val is N, actual IDs are 0 to N-1.
        let max_original_llm_token_id = if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };

        // Define a conceptual EOF token ID (not in llm_token_map for precomputation)
        let eof_llm_token_id = LLMTokenID(next_llm_id_val);


        // Helper to create expected HybridBitset mask
        let ids_to_mask = |ids: &[LLMTokenID]| -> HybridBitset {
            let mut bs = HybridBitset::new();
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id.0, // Pass the usize value for the old eof_llm_token_id param
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens();

        // 1. Initial mask: Expect tokens for rule A
        let mut expected_A_tokens = vec![];
        let mut current_mask = state.get_mask();
        assert_eq!(current_mask, ids_to_mask(&expected_A_tokens), "Initial mask should allow tokens for A");
    }

    #[test]
    fn test_python_reported_bug_def_rep_space_f() {
        // 1. Define Grammar: start -> "<space>* "f"
        let start_expr = sequence(vec![
            regex(rep(eat_u8(b' '))), // Represents one or more spaces matched by the tokenizer
            literal(b"f".to_vec()),
        ]);
        let exprs = vec![("start".to_string(), start_expr)];
        let compiled_grammar = CompiledGrammar::from_exprs(exprs)
            .expect("Failed to compile grammar for bug replication test");
        println!("Compiled Grammar: {}", compiled_grammar);

        // 2. Define LLM Token Map based on the Python example's problematic vocabulary
        let mut llm_token_map = BiBTreeMap::new();
        let tok_space_id = LLMTokenID(0);    // Token for a single space " "
        let tok_f_space_id = LLMTokenID(1); // Token for " f"

        llm_token_map.insert(b" ".to_vec(), tok_space_id);
        llm_token_map.insert(b" f".to_vec(), tok_f_space_id);

        // max_original_llm_token_id is the highest ID value present in the map.
        let max_original_llm_token_id = 2;
        // _eof_llm_token_id parameter for from_compiled_grammar is a placeholder in current setup.
        // Python binding passes 0.
        let dummy_eof_placeholder = 0;

        // 3. Create GrammarConstraint and State
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            dummy_eof_placeholder,
            max_original_llm_token_id,
        );
        let mut state = grammar_constraint.init();
        // In the Python example, step_with_all_llm_tokens() is called after init
        // and after each commit. We replicate that behavior here.
        state.step_with_all_llm_tokens();

        // 4. Initial Mask Check - This is where the bug is expected
        // Allowed LLM tokens should be:
        // - " " (tok_space_id): Consumes one space from <space>*. Remaining: <space>* "f"
        // - " f" (tok_f_space_id): Consumes the space from <space>* and "f" from literal("f").
        // The bug reported is that " f" is NOT in the mask.
        let initial_mask = state.get_mask();

        // This assertion is expected to FAIL, revealing the bug.
        assert!(
            initial_mask.contains(tok_f_space_id.0),
            "BUG REPLICATION: Initial mask should contain ' f' (ID {}), but it does not. Mask: {:?}",
            tok_f_space_id.0,
            &initial_mask
        );

        // For completeness, also check for " " which should be present.
        // This assertion should ideally pass if the logic for single space tokens is correct.
        assert!(
            initial_mask.contains(tok_space_id.0),
            "Initial mask should contain ' ' (ID {}). Mask: {:?}",
            tok_space_id.0,
            &initial_mask
        );
    }
}
