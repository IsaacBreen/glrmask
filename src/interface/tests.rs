#[cfg(test)]
mod tests {
    use crate::constraint::{GrammarConstraint, GrammarConstraintState};
    use crate::finite_automata::eat_u8;
    // Add RegexExpr specific imports if you use its constructors directly for regexes
    use crate::finite_automata::{Expr as RegexAPIExpr, rep as regex_rep, eat_u8 as regex_eat_u8};
    use crate::interface::tokenizer_combinators::{
        eat_u8_fast, eat_u8_range_fast, repeat1_fast,
    };
    use crate::tokenizer::LLMTokenID;
    // Import new structs and renamed items
    use crate::interface::{choice, sequence, literal as grammar_literal, regex as grammar_regex, CompiledGrammar, IncrementalParser, GrammarExpr};
    use crate::tokenizer::TokenizerStateID;
    use crate::datastructures::hybrid_bitset::HybridBitset; // For test assertions
    use bimap::BiBTreeMap; // For llm_token_map

    #[test]
    fn test_incremental_parser_simple() {
        // Grammar: S -> 'a' 'b' | 'a' 'c'
        let exprs = vec![(
            "S".to_string(),
            choice(vec![
                sequence(vec![grammar_regex(eat_u8(b'a')), grammar_regex(eat_u8(b'b'))]),
                sequence(vec![grammar_regex(eat_u8(b'a')), grammar_regex(eat_u8(b'c'))]),
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
        let number_expr = grammar_regex(repeat1_fast(digit_regex)); // This is a GrammarExpr
        let plus_expr = grammar_regex(eat_u8_fast(b'+'));   // This is a GrammarExpr

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

        let max_llm_token_id = plus_token_id; // Max ID is 10 (0-10 means 11 tokens)
        let eof_llm_token_id = max_llm_token_id + 1; // EOF is the next ID after all actual tokens

        println!("Creating constraint...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar, // compiled_grammar is moved
            llm_token_map.clone(),
            eof_llm_token_id,
            max_llm_token_id,
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
                "Token ID {} not in mask. Mask: {:?}", token_id.0, state.get_mask().iter_bools().collect::<Vec<_>>()
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
                (b'0' + i as u8) as char, i, final_mask.iter_bools().collect::<Vec<_>>()
            );
        }
        assert!(
            !final_mask.contains(plus_token_id), // LLM Token ID for '+'
            "Expected '+' (LLM Token ID {}) to be disallowed. Mask: {:?}",
            plus_token_id, final_mask.iter_bools().collect::<Vec<_>>()
        );

        if final_mask.capacity() > eof_llm_token_id { // Check if eof_llm_token_id is a valid index
             assert!(!final_mask.contains(eof_llm_token_id), "Expected EOF (ID {}) to be disallowed at this stage. Mask: {:?}", eof_llm_token_id, final_mask.iter_bools().collect::<Vec<_>>());
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
            // Added tokens that could be prefixes of others to test precomputation robustness
            lit("e"),    // prefix of "eats"
            lit("th"),   // prefix of "the" (if "th" itself is a token)
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
        // println!("{}", compiled_grammar); // For debugging grammar structure

        // Setup LLMTokenMap
        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut next_llm_id_val = 0;

        // Helper closure to add tokens to the map and return their ID
        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
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
        let tok_th = add_token("th"); // Potential prefix token

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
        let tok_e = add_token("e"); // Potential prefix token


        let max_original_llm_token_id = if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };
        let eof_llm_token_id_val = next_llm_id_val;


        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id_val,
            max_original_llm_token_id,
        );

        let internal_token_capacity = grammar_constraint.internal_max_llm_token + 1;

        let ids_to_mask = |ids: &[LLMTokenID], capacity: usize| -> HybridBitset {
            let mut bs = HybridBitset::new_with_capacity(capacity);
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };


        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens();

        // 1. Initial mask: Expect tokens for rule A
        // "th" can be part of "the" or a standalone token if A -> "th" was a rule.
        // Since "th" is an LLM token and "the" is a literal in rule A,
        // precomputation should allow "th" if it can lead to "the".
        // And "the" itself.
        let expected_A_tokens = vec![tok_a, tok_the, tok_apple, tok_banana, tok_person, tok_th];
        let mut current_mask = state.get_mask();
        assert_eq!(current_mask, ids_to_mask(&expected_A_tokens, internal_token_capacity), "Initial mask should allow tokens for A");

        // 2. Commit "apple" (tok_apple)
        state.commit(tok_apple);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        let expected_IGNORE_tokens = vec![tok_space];
        assert_eq!(current_mask, ids_to_mask(&expected_IGNORE_tokens, internal_token_capacity), "Mask after 'apple' should allow token for IGNORE (' ')");

        // 3. Commit " " (tok_space)
        state.commit(tok_space);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        // Rule B tokens: "eats", "likes", "is", "tasty", "red", "happy", ".", "and", "e"
        // "e" is an LLM token and can be a prefix of "eats".
        let expected_B_tokens = vec![tok_eats, tok_likes, tok_is, tok_tasty, tok_red, tok_happy, tok_dot, tok_and, tok_e];
        assert_eq!(current_mask, ids_to_mask(&expected_B_tokens, internal_token_capacity), "Mask after 'apple ' should allow tokens for B");

        // 4. Commit "eats" (tok_eats)
        state.commit(tok_eats);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        let expected_eof_mask = HybridBitset::new_with_capacity(internal_token_capacity); // Expect empty mask (parse complete)
        assert_eq!(current_mask, expected_eof_mask, "Mask after 'apple eats' should be empty (parse complete)");


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
        // println!("{}", compiled_grammar); // For debugging grammar structure

        // Setup LLMTokenMap
        let mut llm_token_map = bimap::BiBTreeMap::new();
        let mut next_llm_id_val = 0;

        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            if let Some(existing_id) = llm_token_map.get_by_left(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        // Tokens
        let tok_ab = add_token("ab"); // Corresponds to rule A
        let tok_bc = add_token("bc"); // Corresponds to rule B
        // No token "b" for this version of the test to keep it minimal to "ab" then "bc"

        let max_original_llm_token_id = if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };
        let eof_llm_token_id_val = next_llm_id_val;


        println!("Creating constraint for sentence test (simplified)...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id_val,
            max_original_llm_token_id,
        );

        let internal_token_capacity = grammar_constraint.internal_max_llm_token + 1;

        let ids_to_mask = |ids: &[LLMTokenID], capacity: usize| -> HybridBitset {
            let mut bs = HybridBitset::new_with_capacity(capacity);
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Initializing state for sentence test (simplified)...");
        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens();

        // 1. Initial mask: Expect tokens for rule A ("ab")
        let expected_A_tokens = vec![tok_ab];
        let mut current_mask = state.get_mask();
        assert_eq!(current_mask, ids_to_mask(&expected_A_tokens, internal_token_capacity), "Initial mask should allow token 'ab' for A");

        // 2. Commit "ab" (tok_ab)
        state.commit(tok_ab);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        let expected_B_tokens = vec![tok_bc]; // Expect "bc" for rule B
        assert_eq!(current_mask, ids_to_mask(&expected_B_tokens, internal_token_capacity), "Mask after 'ab' should allow token 'bc' for B");

        // 3. Commit "bc" (tok_bc)
        state.commit(tok_bc);
        state.step_with_all_llm_tokens();
        current_mask = state.get_mask();
        let expected_eof_mask = HybridBitset::new_with_capacity(internal_token_capacity); // Expect empty mask (parse complete)
        assert_eq!(current_mask, expected_eof_mask, "Mask after 'ab' 'bc' should be empty (parse complete)");

        println!("Simplified sentence grammar test completed successfully.");
    }

    #[test]
    fn test_python_bug_def_space_rep_f() {
        // Grammar: S -> literal(b"def") regex(rep(eat_u8(b' '))) literal(b"f")
        let grammar_exprs = vec![(
            "S".to_string(),
            sequence(vec![
                grammar_literal(b"def".to_vec()),
                grammar_regex(regex_rep(regex_eat_u8(b' '))),
                grammar_literal(b"f".to_vec()),
            ]),
        )];

        let compiled_grammar = CompiledGrammar::from_exprs(grammar_exprs)
            .expect("Failed to compile test grammar");

        let mut llm_token_map = BiBTreeMap::new();
        let tok_def = LLMTokenID(0);
        let tok_space = LLMTokenID(1);
        let tok_f = LLMTokenID(2);
        let tok_space_f = LLMTokenID(3); // " f"

        llm_token_map.insert(b"def".to_vec(), tok_def);
        llm_token_map.insert(b" ".to_vec(), tok_space);
        llm_token_map.insert(b"f".to_vec(), tok_f);
        llm_token_map.insert(b" f".to_vec(), tok_space_f);

        // Max original ID is 3. internal_max_llm_token will be 3. Capacity for bitsets will be 4.
        let max_original_llm_token_id = 3;
        let eof_llm_token_id_val = 4; // Conceptual EOF ID

        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id_val,
            max_original_llm_token_id,
        );

        let internal_token_capacity = grammar_constraint.internal_max_llm_token + 1;

        let mut state = grammar_constraint.init();
        state.step_with_all_llm_tokens();

        // Initial mask: should only allow "def"
        let mut expected_mask = HybridBitset::new_with_capacity(internal_token_capacity);
        expected_mask.insert(tok_def.0);
        assert_eq!(state.get_mask(), expected_mask, "Initial mask should be {{'def'}}");

        // Commit "def"
        state.commit(tok_def);
        state.step_with_all_llm_tokens();

        // Mask after "def":
        // Grammar is: (def) ( <space>* ) (f)
        // After "def", we expect <space>* f
        // This can be:
        //   - " " (space token, matching start of <space>*) -> tok_space
        //   - "f" (f token, matching f after <space>* matched epsilon) -> tok_f
        //   - " f" (space_f token, matching <space>* then f) -> tok_space_f
        let mut expected_mask_after_def = HybridBitset::new_with_capacity(internal_token_capacity);
        expected_mask_after_def.insert(tok_space.0);   // " "
        expected_mask_after_def.insert(tok_f.0);       // "f"
        expected_mask_after_def.insert(tok_space_f.0); // " f"

        let current_mask = state.get_mask();

        // For debugging:
        // println!("Expected mask after def: {:?}", expected_mask_after_def.iter_bools().collect::<Vec<_>>());
        // println!("Actual mask after def:   {:?}", current_mask.iter_bools().collect::<Vec<_>>());
        // let token_map_for_debug: std::collections::HashMap<usize, String> = llm_token_map.iter().map(|(k,v)| (v.0, String::from_utf8_lossy(k).into_owned())).collect();
        // println!("Expected tokens: {:?}", expected_mask_after_def.iter().map(|id| token_map_for_debug.get(&id).unwrap_or(&"UNKNOWN".to_string()).clone()).collect::<Vec<_>>());
        // println!("Actual tokens:   {:?}", current_mask.iter().map(|id| token_map_for_debug.get(&id).unwrap_or(&"UNKNOWN".to_string()).clone()).collect::<Vec<_>>());


        assert_eq!(current_mask, expected_mask_after_def, "Mask after 'def' is incorrect. Expected ' ', 'f', ' f'.");
    }
}
