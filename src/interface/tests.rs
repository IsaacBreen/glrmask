#[cfg(test)]
mod tests {
    use crate::constraint::{GrammarConstraint, GrammarConstraintState};
    use crate::finite_automata::{eat_u8, rep, Expr as RegexExpr, QuantifierType, Expr, eat_u8_seq};
    use crate::interface::{choice, sequence, literal, CompiledGrammar, IncrementalParser, GrammarExpr, GrammarDefinition};
    use crate::tokenizer::LLMTokenID;
    use crate::datastructures::hybrid_bitset::HybridBitset;
    use bimap::BiBTreeMap; // Add this line
    use crate::glr::grammar::{NonTerminal as NT, Production as Prod, Symbol as Sym, Terminal as Term};
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn test_incremental_parser_simple() {
        // Grammar: S -> 'a' 'b' | 'a' 'c'
        let terminals = vec![
            ("a".to_string(), eat_u8(b'a')),
            ("b".to_string(), eat_u8(b'b')),
            ("c".to_string(), eat_u8(b'c')),
        ];
        let rules = vec![(
            "S".to_string(),
            choice(vec![
                sequence(vec![crate::interface::r#ref("a"), crate::interface::r#ref("b")]),
                sequence(vec![crate::interface::r#ref("a"), crate::interface::r#ref("c")]),
            ]),
        )];
        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).unwrap();
        let grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
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
        let terminals = vec![
            ("NUMBER".to_string(), crate::interface::tokenizer_combinators::repeat1_fast(crate::interface::tokenizer_combinators::eat_u8_range_fast(b'0', b'9'))),
            ("PLUS".to_string(), crate::interface::tokenizer_combinators::eat_u8_fast(b'+')),
        ];

        let rules = vec![(
            "S".to_string(),
            sequence(vec![
                crate::interface::r#ref("NUMBER"),
                crate::interface::r#ref("PLUS"),
                crate::interface::r#ref("NUMBER"),
                crate::interface::r#ref("PLUS"),
                crate::interface::r#ref("NUMBER"),
            ]),
        )];

        println!("Building grammar...");
        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).unwrap();
        let compiled_grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
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
            LLMTokenID(eof_llm_token_id),
            max_llm_token_id, // This is the capacity for the bitset (num_tokens including EOF)
        );

        println!("Initializing state...");
        let mut state = grammar_constraint.init();

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
        let terminals = vec![
            ("a".to_string(), eat_u8(b'a')),
            ("the".to_string(), eat_u8_seq(b"the".to_vec())),
            ("apple".to_string(), eat_u8_seq(b"apple".to_vec())),
            ("banana".to_string(), eat_u8_seq(b"banana".to_vec())),
            ("person".to_string(), eat_u8_seq(b"person".to_vec())),
            (" ".to_string(), eat_u8(b' ')),
            ("eats".to_string(), eat_u8_seq(b"eats".to_vec())),
            ("likes".to_string(), eat_u8_seq(b"likes".to_vec())),
            ("is".to_string(), eat_u8_seq(b"is".to_vec())),
            ("tasty".to_string(), eat_u8_seq(b"tasty".to_vec())),
            ("red".to_string(), eat_u8_seq(b"red".to_vec())),
            ("happy".to_string(), eat_u8_seq(b"happy".to_vec())),
            (".".to_string(), eat_u8(b'.')),
            ("and".to_string(), eat_u8_seq(b"and".to_vec())),
        ];

        // Define GrammarExprs for non-terminals
        let expr_A = choice(vec![crate::interface::r#ref("a"), crate::interface::r#ref("the"), crate::interface::r#ref("apple"), crate::interface::r#ref("banana"), crate::interface::r#ref("person")]);
        let expr_IGNORE = crate::interface::r#ref(" ");
        let expr_B = choice(vec![crate::interface::r#ref("eats"), crate::interface::r#ref("likes"), crate::interface::r#ref("is"), crate::interface::r#ref("tasty"), crate::interface::r#ref("red"), crate::interface::r#ref("happy"), crate::interface::r#ref("."), crate::interface::r#ref("and")]);

        let expr_start = sequence(vec![
            crate::interface::r#ref("A"),
            crate::interface::r#ref("IGNORE"),
            crate::interface::r#ref("B"),
        ]);

        // Grammar rules
        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("IGNORE".to_string(), expr_IGNORE),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let grammar_def = GrammarDefinition::from_exprs(grammar_exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
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
            let mut bs = HybridBitset::zeros();
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id, // Pass the usize value for the old eof_llm_token_id param
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();

        // 1. Initial mask: Expect tokens for rule A
        let mut expected_A_tokens = vec![tok_a, tok_the, tok_apple, tok_banana, tok_person];
        let mut current_mask = state.get_mask();
        assert_eq!(current_mask, ids_to_mask(&expected_A_tokens), "Initial mask should allow tokens for A");

        // 2. Commit "apple" (tok_apple)
        state.commit(tok_apple);
        current_mask = state.get_mask();
        let expected_IGNORE_tokens = vec![tok_space];
        assert_eq!(current_mask, ids_to_mask(&expected_IGNORE_tokens), "Mask after 'apple' should allow token for IGNORE (' ')");

        // 3. Commit " " (tok_space)
        state.commit(tok_space);
        current_mask = state.get_mask();
        let mut expected_B_tokens = vec![tok_a, tok_eats, tok_likes, tok_is, tok_tasty, tok_red, tok_happy, tok_dot, tok_and, tok_e];
        assert_eq!(current_mask, ids_to_mask(&expected_B_tokens), "Mask after 'apple ' should allow tokens for B");

        // 4. Commit "eats" (tok_eats)
        state.commit(tok_eats);
        current_mask = state.get_mask();
        // After "apple eats", the rule "start -> A IGNORE B" is complete.
        // The augmented rule "start' -> start" is also complete.
        // So, we expect EOF to be allowed.
        let mut expected_eof_mask = HybridBitset::zeros();
        assert_eq!(current_mask, expected_eof_mask);

        println!("Sentence grammar test completed successfully.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt_simplified() {
        let terminals = vec![
            ("A_T".to_string(), eat_u8_seq(b"ab".to_vec())),
            ("B_T".to_string(), eat_u8_seq(b"bc".to_vec())),
        ];

        let expr_A = crate::interface::r#ref("A_T");
        let expr_B = crate::interface::r#ref("B_T");

        let expr_start = sequence(vec![
            crate::interface::r#ref("A"),
            crate::interface::r#ref("B"),
        ]);

        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let grammar_def = GrammarDefinition::from_exprs(grammar_exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
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
            let mut bs = HybridBitset::zeros();
            for id in ids {
                bs.insert(id.0);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            eof_llm_token_id, // Pass the usize value for the old eof_llm_token_id param
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();

        // 1. Initial mask: Expect tokens for rule A
        let mut expected_A_tokens = vec![];
        let mut current_mask = state.get_mask();
        assert_eq!(current_mask, ids_to_mask(&expected_A_tokens), "Initial mask should allow tokens for A");
    }

    #[test]
    fn test_python_reported_bug_def_rep_space_f() {
        // 1. Define Grammar: start -> "<space>* "f"
        let terminals = vec![
            ("SPACE".to_string(), eat_u8(b' ')),
            ("F".to_string(), eat_u8(b'f')),
        ];
        let start_expr = sequence(vec![
            repeat(crate::interface::r#ref("SPACE")),
            crate::interface::r#ref("F"),
        ]);
        let exprs = vec![("start".to_string(), start_expr)];
        let grammar_def = GrammarDefinition::from_exprs(exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
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
            LLMTokenID(dummy_eof_placeholder),
            max_original_llm_token_id,
        );
        let mut state = grammar_constraint.init();
        // In the Python example, step_with_all_llm_tokens() is called after init
        // and after each commit. We replicate that behavior here.

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

    #[test]
    fn test_nullability_handling_in_from_exprs() {
        // Terminals:
        // - X_OPT: x? (sometimes null)
        // - EPS: epsilon (always null)
        // - Z: "z" (never null)
        let terminals = vec![
            ("X_OPT".to_string(), RegexExpr::Quantifier(Box::new(eat_u8(b'x')), QuantifierType::ZeroOrOne)),
            ("EPS".to_string(), RegexExpr::Epsilon),
            ("Z".to_string(), eat_u8(b'z')),
        ];
        let rules = vec![
            ("Root".to_string(), sequence(vec![
                crate::interface::r#ref("X_OPT"),
                crate::interface::r#ref("EPS"),
                crate::interface::r#ref("Z"),
            ])),
        ];

        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).expect("Failed to create GrammarDefinition");

        // For debugging if the test fails:
        // println!("GrammarDefinition:\n{}", grammar_def);
        // println!("Terminal Name to Group ID: {:?}", grammar_def.terminal_name_to_group_id);
        // println!("Terminal Expr to Group ID: {:?}", grammar_def.terminal_expr_to_group_id);
        // println!("All Productions:");
        // for (idx, prod) in grammar_def.productions.iter().enumerate() {
        //     println!("  {}: {} -> {}", idx, prod.lhs.0, prod.rhs.iter().map(|s| match s {
        //         Sym::Terminal(t) => t.0.clone(),
        //         Sym::NonTerminal(nt) => nt.0.clone(),
        //     }).collect::<Vec<_>>().join(" "));
        // }


        // Dynamically find the names of the relevant terminals
        let term_x_opt_expr = RegexExpr::Quantifier(Box::new(eat_u8(b'x')), QuantifierType::ZeroOrOne);
        let term_eps_expr = RegexExpr::Epsilon;
        let term_z_expr = eat_u8(b'z');

        let term_x_opt_gid = grammar_def.terminal_expr_to_group_id.get_by_left(&term_x_opt_expr)
            .unwrap_or_else(|| panic!("Could not find group ID for sometimes-null terminal expression: {:?}", term_x_opt_expr));
        let name_term_x_opt = grammar_def.terminal_name_to_group_id.get_by_right(term_x_opt_gid)
            .unwrap_or_else(|| panic!("Could not find name for sometimes-null terminal group ID: {}", term_x_opt_gid))
            .clone();

        let term_eps_gid = grammar_def.terminal_expr_to_group_id.get_by_left(&term_eps_expr)
            .unwrap_or_else(|| panic!("Could not find group ID for always-null terminal expression: {:?}", term_eps_expr));
        let name_term_eps = grammar_def.terminal_name_to_group_id.get_by_right(term_eps_gid)
            .unwrap_or_else(|| panic!("Could not find name for always-null terminal group ID: {}", term_eps_gid))
            .clone();

        let term_z_gid = grammar_def.terminal_expr_to_group_id.get_by_left(&term_z_expr)
            .unwrap_or_else(|| panic!("Could not find group ID for never-null terminal expression: {:?}", term_z_expr));
        let name_term_z = grammar_def.terminal_name_to_group_id.get_by_right(term_z_gid)
            .unwrap_or_else(|| panic!("Could not find name for never-null terminal group ID: {}", term_z_gid))
            .clone();

        // Find the generated non-terminal for the optional version of name_term_x_opt
        // This NT should have two productions: NT -> name_term_x_opt and NT -> epsilon
        let mut nt_optional_term_x_opt_name = "".to_string();
        let mut found_prod_to_terminal = false;
        let mut found_prod_to_epsilon = false;

        for prod in &grammar_def.productions {
            // Check for NT -> name_term_x_opt
            if prod.rhs.len() == 1 {
                if let Sym::Terminal(t) = &prod.rhs[0] {
                    if t.0 == name_term_x_opt {
                        // This production is NT -> name_term_x_opt. The LHS is a candidate.
                        let candidate_nt_name = prod.lhs.0.clone();
                        // Verify this candidate also has a production to epsilon
                        if grammar_def.productions.iter().any(|p| p.lhs.0 == candidate_nt_name && p.rhs.is_empty()) {
                            nt_optional_term_x_opt_name = candidate_nt_name;
                            found_prod_to_terminal = true;
                            break; 
                        }
                    }
                }
            }
        }
        
        if !nt_optional_term_x_opt_name.is_empty() {
             if grammar_def.productions.iter().any(|p| p.lhs.0 == nt_optional_term_x_opt_name && p.rhs.is_empty()) {
                found_prod_to_epsilon = true;
            }
        }

        assert!(found_prod_to_terminal, "Could not find production NT -> {} for the optional NT", name_term_x_opt);
        assert!(found_prod_to_epsilon, "Could not find production {} -> epsilon for the optional NT", nt_optional_term_x_opt_name);
        assert!(!nt_optional_term_x_opt_name.is_empty(), "Could not find the generated optional NT for {}", name_term_x_opt);

        // Determine the augmented start symbol's name
        let augmented_start_nt_name = grammar_def.productions[grammar_def.start_production_id].lhs.0.clone();

        // Define the set of expected productions
        let expected_prods_set = BTreeSet::from([
            Prod { lhs: NT(augmented_start_nt_name), rhs: vec![Sym::NonTerminal(NT("Root".to_string()))] },
            Prod { lhs: NT("Root".to_string()), rhs: vec![Sym::NonTerminal(NT(nt_optional_term_x_opt_name.clone())), Sym::Terminal(Term(name_term_z.clone()))] },
            Prod { lhs: NT(nt_optional_term_x_opt_name.clone()), rhs: vec![Sym::Terminal(Term(name_term_x_opt.clone()))] },
            Prod { lhs: NT(nt_optional_term_x_opt_name.clone()), rhs: vec![] }, // Epsilon production
        ]);

        let actual_prods_set: BTreeSet<_> = grammar_def.productions.iter().cloned().collect();
        
        // Assert that the actual productions match the expected ones
        if expected_prods_set != actual_prods_set {
            println!("Expected productions ({}) vs Actual productions ({})", expected_prods_set.len(), actual_prods_set.len());
            println!("Expected (not found in actual):");
            for p in expected_prods_set.difference(&actual_prods_set) {
                 println!("  {} -> {}", p.lhs.0, p.rhs.iter().map(|s| match s { Sym::Terminal(t) => t.0.clone(), Sym::NonTerminal(nt) => nt.0.clone() }).collect::<Vec<_>>().join(" "));
            }
            println!("Actual (not found in expected):");
            for p in actual_prods_set.difference(&expected_prods_set) {
                 println!("  {} -> {}", p.lhs.0, p.rhs.iter().map(|s| match s { Sym::Terminal(t) => t.0.clone(), Sym::NonTerminal(nt) => nt.0.clone() }).collect::<Vec<_>>().join(" "));
            }
        }

        assert_eq!(actual_prods_set.len(), expected_prods_set.len(), "Number of productions mismatch");
        assert_eq!(actual_prods_set, expected_prods_set, "Production sets do not match");

        // Verify that the always-null terminal (name_term_eps) is not present in any RHS of the final productions
        for prod in &grammar_def.productions {
            for sym in &prod.rhs {
                if let Sym::Terminal(t) = sym {
                    assert_ne!(t.0, name_term_eps, "Always-null terminal '{}' should not appear in the RHS of any final production (found in {} -> ...)", name_term_eps, prod.lhs.0);
                }
            }
        }
    }
}
