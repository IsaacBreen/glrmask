#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use crate::constraint::GrammarConstraint;
    use crate::datastructures::bitset::Bitset;
    use crate::datastructures::hybrid_bitset::RangeSet;
    use crate::finite_automata::{
        eat_u8, eat_u8_seq, greedy_group, groups, Expr as RegexExpr, QuantifierType,
    };
    use crate::glr::grammar::{
        regex_name, NonTerminal as NT, Production as Prod, Symbol as Sym, Terminal,
    };
    use crate::interface::tokenizer_combinators::{
        eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast, repeat1_fast,
    };
    use crate::interface::{
        choice, r#ref, repeat, sequence, CompiledGrammar, GrammarDefinition, IncrementalParser,
    };
    use crate::dfa_u8::{LLMTokenID, LLMTokenMap, Tokenizer};
    use crate::{choice_fast, groups, seq_fast};
    use bimap::BiBTreeMap;
    use bitvec::prelude::*;
    use std::collections::{BTreeSet, HashSet};
    use crate::glr::table::generate_glr_parser;

    fn bitvec_with_capacity_and_values(capacity: usize, values: Vec<usize>) -> RangeSet {
        let mut bitvec = BitVec::new();
        bitvec.resize(capacity, false);
        for value in values {
            if value < capacity {
                bitvec.set(value, true);
            }
        }
        bitvec.into()
    }

    #[test]
    fn test_precompute_for_python_name_token_with_names() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
        let ignore_expr = repeat0_fast(choice_fast!(
            eat_u8_fast(b' '),
            seq_fast!(
                eat_u8_fast(b'#'),
                repeat0_fast(eat_u8_negation_fast(b'\n')),
                eat_u8_fast(b'\n')
            )
        ));
        let digit_expr = eat_u8_range_fast(b'0', b'9');
        let alph_lower_expr = eat_u8_range_fast(b'a', b'z');
        let alph_upper_expr = eat_u8_range_fast(b'A', b'Z');
        let underscore_expr = eat_u8_fast(b'_');

        let name_start_expr =
            choice_fast!(alph_lower_expr.clone(), alph_upper_expr.clone(), underscore_expr.clone());
        let name_middle_expr = choice_fast!(name_start_expr.clone(), digit_expr.clone());
        let name_expr = seq_fast!(
            ignore_expr.clone(),
            name_start_expr.clone(),
            repeat0_fast(name_middle_expr.clone())
        );

        let tokenizer = Tokenizer::new(groups![
            greedy_group(ignore_expr),
            greedy_group(digit_expr),
            greedy_group(alph_lower_expr),
            greedy_group(alph_upper_expr),
            greedy_group(underscore_expr),
            greedy_group(name_start_expr),
            greedy_group(name_middle_expr),
            greedy_group(name_expr)
        ]
        .build());

        let llm_tokens: Vec<Vec<u8>> =
            (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
        let llm_token_map: LLMTokenMap = llm_tokens
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), LLMTokenID(i)))
            .collect();
        let max_llm_token_id = llm_tokens.len();

        let mut regex_name_to_group_id = BiBTreeMap::new();
        regex_name_to_group_id.insert(regex_name(&"ignore"), 0);
        regex_name_to_group_id.insert(regex_name(&"digit"), 1);
        regex_name_to_group_id.insert(regex_name(&"alph_lower"), 2);
        regex_name_to_group_id.insert(regex_name(&"alph_upper"), 3);
        regex_name_to_group_id.insert(regex_name(&"underscore"), 4);
        regex_name_to_group_id.insert(regex_name(&"name_start"), 5);
        regex_name_to_group_id.insert(regex_name(&"name_middle"), 6);
        regex_name_to_group_id.insert(regex_name(&"name"), 7);

        let dummy_productions = vec![Prod {
            lhs: NT("S".to_string()),
            rhs: vec![],
        }];
        let dummy_glr_parser = generate_glr_parser(&dummy_productions, &HashSet::new(), HashSet::new());

        let constraint = GrammarConstraint::new(
            tokenizer,
            dummy_glr_parser,
            llm_token_map,
            regex_name_to_group_id,
            max_llm_token_id,
        );
        println!("Precomputation (implicitly done by GrammarConstraint::new) successful.");
    }

    #[test]
    fn test_incremental_parser_simple() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
        let terminals = vec![
            ("a".to_string(), eat_u8(b'a')),
            ("b".to_string(), eat_u8(b'b')),
            ("c".to_string(), eat_u8(b'c')),
        ];
        let rules = vec![(
            "S".to_string(),
            choice(vec![
                sequence(vec![r#ref("a"), r#ref("b")]),
                sequence(vec![r#ref("a"), r#ref("c")]),
            ]),
        )];
        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).unwrap();
        println!("Grammar: {}", grammar_def);
        let grammar = CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        let mut parser = IncrementalParser::new(&grammar);

        assert!(parser.is_valid());

        parser.feed(b"a");
        println!("After 'a': is_valid={}, state.len()={}", parser.is_valid(), parser.state.len());
        println!("Parser state keys: {:?}", parser.state.keys().collect::<Vec<_>>());
        println!("Initial tokenizer state: {:?}", grammar.tokenizer().initial_state_id());
        
        assert!(parser.is_valid());
        assert_eq!(parser.state.len(), 1, "Expected 1 state after feeding 'a'");
        
        // Check that we can continue parsing from this state
        let parser_after_a = parser.clone();
        
        // Test 'ab' path
        let mut parser_ab = parser_after_a.clone();
        parser_ab.feed(b"b");
        println!("After 'ab': is_valid={}, state.len()={}", parser_ab.is_valid(), parser_ab.state.len());
        assert!(parser_ab.is_valid(), "'ab' should be valid");
        
        // After completing 'ab', we should be at the initial tokenizer state
        // (a complete terminal was matched)
        println!("After 'ab' state keys: {:?}", parser_ab.state.keys().collect::<Vec<_>>());
        assert!(
            parser_ab.state.contains_key(&grammar.tokenizer().initial_state_id()),
            "After complete input 'ab', should be at initial tokenizer state"
        );

        // Test 'ac' path  
        let mut parser_ac = IncrementalParser::new(&grammar);
        parser_ac.feed(b"a");
        parser_ac.feed(b"c");
        assert!(parser_ac.is_valid(), "'ac' should be valid");
        assert!(
            parser_ac.state.contains_key(&grammar.tokenizer().initial_state_id()),
            "After complete input 'ac', should be at initial tokenizer state"
        );

        // Test invalid path
        let mut parser_ad = IncrementalParser::new(&grammar);
        parser_ad.feed(b"a");
        parser_ad.feed(b"d");
        assert!(!parser_ad.is_valid(), "'ad' should be invalid");
    }

    #[test]
    fn test_minimal_python_example_with_compiled_grammar() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
        let terminals = vec![
            (
                "NUMBER".to_string(),
                repeat1_fast(
                    eat_u8_range_fast(b'0', b'9'),
                ),
            ),
            (
                "PLUS".to_string(),
                eat_u8_fast(b'+'),
            ),
        ];

        let rules = vec![(
            "S".to_string(),
            sequence(vec![
                r#ref("NUMBER"),
                r#ref("PLUS"),
                r#ref("NUMBER"),
                r#ref("PLUS"),
                r#ref("NUMBER"),
            ]),
        )];

        println!("Building grammar...");
        let grammar_def = GrammarDefinition::from_exprs(rules, terminals).unwrap();
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));

        let mut llm_token_map = BTreeMap::new();
        let mut llm_tokens_vec: Vec<Vec<u8>> = Vec::new();
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

        let max_llm_token_id = plus_token_id + 1;
        let eof_llm_token_id = max_llm_token_id;

        println!("Creating constraint...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_llm_token_id,
        );
        grammar_constraint.dump_parser_dwa();

        println!("Initializing state...");
        let mut state = grammar_constraint.init();

        let input_token_ids = vec![
            LLMTokenID(1),
            LLMTokenID(2),
            LLMTokenID(3),
            LLMTokenID(10),
            LLMTokenID(4),
            LLMTokenID(5),
            LLMTokenID(6),
            LLMTokenID(10),
        ];

        println!("Committing tokens...");
        for token_id in input_token_ids {
            assert!(
                state.get_mask().contains(token_id.0),
                "Token ID {} not in mask. Mask: {:?}",
                token_id.0,
                state.get_mask().iter_bits().collect::<Vec<_>>()
            );
            state.commit(token_id);
        }

        println!("Getting final mask...");
        let final_mask = state.get_mask();

        for i in 0..=9 {
            assert!(
                final_mask.contains(i),
                "Expected digit '{}' (LLM Token ID {}) to be allowed. Mask: {:?}",
                (b'0' + i as u8) as char,
                i,
                final_mask.iter_bits().collect::<Vec<_>>()
            );
        }
        assert!(
            !final_mask.contains(plus_token_id),
            "Expected '+' (LLM Token ID {}) to be disallowed. Mask: {:?}",
            plus_token_id,
            final_mask.iter_bits().collect::<Vec<_>>()
        );
        if final_mask.len() > eof_llm_token_id {
            assert!(
                !final_mask.contains(eof_llm_token_id),
                "Expected EOF (ID {}) to be disallowed at this stage",
                eof_llm_token_id
            );
        }

        println!("Final mask check passed.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
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

        let expr_A = choice(vec![
            r#ref("a"),
            r#ref("the"),
            r#ref("apple"),
            r#ref("banana"),
            r#ref("person"),
        ]);
        let expr_IGNORE = r#ref(" ");
        let expr_B = choice(vec![
            r#ref("eats"),
            r#ref("likes"),
            r#ref("is"),
            r#ref("tasty"),
            r#ref("red"),
            r#ref("happy"),
            r#ref("."), //
            r#ref("and"),
        ]);

        let expr_start = sequence(vec![
            r#ref("A"),
            r#ref("IGNORE"),
            r#ref("B"),
        ]);

        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("IGNORE".to_string(), expr_IGNORE),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let grammar_def =
            GrammarDefinition::from_exprs(grammar_exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        println!("{}", compiled_grammar);

        let mut llm_token_map = BTreeMap::new();
        let mut next_llm_id_val = 0;

        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            if let Some(existing_id) = llm_token_map.get(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        let tok_a = add_token("a");
        let tok_the = add_token("the");
        let tok_apple = add_token("apple");
        let tok_banana = add_token("banana");
        let tok_person = add_token("person");

        let tok_space = add_token(" ");

        let tok_eats = add_token("eats");
        let tok_likes = add_token("likes");
        let tok_is = add_token("is");
        let tok_tasty = add_token("tasty");
        let tok_red = add_token("red");
        let tok_happy = add_token("happy");
        let tok_dot = add_token(".");
        let tok_and = add_token("and");

        let tok_e = add_token("e");
        let _tok_eth = add_token("eth");

        let max_original_llm_token_id =
            if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };

        let _eof_llm_token_id = LLMTokenID(next_llm_id_val);

        let ids_to_mask = |ids: &[LLMTokenID]| -> Bitset {
            let mut bs = Bitset::zeros();
            for id in ids {
                bs.insert(id.0 as usize);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();

        let expected_A_tokens = vec![tok_a, tok_the, tok_apple, tok_banana, tok_person];
        let mut current_mask = state.get_mask();
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_A_tokens),
            "Initial mask should allow tokens for A"
        );

        state.commit(tok_apple);
        current_mask = state.get_mask();
        let expected_IGNORE_tokens = vec![tok_space];
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_IGNORE_tokens),
            "Mask after 'apple' should allow token for IGNORE (' ')"
        );

        state.commit(tok_space);
        current_mask = state.get_mask();
        let expected_B_tokens = vec![
            tok_a, tok_eats, tok_likes, tok_is, tok_tasty, tok_red, tok_happy, tok_dot, tok_and,
            tok_e,
        ];
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_B_tokens),
            "Mask after 'apple ' should allow tokens for B"
        );

        state.commit(tok_eats);
        current_mask = state.get_mask();
        let expected_eof_mask = Bitset::zeros();
        assert_eq!(current_mask, expected_eof_mask);

        println!("Sentence grammar test completed successfully.");
    }

    #[test]
    fn test_sentence_grammar_from_prompt_minimized() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
        let terminals = vec![
            ("A_T".to_string(), eat_u8_seq(b"ab".to_vec())),
            ("B_T".to_string(), eat_u8_seq(b"bc".to_vec())),
        ];

        let expr_A = r#ref("A_T");
        let expr_B = r#ref("B_T");

        let expr_start = sequence(vec![
            r#ref("A"),
            r#ref("B"),
        ]);

        let grammar_exprs = vec![
            ("start".to_string(), expr_start),
            ("A".to_string(), expr_A),
            ("B".to_string(), expr_B),
        ];

        println!("Building grammar for sentence test...");
        let grammar_def =
            GrammarDefinition::from_exprs(grammar_exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        println!("{}", compiled_grammar);

        let mut llm_token_map = BTreeMap::new();
        let mut next_llm_id_val = 0;

        let mut add_token = |s: &str| {
            let token_bytes = s.as_bytes().to_vec();
            if let Some(existing_id) = llm_token_map.get(&token_bytes) {
                return *existing_id;
            }
            let id = LLMTokenID(next_llm_id_val);
            llm_token_map.insert(token_bytes, id);
            next_llm_id_val += 1;
            id
        };

        let _tok_b = add_token("b");

        let max_original_llm_token_id =
            if next_llm_id_val == 0 { 0 } else { next_llm_id_val - 1 };

        let _eof_llm_token_id = LLMTokenID(next_llm_id_val);

        let ids_to_mask = |ids: &[LLMTokenID]| -> Bitset {
            let mut bs = Bitset::zeros();
            for id in ids {
                bs.insert(id.0 as usize);
            }
            bs
        };

        println!("Creating constraint for sentence test...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_original_llm_token_id,
        );

        println!("Initializing state for sentence test...");
        let mut state = grammar_constraint.init();

        let expected_A_tokens = vec![];
        let current_mask = state.get_mask();
        assert_eq!(
            current_mask,
            ids_to_mask(&expected_A_tokens),
            "Initial mask should allow tokens for A"
        );
    }

    #[test]
    fn test_python_reported_bug_def_rep_space_f() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
        let terminals = vec![
            ("SPACE".to_string(), eat_u8(b' ')),
            ("F".to_string(), eat_u8(b'f')),
        ];
        let start_expr = sequence(vec![
            repeat(r#ref("SPACE")),
            r#ref("F"),
        ]);
        let exprs = vec![("start".to_string(), start_expr)];
        let grammar_def =
            GrammarDefinition::from_exprs(exprs, terminals).expect("Failed to create grammar definition");
        let compiled_grammar =
            CompiledGrammar::from_definition(std::sync::Arc::new(grammar_def));
        println!("Compiled Grammar: {}", compiled_grammar);

        let mut llm_token_map = BTreeMap::new();
        let tok_space_id = LLMTokenID(0);
        let tok_f_space_id = LLMTokenID(1);

        llm_token_map.insert(b" ".to_vec(), tok_space_id);
        llm_token_map.insert(b" f".to_vec(), tok_f_space_id);

        let max_original_llm_token_id = 2;

        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar,
            llm_token_map.clone(),
            max_original_llm_token_id,
        );
        let mut state = grammar_constraint.init();

        let initial_mask = state.get_mask();

        assert!(
            initial_mask.contains(tok_f_space_id.0),
            "BUG REPLICATION: Initial mask should contain ' f' (ID {}), but it does not. Mask: {:?}",
            tok_f_space_id.0,
            &initial_mask
        );

        assert!(
            initial_mask.contains(tok_space_id.0),
            "Initial mask should contain ' ' (ID {}). Mask: {:?}",
            tok_space_id.0,
            &initial_mask
        );
    }

    #[test]
    fn test_nullability_handling_in_from_exprs() {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap();
        // This test verifies the raw grammar structure before optimization,
        // so we use from_exprs_no_optimize to avoid terminal consolidation.
        let terminals = vec![
            (
                "X_OPT".to_string(),
                RegexExpr::Quantifier(Box::new(eat_u8(b'x')), QuantifierType::ZeroOrOne),
            ),
            ("EPS".to_string(), RegexExpr::Epsilon),
            ("Z".to_string(), eat_u8(b'z')),
        ];
        let rules = vec![(
            "Root".to_string(),
            sequence(vec![
                r#ref("X_OPT"),
                r#ref("EPS"),
                r#ref("Z"),
            ]),
        )];

        let grammar_def =
            GrammarDefinition::from_exprs_no_optimize(rules, terminals).expect("Failed to create GrammarDefinition");

        let name_term_x_opt = "X_OPT".to_string();
        let _term_x_opt_gid = *grammar_def
            .regex_name_to_group_id
            .get_by_left(&name_term_x_opt)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find group ID for sometimes-null terminal name: {}",
                    name_term_x_opt
                )
            });

        let name_term_eps = "EPS".to_string();
        let _term_eps_gid = *grammar_def
            .regex_name_to_group_id
            .get_by_left(&name_term_eps)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find group ID for always-null terminal name: {}",
                    name_term_eps
                )
            });

        let name_term_z = "Z".to_string();
        let _term_z_gid = *grammar_def
            .regex_name_to_group_id
            .get_by_left(&name_term_z)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find group ID for never-null terminal name: {}",
                    name_term_z
                )
            });

        let mut nt_optional_term_x_opt_name = "".to_string();
        let mut found_prod_to_terminal = false;
        let mut found_prod_to_epsilon = false;

        for prod in &grammar_def.productions {
            if prod.rhs.len() == 1 {
                if let Sym::Terminal(Terminal::RegexName(t)) = &prod.rhs[0] {
                    if t == &name_term_x_opt {
                        let candidate_nt_name = prod.lhs.0.clone();
                        if grammar_def.productions.iter().any(|p| {
                            p.lhs.0 == candidate_nt_name && p.rhs.is_empty()
                        }) {
                            nt_optional_term_x_opt_name = candidate_nt_name;
                            found_prod_to_terminal = true;
                            break;
                        }
                    }
                }
            }
        }

        if !nt_optional_term_x_opt_name.is_empty() {
            if grammar_def.productions.iter().any(|p| {
                p.lhs.0 == nt_optional_term_x_opt_name && p.rhs.is_empty()
            }) {
                found_prod_to_epsilon = true;
            }
        }

        assert!(
            found_prod_to_terminal,
            "Could not find production NT -> {} for the optional NT",
            name_term_x_opt
        );
        assert!(
            found_prod_to_epsilon,
            "Could not find production {} -> epsilon for the optional NT",
            nt_optional_term_x_opt_name
        );
        assert!(
            !nt_optional_term_x_opt_name.is_empty(),
            "Could not find the generated optional NT for {}",
            name_term_x_opt
        );

        let augmented_start_nt_name =
            grammar_def.productions[grammar_def.start_production_id].lhs.0.clone();

        let expected_prods_set = BTreeSet::from([
            Prod {
                lhs: NT(augmented_start_nt_name),
                rhs: vec![Sym::NonTerminal(NT("Root".to_string()))],
            },
            Prod {
                lhs: NT("Root".to_string()),
                rhs: vec![
                    Sym::NonTerminal(NT(nt_optional_term_x_opt_name.clone())),
                    Sym::Terminal(regex_name(&name_term_z)),
                ],
            },
            Prod {
                lhs: NT(nt_optional_term_x_opt_name.clone()),
                rhs: vec![Sym::Terminal(regex_name(&name_term_x_opt))],
            },
            Prod {
                lhs: NT(nt_optional_term_x_opt_name.clone()),
                rhs: vec![],
            },
        ]);

        let actual_prods_set: BTreeSet<_> =
            grammar_def.productions.iter().cloned().collect();

        if expected_prods_set != actual_prods_set {
            println!(
                "Expected productions ({}) vs Actual productions ({})",
                expected_prods_set.len(),
                actual_prods_set.len()
            );
            println!("Expected (not found in actual):");
            for p in expected_prods_set.difference(&actual_prods_set) {
                println!(
                    "  {} -> {}",
                    p.lhs.0,
                    p.rhs
                        .iter()
                        .map(|s| match s {
                            Sym::Terminal(t) => t.to_string(),
                            Sym::NonTerminal(nt) => nt.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            println!("Actual (not found in expected):");
            for p in actual_prods_set.difference(&expected_prods_set) {
                println!(
                    "  {} -> {}",
                    p.lhs.0,
                    p.rhs
                        .iter()
                        .map(|s| match s {
                            Sym::Terminal(t) => t.to_string(),
                            Sym::NonTerminal(nt) => nt.to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }

        assert_eq!(
            actual_prods_set.len(),
            expected_prods_set.len(),
            "Number of productions mismatch"
        );
        assert_eq!(
            actual_prods_set, expected_prods_set,
            "Production sets do not match"
        );

        for prod in &grammar_def.productions {
            for sym in &prod.rhs {
                if let Sym::Terminal(t) = sym {
                    assert_ne!(
                        t,
                        &regex_name(&name_term_eps),
                        "Always-null terminal '{}' should not appear in the RHS of any final production (found in {} -> ...)",
                        name_term_eps,
                        prod.lhs.0
                    );
                }
            }
        }
    }
}
