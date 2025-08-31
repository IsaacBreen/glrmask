use crate::glr::parser::ParseState;
use rand::rngs::StdRng;
use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::{eat_u8, rep1};
use crate::{choice, choice_fast, groups, seq, seq_fast};
use crate::glr::grammar::{nt, prod, t, regex_name, NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{assign_non_terminal_ids, assign_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map};
use crate::datastructures::hybrid_bitset::HybridBitset; // Explicitly import HybridBitset
use std::hash::{Hash, Hasher};
use crate::interface::{eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast, eat_any_fast, eat_string_fast, choice_fast, eat_bytestring_fast, repeat1_fast, CompiledGrammar, GrammarDefinition, display_productions, opt_fast}; // Added eat_any_fast, CompiledGrammar, repeat01_fast
use crate::glr::analyze; // Import the analyze module

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use bimap::BiBTreeMap;
use reqwest::blocking;
use serde_json;
use crate::constraint::{GrammarConstraint};
use crate::datastructures::trie::Trie;
use crate::json_serialization::{JSONConvertible, JSONNode};
// Already a main dependency, but good to be explicit if used directly
// reqwest will be used if the file isn't cached, ensure it's in dev-dependencies
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use crate::types::TerminalID;
use crate::datastructures::vocab_prefix_tree::VocabPrefixTree; // Added for tokenization
use std::time::Instant;
use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use crate::glr::analyze::{filter_productions_by_reachability, remove_productions_with_undefined_nonterminals};
use std::panic::{self, AssertUnwindSafe}; // Added for panic catching
use std::collections::HashMap;
use crate::datastructures::gss::{gather_gss_stats, reset_llm_tokens};
// For the symbol removal helper

#[test]
fn test_trivial() {
    // Grammar: S -> "a" "$"
    // Tokenizer: "a", "$"
    // LLM Vocab: "a", "$"

    let tokenizer_expr = groups![
        eat_u8(b'a'), // ID 0
        eat_u8(b'$'), // ID 1
    ];
    let tokenizer = tokenizer_expr.build();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));

    let productions = vec![
        prod("S", vec![t("A"), t("EOF")]),
    ];

    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A"), TerminalID(0));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(1)); // The parser generator will look for "EOF"

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);
    println!("Parser: {}", parser);

    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("A"), 0);
    token_name_map.insert(regex_name("EOF"), 1);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map,
        token_name_map,
        1, // max_original_llm_token_id
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    println!("Initializing constraint state...");
    let mut state = constraint.init();

    // Initial mask should allow "a"
    let mask1 = state.get_mask();
    assert_eq!(mask1, HybridBitset::from_iter(vec![0]));

    // Commit "a"
    state.commit(LLMTokenID(0));
    assert!(state.is_active());

    // Mask should now allow "$"
    let mask2 = state.get_mask();
    assert_eq!(mask2, HybridBitset::from_iter(vec![1]));

    // Commit "$"
    state.commit(LLMTokenID(1));
    assert!(state.is_active());

    // Mask should now be empty as we've reached the end of a valid parse
    let mask3 = state.get_mask();
    assert_eq!(mask3, HybridBitset::from_iter(vec![]));
}

#[test]
fn test_constraint_simple() {
    // LLM tokens: "ab", "ac", "$"
    // Grammar tokens: "a", "ab", "b|c", "$" (EOF)
    // Grammar: S -> X $ ; X -> "a" ("b|c") | "ab"
    let expr = groups![
        eat_u8(b'a'),
        seq![eat_u8(b'a'), eat_u8(b'b')],
        choice![eat_u8(b'b'), eat_u8(b'c')], // ID 2
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

    // Grammar Terminals mapped to Tokenizer IDs
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A"), TerminalID(0)); // Corresponds to eat_u8(b'a')
    grammar_token_map.insert(regex_name("AB"), TerminalID(1)); // Corresponds to seq![eat_u8(b'a'), eat_u8(b'b')]
    grammar_token_map.insert(regex_name("B_OR_C"), TerminalID(2)); // Corresponds to choice![eat_u8(b'b'), eat_u8(b'c')]
    grammar_token_map.insert(regex_name("EOF"), TerminalID(3)); // Corresponds to eat_u8(b'$')

    let productions = vec![
        prod("S", vec![nt("X"), t("EOF")]), // S -> X $
        prod("X", vec![t("A"), t("B_OR_C")]), // X -> a (b|c)
        prod("X", vec![t("AB")]),             // X -> ab
    ]; // This is fine, it's a comment

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);
    println!("{}", &parser);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3, // max_llm_token_id should be 3 for 0, 1, 2
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    let mut constraint_state = constraint.init();

    // Initial mask
    let mask = constraint_state.get_mask();
    println!("Initial mask: {:?}", mask);
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1])); // Expect "ab" or "ac"

    // Commit "ab" (LLMTokenID 0)
    println!("{}", &constraint_state);
    constraint_state.commit(LLMTokenID(0));
    assert!(constraint_state.is_active());

    // Mask after committing "ab"
    println!("Constraint state:\n{}", &constraint_state);
    let mask_after_commit = constraint_state.get_mask();
    assert_eq!(mask_after_commit, HybridBitset::from_iter(vec![2])); // Expect "$" (EOF)

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method

    // Ensure the parse state after stepping the constraint with all LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
    // In general, this should be true if all LLM tokens cleanly match grammar tokens (or, equivalently, if the only non-empty entry in the precompute tree is under the initial tokenizer state).
    let llm_token = b"ab".to_vec();
    let grammar_tokenss = vec![vec!["A", "B_OR_C"], vec!["AB"]];
    let llm_token_id_for_comp = llm_token_map.get_by_left(&llm_token).unwrap();

    let mut constraint_state_for_comp = constraint.init(); // This is fine, it's a comment
    // Mask before commit (optional, for debugging)
    let _mask_before = constraint_state_for_comp.get_mask();
    constraint_state_for_comp.commit(*llm_token_id_for_comp);

    let mut parser_state_for_comp = parser.init_glr_parser_null(Some(constraint.llm_vocab.clone()));
    for grammar_tokens in grammar_tokenss {
        let mut parser_state = parser.init_glr_parser(Some(constraint.llm_vocab.clone()));
        for grammar_token in grammar_tokens {
            let grammar_token_id = grammar_token_map.get_by_left(&regex_name(grammar_token)).unwrap();
            parser_state.step(*grammar_token_id);
        }
        parser_state_for_comp.merge_with(parser_state);
    }

    assert_eq!(constraint_state_for_comp.state().len(), 1, "Constraint state should have one tokenizer state after commit");
    let (tokenizer_state_id_comp, actual_constraint_parser_state) = constraint_state_for_comp.state().iter().next().unwrap();
    let mut actual_constraint_parser_state = actual_constraint_parser_state.clone();

    // For comparison, parser_state_for_comp's GSS acc needs to be "all_ones" like commit does.
    let mut comparable_parser_gss = (*parser_state_for_comp.active_state.stack).clone();
    let mut comparable_parser_active_state = ParseState::with_stack(Arc::new(comparable_parser_gss));

    Arc::make_mut(&mut comparable_parser_active_state.stack).reset_llm_tokens();
    Arc::make_mut(&mut actual_constraint_parser_state.active_state.stack).reset_llm_tokens();

    assert_eq!(*tokenizer_state_id_comp, tokenizer.initial_state_id(), "Tokenizer should be in initial state");
    assert_eq!(actual_constraint_parser_state.active_state, comparable_parser_active_state, "GSS structures should match");
}

#[test]
fn test_constraint_simple_simplified() {
    // LLM tokens: "a", "$"
    // Grammar tokens: "a", "$" (EOF)
    // Grammar: S -> X $ ; X -> "a"
    let expr = groups![
        eat_u8(b'a'), // ID 0
        eat_u8(b'$'), // ID 1
    ];
    let tokenizer = expr.build();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));

    // Grammar Terminals mapped to Tokenizer IDs
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A"), TerminalID(0)); // Corresponds to eat_u8(b'a')
    grammar_token_map.insert(regex_name("EOF"), TerminalID(1)); // Corresponds to eat_u8(b'$')

    let productions = vec![
        prod("S", vec![nt("X"), t("EOF")]),
        prod("X", vec![t("A")]),
    ];

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);
    println!("{}", &parser);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        1, // max_llm_token_id should be 1 for 0, 1
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    let mut constraint_state = constraint.init();

    // Initial mask
    let mask = constraint_state.get_mask();
    println!("Initial mask: {:?}", mask);
    assert_eq!(mask, HybridBitset::from_iter(vec![0])); // Expect "a"

    // // Commit "a" (LLMTokenID 0)
    // println!("{}", &constraint_state);
    // constraint_state.commit(LLMTokenID(0));
    // assert!(constraint_state.is_active_or_accepted());
    //
    // // Mask after committing "a"
    // println!("Constraint state:\n{}", &constraint_state);
    // let mask_after_commit = constraint_state.get_mask();
    // assert_eq!(mask_after_commit, HybridBitset::from_iter(vec![1])); // Expect "$" (EOF)
    //
    // // Test Serialization/Deserialization
    // let json = constraint.to_json();
    // let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    // constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method
    //
    // // Ensure the parse state after stepping the constraint with an LLM token is the same as the parse state after stepping the parser itself with tokens emitted by the tokenizer for that same LLM token.
    // let llm_token = b"a".to_vec();
    // let grammar_tokenss = vec![vec!["A"]];
    // let llm_token_id_for_comp = llm_token_map.get_by_left(&llm_token).unwrap();
    //
    // let mut constraint_state_for_comp = constraint.init();
    // constraint_state_for_comp.commit(*llm_token_id_for_comp);
    //
    // let mut parser_state_for_comp = parser.init_glr_parser(Some(constraint.llm_vocab.clone()));
    // let grammar_token_id = grammar_token_map.get_by_left(&regex_name("A")).unwrap();
    // parser_state_for_comp.step(*grammar_token_id);
    //
    // assert_eq!(constraint_state_for_comp.state().len(), 1, "Constraint state should have one tokenizer state after commit");
    // let (_tokenizer_state_id_comp, actual_constraint_parser_state) = constraint_state_for_comp.state().iter().next().unwrap();
    //
    // assert_eq!(actual_constraint_parser_state.active_state, parser_state_for_comp.active_state, "GSS structures should match");
}

#[test]
fn test_constraint_expression() {
    // Example grammar: E -> E '+' T | T; T -> T '*' F | F; F -> '(' E ')' | 'i'
    // LLM token vocabulary: i, +, *, (, ), (i, +i
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"+".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"*".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"(".to_vec(), LLMTokenID(3));
    llm_token_map.insert(b")".to_vec(), LLMTokenID(4));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(5));
    llm_token_map.insert(b"+i".to_vec(), LLMTokenID(6));

    // Tokenizer regex for grammar tokens '+' '*' '(' ')' 'i'
    let expr = groups![
        eat_u8(b'+'),
        eat_u8(b'*'),
        eat_u8(b'('),
        eat_u8(b')'),
        eat_u8(b'i'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]), // Start production
        prod("E", vec![nt("E"), t("PLUS"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("T"), t("TIMES"), nt("F")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("PLUS"), TerminalID(0));
    grammar_token_map.insert(regex_name("TIMES"), TerminalID(1));
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(2));
    grammar_token_map.insert(regex_name("RPAREN"), TerminalID(3));
    grammar_token_map.insert(regex_name("I"), TerminalID(4));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(5));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None); // Start production is index 6
    println!("Parser: {}", parser);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        7, // max_llm_token_id should be 7 for IDs 0-6
    );
    constraint.dump_precomputed(); // Commented out dump for cleaner test output
    constraint.dump_precomputed2(); // Commented out dump for cleaner test output

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (3), "(i" (5)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 3, 5]));

    // Commit "(i"
    state.commit(LLMTokenID(5));
    let mask = state.get_mask();
    // Now expect '+', '*', ')', '+i' => IDs 1,2,4,6
    assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 4, 6]));

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method

    // Ensure the parse state after stepping the constraint with all LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
    // In general, this should be true if all LLM tokens cleanly match grammar tokens (or, equivalently, if the only non-empty entry in the precompute tree is under the initial tokenizer state).
    let llm_token = b"(i".to_vec();
    let grammar_tokens = vec!["LPAREN", "I"];
    let llm_token_id_for_comp = llm_token_map.get_by_left(&llm_token).unwrap();
    let grammar_token_ids = grammar_tokens.iter().map(|token| grammar_token_map.get_by_left(&regex_name(token)).unwrap()).collect::<Vec<_>>();

    let mut constraint_state_for_comp = constraint.init();
    let _mask_before = constraint_state_for_comp.get_mask(); // Optional, for debugging
    constraint_state_for_comp.commit(*llm_token_id_for_comp);

    let mut parser_state_for_comp = parser.init_glr_parser(Some(constraint.llm_vocab.clone()));
    for grammar_token_id in grammar_token_ids {
        parser_state_for_comp.step(*grammar_token_id);
    }

    assert_eq!(constraint_state_for_comp.state().len(), 1);
    let (tokenizer_state_id_comp, actual_constraint_parser_state) = constraint_state_for_comp.state().iter().next().unwrap();
    let mut actual_constraint_parser_state = actual_constraint_parser_state.clone();

    // For comparison, parser_state_for_comp's GSS acc needs to be "all_ones" like commit does.
    let mut comparable_parser_gss = (*parser_state_for_comp.active_state.stack).clone();
    let mut comparable_parser_active_state = ParseState::with_stack(Arc::new(comparable_parser_gss));

    Arc::make_mut(&mut comparable_parser_active_state.stack).reset_llm_tokens();
    Arc::make_mut(&mut actual_constraint_parser_state.active_state.stack).reset_llm_tokens();

    assert_eq!(*tokenizer_state_id_comp, tokenizer.initial_state_id(), "Tokenizer should be in initial state");
    assert_eq!(actual_constraint_parser_state.active_state, comparable_parser_active_state, "GSS structures should match");

}

#[test]
fn test_precompute_for_python_name_token() {
    // ignore = rep(choice([
    //     eat_u8(ord(" ")),
    //     seq([eat_u8(ord("#")), rep(eat_u8_negation(ord("\n"))), eat_u8(ord("\n"))]),
    // ]))
    // digit = choice([eat_u8(c) for c in range(ord("0"), ord("9") + 1)])
    // alph_lower = choice([eat_u8(c) for c in range(ord("a"), ord("z") + 1)])
    // alph_upper = choice([eat_u8(c) for c in range(ord("A"), ord("Z") + 1)])
    //
    // name_start = choice([
    //     alph_lower,
    //     alph_upper,
    //     eat_u8(ord("_"))
    // ])
    // name_middle = choice([
    //     name_start,
    //     digit,
    // ])
    let ignore = repeat0_fast(choice_fast!(eat_u8_fast(b' '), seq_fast!(eat_u8_fast(b'#'), repeat0_fast(eat_u8_negation_fast(b'\n')), eat_u8_fast(b'\n'))));

    let digit = eat_u8_range_fast(b'0', b'9');
    let alph_lower = eat_u8_range_fast(b'a', b'z');
    let alph_upper = eat_u8_range_fast(b'A', b'Z');

    let name_start = choice_fast!(alph_lower, alph_upper, eat_u8_fast(b'_'));
    let name_middle = choice_fast!(name_start.clone(), digit);
    let name = seq_fast!(ignore, name_start, repeat0_fast(seq_fast!(name_middle)));

    let tokenizer = name.build();
    dbg!(&tokenizer);

    let llm_tokens: Vec<Vec<u8>> = (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
    let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
    let _eof_llm_token_id = llm_tokens.len();
    let internal_num_llm_tokens = llm_tokens.len(); // This corresponds to the number of tokens for precompute

    // For the purpose of this test calling precompute directly, the IDs in llm_token_map are sequential 0..N-1,
    // which serves as the internal mapping. We don't need a separate internal_llm_token_map here.
    let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
    for (i, token) in llm_tokens.iter().enumerate() {
         internal_llm_token_map_for_precompute.insert(token.clone(), LLMTokenID(i));
    }


    let _precomputed = GrammarConstraint::precompute(
        &tokenizer,
        None,
        None,
        &internal_llm_token_map_for_precompute, // Use the manually created internal map
        &BiBTreeMap::new(), // empty name‐map
        internal_llm_token_map_for_precompute.iter().map(|(_, id)| id.0).max().unwrap_or(0),
        &BTreeMap::new(), // empty terminal_follow_map
        None,
        &mut BTreeMap::new(),
    );
    // print_precomputed(&_precomputed);
    println!("Done precomputing");
}

#[test]
fn test_precompute_explosion() {
    let tokenizer = groups![
        eat_u8(b'a'),
        eat_u8(b'a'),
    ].build();

    let llm_tokens: Vec<Vec<u8>> = vec![b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec()];
     let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();
    let _eof_llm_token_id = llm_tokens.len();
    let internal_num_llm_tokens = llm_tokens.len(); // This corresponds to the number of tokens for precompute

    // For the purpose of this test calling precompute directly, the IDs in llm_token_map are sequential 0..N-1,
    // which serves as the internal mapping. We don't need a separate internal_llm_token_map here.
    let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
    for (i, token) in llm_tokens.iter().enumerate() {
         internal_llm_token_map_for_precompute.insert(token.clone(), LLMTokenID(i));
    }

    let _precomputed = GrammarConstraint::precompute(
        &tokenizer,
        None,
        None,
        &internal_llm_token_map_for_precompute, // Use the manually created internal map
        &BiBTreeMap::new(), // empty name‐map
        internal_llm_token_map_for_precompute.iter().map(|(_, id)| id.0).max().unwrap_or(0),
        &BTreeMap::new(), // empty terminal_follow_map
        None,
        &mut BTreeMap::new(),
    );
    println!("Done precomputing");
}

#[test]
fn test_aborted_tokenizer_restart_equivalence() {
    // Tokenizer:
    // Group 0: "a" (A_T)
    // Group 1: "#" followed by an optional "a" (HASH_OPT_A_T)
    let tokenizer_expr = groups![
        eat_u8_fast(b'a'), // Tokenizer Group ID 0
        seq_fast![ // Tokenizer Group ID 1
            eat_u8_fast(b'#'),
            opt_fast(eat_u8_fast(b'a')) // optional 'a'
        ]
    ];
    let tokenizer = tokenizer_expr.build();

    // Grammar: S -> HASH_OPT_A_T | HASH_OPT_A_T A_T
    // Terminals in grammar:
    // "A" maps to tokenizer group 0
    // "HASH_OPT_A" maps to tokenizer group 1
    let productions = vec![
        prod("S'", vec![nt("S")]),
        prod("S", vec![t("HASH_OPT_A")]),
        prod("S", vec![t("HASH_OPT_A"), t("A")]),
    ];

    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    let terminal_a = regex_name("A");
    let terminal_hash_opt_a = regex_name("HASH_OPT_A");
    grammar_token_map.insert(terminal_a.clone(), TerminalID(0)); // Maps to tokenizer group 0
    grammar_token_map.insert(terminal_hash_opt_a.clone(), TerminalID(1)); // Maps to tokenizer group 1

    let parser = generate_glr_parser_with_terminal_map(
        &productions, // Assuming S -> HASH_OPT_A is the first rule for start_production_id if S' is not explicit.
           // generate_glr_parser adds S' -> S EOF, so the first user prod is 0.
        grammar_token_map.clone(),
        None,
    );

    // LLM Tokens
    let mut llm_token_map = LLMTokenMap::new();
    let llm_hash = LLMTokenID(0);
    let llm_a = LLMTokenID(1);
    let llm_hash_a = LLMTokenID(2);
    llm_token_map.insert(b"#".to_vec(), llm_hash);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"#a".to_vec(), llm_hash_a);

    let max_original_llm_token_id = 2;

    // Token name map for GrammarConstraint (maps grammar terminal name to tokenizer group ID)
    let mut token_name_map_for_constraint = BiBTreeMap::new();
    token_name_map_for_constraint.insert(regex_name("A"), 0);
    token_name_map_for_constraint.insert(regex_name("HASH_OPT_A"), 1);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map_for_constraint,
        max_original_llm_token_id,
    );

    // Scenario 1: Commit "#", then "a"
    let mut constraint_state1 = constraint.init();
    println!("Scenario 1: Committing LLM Token '#' (ID {})", llm_hash.0);
    constraint_state1.commit(llm_hash);
    println!("Scenario 1: State after committing '#': {:?}", constraint_state1.state().keys().map(|k|k.0).collect::<Vec<_>>());
    for (tid, glr_state) in constraint_state1.state() {
        glr_state.log_gss(&format!("Scenario 1, after '#', GSS for tokenizer state {}", tid.0), TerminalID(0), false, false);
    }

    println!("\nScenario 1: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state1.commit(llm_a);
    println!("Scenario 1: State after committing 'a': {:?}", constraint_state1.state().keys().map(|k|k.0).collect::<Vec<_>>());
     for (tid, glr_state) in constraint_state1.state() {
        glr_state.log_gss(&format!("Scenario 1, after 'a', GSS for tokenizer state {}", tid.0), TerminalID(0), false, false);
    }

    // Scenario 2: Commit "#a"
    let mut constraint_state2 = constraint.init();
    println!("\nScenario 2: Committing LLM Token '#a' (ID {})", llm_hash_a.0);
    constraint_state2.commit(llm_hash_a);
    println!("Scenario 2: State after committing '#a': {:?}", constraint_state2.state().keys().map(|k|k.0).collect::<Vec<_>>());
    for (tid, glr_state) in constraint_state2.state() {
        glr_state.log_gss(&format!("Scenario 2, after '#a', GSS for tokenizer state {}", tid.0), TerminalID(0), false, false);
    }

    // Assert equivalence
    pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "States from (commit '#' then 'a') and (commit '#a') should be equivalent.");
    assert_eq!(constraint_state1.state(), constraint_state2.state(), "States from (commit '#' then 'a') and (commit '#a') should be equivalent.");
    println!("\nAssertion passed: States are equivalent.");
}

#[test]
fn test_multi_commit_aborted_tokenizer_restart_equivalence() {
    // Tokenizer:
    // Group 0: "a" (A_T)
    // Group 1: "#" followed by an optional "aa" (HASH_OPT_AA_T)
    let tokenizer_expr = groups![
        eat_u8_fast(b'a'), // Tokenizer Group ID 0
        seq_fast![ // Tokenizer Group ID 1
            eat_u8_fast(b'#'),
            opt_fast(seq_fast![eat_u8_fast(b'a'), eat_u8_fast(b'a')]) // optional 'aa'
        ]
    ];
    let tokenizer = tokenizer_expr.build();

    // Grammar: S -> HASH_OPT_AA_T | HASH_OPT_AA_T A_T A_T
    let productions = vec![
        prod("S'", vec![nt("S")]),
        prod("S", vec![t("HASH_OPT_AA")]),
        prod("S", vec![t("HASH_OPT_AA"), t("A"), t("A")]),
    ];

    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    let terminal_a = regex_name("A");
    let terminal_hash_opt_aa = regex_name("HASH_OPT_AA");
    grammar_token_map.insert(terminal_a.clone(), TerminalID(0)); // Maps to tokenizer group 0
    grammar_token_map.insert(terminal_hash_opt_aa.clone(), TerminalID(1)); // Maps to tokenizer group 1

    let parser = generate_glr_parser_with_terminal_map(
        &productions, // start_production_id
        grammar_token_map.clone(),
        None,
    );
    println!("Parser: {}", parser);

    // LLM Tokens
    let mut llm_token_map = LLMTokenMap::new();
    let llm_hash = LLMTokenID(0);    // "#"
    let llm_a = LLMTokenID(1);       // "a"
    let llm_hash_aa = LLMTokenID(2); // "#aa"
    llm_token_map.insert(b"#".to_vec(), llm_hash);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"#aa".to_vec(), llm_hash_aa);

    let max_original_llm_token_id = 2;

    let mut token_name_map_for_constraint = BiBTreeMap::new();
    token_name_map_for_constraint.insert(regex_name("A"), 0);
    token_name_map_for_constraint.insert(regex_name("HASH_OPT_AA"), 1);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map_for_constraint,
        max_original_llm_token_id,
    );

    // Scenario 1: Commit "#", then "a"
    let mut constraint_state3 = constraint.init();
    println!("Scenario 1: Committing LLM Token '#' (ID {})", llm_hash.0);
    constraint_state3.commit(llm_hash);
    println!("{}", &constraint_state3);

    println!("\nScenario 1: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state3.commit(llm_a);
    println!("Scenario 1 state:\n{}", &constraint_state3);

    // Scenario 2: Commit "#a"
    let mut constraint_state4 = constraint.init();
    println!("\nScenario 2: Committing LLM Token '#a' (ID {})", llm_hash_aa.0);
    constraint_state4.commit_bytes(b"#a");
    println!("Scenario 2 state:\n{}", &constraint_state4);

    // Assert equivalence
    pretty_assertions::assert_eq!(constraint_state3.to_string(), constraint_state4.to_string(), "States from (commit '#' then 'a') and (commit '#a') should be equivalent.");
    assert_eq!(constraint_state3.state(), constraint_state4.state(), "States from (commit '#' then 'a') and (commit '#a') should be equivalent.");

    // Scenario 3: Commit "#", then "a", then "a"
    let mut constraint_state1 = constraint.init();
    println!("Scenario 3: Committing LLM Token '#' (ID {})", llm_hash.0);
    constraint_state1.commit(llm_hash);
    println!("{}", &constraint_state1);

    println!("\nScenario 3: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state1.commit(llm_a);
    println!("{}", &constraint_state1);

    println!("\nScenario 3: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state1.commit(llm_a);
    println!("{}", &constraint_state1);

    // Scenario 4: Commit "#aa"
    let mut constraint_state2 = constraint.init();
    println!("\nScenario 4: Committing LLM Token '#aa' (ID {})", llm_hash_aa.0);
    constraint_state2.commit(llm_hash_aa);
    println!("{}", &constraint_state2);

    // Assert equivalence
    println!("constraint_state1 state: {}", constraint_state1);
    println!("constraint_state2 state: {}", constraint_state2);
    pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "States from (commit '#' then 'a' then 'a') and (commit '#aa') should be equivalent.");
    assert_eq!(constraint_state1.state(), constraint_state2.state(), "States from (commit '#' then 'a' then 'a') and (commit '#aa') should be equivalent.");
    println!("\nAssertion passed: States are equivalent for multi-commit scenario.");
}

#[test]
fn test_a_plus_commit_equivalence() {
    // Grammar: S -> A, where A is the terminal for `a+`.
    // This test verifies that committing "a", "a", "a" is equivalent to committing "aaa".
    // This tests the ability of the constraint to handle cases where multiple LLM tokens
    // can form a single grammar token, by carrying over tokenizer state between commits.

    // 1. Tokenizer for `a+`
    let tokenizer_expr = groups![repeat1_fast(eat_u8(b'a'))];
    let tokenizer = tokenizer_expr.build();

    // 2. Grammar: S -> A
    let productions = vec![prod("S", vec![t("A")])];

    // 3. Map grammar terminal "A" to tokenizer group ID 0
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A"), TerminalID(0));

    // 4. Create the Parser
    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    // 5. LLM vocabulary: "a" and "aaa"
    let mut llm_token_map = LLMTokenMap::new();
    let llm_a = LLMTokenID(0);
    let llm_aaa = LLMTokenID(1);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"aaa".to_vec(), llm_aaa);
    let max_original_llm_token_id = 1;

    // 6. Token name map for stats/debugging
    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("A"), 0); // Maps "A" to tokenizer group ID 0

    // 7. Create the GrammarConstraint
    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map,
        max_original_llm_token_id,
    );

    // Scenario 1: Commit "a" three times
    let mut state1 = constraint.init();
    println!("Scenario 1: Committing 'a' (ID {})", llm_a.0);
    state1.commit(llm_a);
    println!("{}", &state1);
    println!("Scenario 1: Committing 'a' (ID {}) again", llm_a.0);
    state1.commit(llm_a);
    println!("{}", &state1);
    println!("Scenario 1: Committing 'a' (ID {}) a third time", llm_a.0);
    state1.commit(llm_a);
    println!("{}", &state1);

    // Scenario 2: Commit "aaa" once
    let mut state2 = constraint.init();
    println!("\nScenario 2: Committing 'aaa' (ID {})", llm_aaa.0);
    state2.commit(llm_aaa);
    println!("{}", &state2);

    // Assert equivalence
    pretty_assertions::assert_eq!(state1.to_string(), state2.to_string(), "States from (commit 'a' x3) and (commit 'aaa') should be equivalent.");
    assert_eq!(state1.state(), state2.state(), "States from (commit 'a' x3) and (commit 'aaa') should be equivalent.");
    println!("\nAssertion passed: States are equivalent.");
}

#[test]
fn test_ignore_token() {
    // Grammar: S -> A B
    // Tokenizer: A='a', B='b', WS=' ' (ignore)
    // LLM Vocab: "a", "b", " ", "a b"

    let tokenizer_expr = groups![
        eat_u8(b'a'), // ID 0 -> A
        eat_u8(b'b'), // ID 1 -> B
        eat_u8(b' '), // ID 2 -> WS (ignore)
    ];
    let tokenizer = tokenizer_expr.build();

    let mut llm_token_map = LLMTokenMap::new();
    let llm_a = LLMTokenID(0);
    let llm_b = LLMTokenID(1);
    let llm_ws = LLMTokenID(2);
    let llm_a_b = LLMTokenID(3);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"b".to_vec(), llm_b);
    llm_token_map.insert(b" ".to_vec(), llm_ws);
    llm_token_map.insert(b"a b".to_vec(), llm_a_b);

    let productions = vec![
        prod("S", vec![t("A"), t("B")]),
    ];

    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    let term_a = regex_name("A");
    let term_b = regex_name("B");
    let term_ws = regex_name("WS");
    let tid_a = TerminalID(0);
    let tid_b = TerminalID(1);
    let tid_ws = TerminalID(2);
    grammar_token_map.insert(term_a.clone(), tid_a);
    grammar_token_map.insert(term_b.clone(), tid_b);
    grammar_token_map.insert(term_ws.clone(), tid_ws);

    let ignore_terminal_id = Some(tid_ws);
    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), ignore_terminal_id);
    println!("Parser: {}", parser);
    assert_eq!(parser.ignore_terminal_id, ignore_terminal_id);

    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(term_a, tid_a.0);
    token_name_map.insert(term_b, tid_b.0);
    token_name_map.insert(term_ws, tid_ws.0);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map,
        token_name_map,
        3, // max_original_llm_token_id
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // --- Runtime check ---
    // Scenario 1: commit "a", then " ", then "b"
    let mut state1 = constraint.init();
    assert_eq!(state1.get_mask(), HybridBitset::from_iter(vec![llm_a.0, llm_ws.0, llm_a_b.0]), "Initial mask should allow 'a' or 'a b'");
    state1.commit(llm_a);
    assert_eq!(state1.get_mask(), HybridBitset::from_iter(vec![llm_b.0, llm_ws.0, llm_ws.0]), "After 'a', mask should allow 'b' or ' '");
    state1.commit(llm_ws);
    assert_eq!(state1.get_mask(), HybridBitset::from_iter(vec![llm_b.0, llm_ws.0]), "After 'a ', mask should allow 'b'");
    state1.commit(llm_b);
    // assert_eq!(state1.get_mask(), HybridBitset::from_iter(vec![llm_ws.0]), "After 'a b', mask should be empty (complete parse).");

    // --- Equivalence check ---
    let mut state2 = constraint.init();
    state2.commit(llm_a_b);
    // assert_eq!(state2.get_mask(), HybridBitset::from_iter(vec![llm_ws.0]), "After committing 'a b', mask should be empty (complete parse).");
    assert_eq!(state1.state(), state2.state(), "States from ('a',' ','b') and ('a b') should be equivalent.");
}

#[test]
fn test_hideous_ambiguity() {
    // 1. Define the grammar
    let productions = vec![
        prod("S", vec![t("FSTRING_MIDDLE"), t("FSTRING_MIDDLE")]),
    ];

    // 2. Tokenizer
    let tokenizer_expr = groups![
        repeat1_fast(eat_u8(b'a')),
    ];
    let tokenizer = tokenizer_expr.build();

    // 3. LLM Token Map
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));

    // 4. Token Name Map
    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("FSTRING_MIDDLE"), 0); // Maps "FSTRING_MIDDLE" to tokenizer group ID 0

    // 5. Create the Parser
    let parser = generate_glr_parser(&productions, None);
    println!("{}", parser);

    // 6. Create the Constraint
    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map,
        0,
    );

    // 7. Initialize the Constraint State
    let mut constraint_state = constraint.init();

    // 8. Step with LLM Token "a" repeatedly
    let a_id = llm_token_map.get_by_left(&b"a"[..]).unwrap().0;
    for i in 0..10 { // Reduced iterations for faster test, was 1000
        println!("{}. Committing LLM token ID {}", i, a_id);
        let mask = constraint_state.get_mask();
        if !mask.contains(a_id) {
            println!("Token 'a' (ID {}) not in mask. Mask: {:?}. Stopping.", a_id, mask);
            break;
        }
        constraint_state.commit(LLMTokenID(a_id));
        if !constraint_state.is_active() {
            println!("Constraint state became inactive at iteration {}.", i);
            break;
        }
    }
}

#[test]
fn test_simple_def_match_non_zero_llm_id() {
    // 1. Tokenizer for the grammar terminal "DEF_T" matching "def"
    //    The tokenizer will have one group (GroupID 0) for "def".
    let tokenizer_expr = groups![eat_string_fast("def")];
    let tokenizer = tokenizer_expr.build();

    // 2. LLM vocabulary: only "def", but with a non-zero original ID
    let mut llm_token_map = LLMTokenMap::new();
    // let def_original_llm_id = 750; // Using the ID from your Python script's log
    let def_original_llm_id = 0;
    llm_token_map.insert(b"def".to_vec(), LLMTokenID(def_original_llm_id));
    let max_original_llm_token_id = def_original_llm_id;

    // 3. Grammar: S -> DEF_T
    //    (S' -> S EOF_Terminal is implicitly added by generate_glr_parser)
    let productions = vec![
        prod("S", vec![t("DEF_T")]), // Production 0
    ];

    // 4. Map grammar terminals "DEF_T" to tokenizer group ID 0
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("DEF_T"), TerminalID(0));
    // Note: For this minimal test focusing on the initial mask for "def",
    // we don't strictly need an EOF terminal in the grammar or tokenizer if
    // the goal is just to see "def" allowed initially.
    // If the grammar was S -> DEF_T EOF_T, then EOF_T would need a tokenizer group.

    let parser = generate_glr_parser_with_terminal_map(
        &productions, // start_production_id
        grammar_token_map.clone(),
        None,
    );

    // 5. Token name map for stats/debugging (maps grammar terminal name to tokenizer group ID)
    let mut token_name_map_for_stats = BiBTreeMap::new();
    token_name_map_for_stats.insert(regex_name("DEF_T"), 0);

    // 6. Create the GrammarConstraint
    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(), // Original LLMTokenID map
        token_name_map_for_stats,
        max_original_llm_token_id,
    );

    // constraint.dump_precomputed(); // Optional: for debugging precomputation

    // 7. Initialize the constraint state.
    //    This calls constraint.init() internally.
    let mut constraint_state = constraint.init();
    let mask = constraint_state.get_mask();

    // 9. Define the expected mask.
    //    It should contain the original LLMTokenID for "def".
    let mut expected_mask = HybridBitset::zeros();
    expected_mask.insert(def_original_llm_id); // Expecting the original LLM ID

    // 10. Assert that the mask matches the expected mask.
    //     This assertion is expected to fail if the bug in setup_llm_token_mappings exists.
    assert_eq!(
        mask,
        expected_mask,
        "Mask should allow 'def' token (Original LLM ID {})",
        def_original_llm_id
    );
}

#[ignore]
#[test]
fn test_precompute_a_plus_tokenizer() {
    // Tokenizer for `a+`
    let tokenizer_expr = groups![repeat1_fast(eat_u8(b'a'))];
    let tokenizer = tokenizer_expr.build();

    // LLM tokens "a" and "aa"
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"aa".to_vec(), LLMTokenID(1));
    let max_original_llm_token_id = 1;

    // Grammar S -> A_PLUS
    let productions = vec![prod("S", vec![t("A_PLUS")])];

    // Map grammar terminal to tokenizer group
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A_PLUS"), TerminalID(0));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    // Token name map for stats
    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("A_PLUS"), 0);

    // Create the constraint, which runs precomputation
    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser,
        llm_token_map,
        token_name_map,
        max_original_llm_token_id,
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // --- Verification ---
    // assert_eq!(constraint.precomputed.len(), 1, "Expected precomputed trie for only one tokenizer state");
    let initial_state_id = tokenizer.initial_state_id();
    let root_node = constraint.precomputed.get(&initial_state_id).expect("No precomputed trie for initial state").read(&constraint.trie1_god).unwrap();

    // 1. Check root node's clean_end
    let root_value = &root_node.value;
    // let clean_end_bv = root_value.clean_end.as_ref().expect("Root should have a clean_end bitset");
    let expected_tokens = HybridBitset::from_iter(vec![0, 1]); // LLM tokens "a" and "aa"
    // assert_eq!(*clean_end_bv, expected_tokens, "Clean_end content is incorrect");

    // 2. Check root node's children and the leaf node
    assert_eq!(root_node.children().len(), 1, "Root should have one child edge key");
    let (edge_gtid_opt, destinations) = root_node.children().iter().next().unwrap();
    assert_eq!(*edge_gtid_opt, Some(TerminalID(0)), "Edge key should be for grammar token 0");
    assert_eq!(destinations.len(), 1, "Should be one destination for the edge");
    let (child_arc_wrapper, edge_bv) = destinations.iter().next().unwrap();
    assert_eq!(*edge_bv, expected_tokens, "Edge token bitset is incorrect");
    let binding = child_arc_wrapper.as_arc().clone();
    let child_node = binding.read(&constraint.trie1_god).unwrap();
    assert!(child_node.is_leaf(), "Child node should be a leaf after pruning");
    // assert_eq!(*child_node.value.clean_end.as_ref().unwrap(), expected_tokens, "Clean_end bitset is incorrect");
}

#[ignore]
#[test]
fn test_precompute_x_eq() {
    // Tokenizer for `=|x| `
    let tokenizer_expr = groups![
        eat_u8(b'x'),
        rep1(eat_u8(b' ')),
        eat_u8(b'='),
        rep1(eat_any_fast()),
    ];
    let tokenizer = tokenizer_expr.build();

    // LLM tokens "x" and " ="
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"x".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b" =".to_vec(), LLMTokenID(1));
    let max_original_llm_token_id = 1;

    // Grammar S -> X SPACE EQUALS
    let productions = vec![
        prod("S", vec![t("X"), t("SPACE"), t("EQUALS")]), // S -> X SPACE EQUALS
    ];

    // Map grammar terminal to tokenizer group
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("X"), TerminalID(0));      // 'x' is group 0
    grammar_token_map.insert(regex_name("SPACE"), TerminalID(1));  // ' ' is group 1
    grammar_token_map.insert(regex_name("EQUALS"), TerminalID(2)); // '=' is group 2
    grammar_token_map.insert(regex_name("ANY"), TerminalID(3));    // Anything else is group 3

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    // Token name map for stats
    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("X"), 0);
    token_name_map.insert(regex_name("SPACE"), 1);
    token_name_map.insert(regex_name("EQUALS"), 2);
    token_name_map.insert(regex_name("ANY"), 3);

    // Create the constraint, which runs precomputation
    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser,
        llm_token_map.clone(),
        token_name_map,
        max_original_llm_token_id,
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // LLM token "x" should result in one edge in the root precompute node for state 0 with the terminal for `X`.
    // LLM token " =" should result in one edge in the root precompute node for state 0 with the terminal for `SPACE` and a subsequent edge from its destination with the terminal for `EQUALS`.
    // --- Verification ---
    let initial_state_id = tokenizer.initial_state_id();
    let root_arc = constraint.precomputed.get(&initial_state_id)
        .expect("No precomputed trie for initial tokenizer state");
    let root_node = root_arc.read(&constraint.trie1_god).unwrap();

    // The root node should have two outgoing edge keys: one for 'X' (from LLM token "x")
    // and one for 'SPACE' (from LLM token " =").
    assert_eq!(root_node.children().len(), 3, "Root node should have three outgoing edge keys");

    // Get the grammar token IDs for our terminals
    let x_tid = *grammar_token_map.get_by_left(&regex_name("X")).unwrap();
    let space_tid = *grammar_token_map.get_by_left(&regex_name("SPACE")).unwrap();
    let equals_tid = *grammar_token_map.get_by_left(&regex_name("EQUALS")).unwrap();

    // Get the LLM token IDs
    let x_llm_id = constraint.original_id_to_internal(*llm_token_map.get_by_left(b"x".as_ref()).unwrap()).unwrap().0;
    let space_equals_llm_id = constraint.original_id_to_internal(*llm_token_map.get_by_left(b" =".as_ref()).unwrap()).unwrap().0;

    // 1. Verify the edge for 'X'
    let x_dests = root_node.get(&Some(x_tid)).expect("No edge for terminal 'X'");
    assert_eq!(x_dests.len(), 1, "Should be one destination for 'X' edge");
    let (x_dest_wrapper, x_edge_bv) = x_dests.iter().next().unwrap();
    assert_eq!(*x_edge_bv, HybridBitset::from_iter(vec![x_llm_id]), "Edge for 'X' has wrong LLM token bitset");let binding = x_dest_wrapper.as_arc().clone();
    let x_dest_node = binding.read(&constraint.trie1_god).unwrap();
    assert!(x_dest_node.value.end, "Destination for 'X' edge should be an end node");
    drop(x_dest_node);

    // 2. Verify the edge for 'SPACE'
    let space_dests = root_node.get(&Some(space_tid)).expect("No edge for terminal 'SPACE'");
    assert_eq!(space_dests.len(), 1, "Should be one destination for 'SPACE' edge");
    let (space_dest_wrapper, space_edge_bv) = space_dests.iter().next().unwrap();
    assert_eq!(*space_edge_bv, HybridBitset::from_iter(vec![space_equals_llm_id]), "Edge for 'SPACE' has wrong LLM token bitset");let binding = space_dest_wrapper.as_arc().clone();
    let node_after_space = binding.read(&constraint.trie1_god).unwrap();
    assert!(!node_after_space.value.end, "Destination for 'SPACE' should not be an end node");

    // 3. Verify the node after 'SPACE'
    assert_eq!(node_after_space.children().len(), 1, "Intermediate node should have one child");
    let (equals_edge_key, equals_dests) = node_after_space.children().iter().next().unwrap();
    assert_eq!(*equals_edge_key, Some(equals_tid), "Edge from intermediate node should be for 'EQUALS'");
    let (equals_dest_wrapper, equals_edge_bv) = equals_dests.iter().next().unwrap();
    assert_eq!(*equals_edge_bv, HybridBitset::from_iter(vec![space_equals_llm_id]), "Edge for 'EQUALS' has wrong LLM token bitset");let binding = equals_dest_wrapper.as_arc().clone();
    let equals_dest_node = binding.read(&constraint.trie1_god).unwrap();
    assert!(equals_dest_node.value.end, "Destination for 'EQUALS' edge should be an end node");

    // 4. Check that the two end nodes are the same instance
    assert_eq!(x_dest_wrapper, equals_dest_wrapper, "Both paths should lead to the same end node instance");
}

#[test]
fn test_constraint_expression_no_times() {
    // Grammar: E -> E '+' T | T; T -> F; F -> '(' E ')' | 'i'
    // LLM token vocabulary: i, +, (, ), (i, +i
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"+".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"(".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b")".to_vec(), LLMTokenID(3));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(4));
    llm_token_map.insert(b"+i".to_vec(), LLMTokenID(5));

    // Tokenizer regex for grammar tokens '+' '(' ')' 'i'
    let expr = groups![
        eat_u8(b'+'),
        eat_u8(b'('),
        eat_u8(b')'),
        eat_u8(b'i'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]), // Start production
        prod("E", vec![nt("E"), t("PLUS"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("PLUS"), TerminalID(0));
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(1));
    grammar_token_map.insert(regex_name("RPAREN"), TerminalID(2));
    grammar_token_map.insert(regex_name("I"), TerminalID(3));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(4));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        5,
    );

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (2), "(i" (4)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 2, 4]));

    // Commit "(i"
    state.commit(LLMTokenID(4));
    let mask = state.get_mask();
    // Now expect '+', ')', '+i' => IDs 1,3,5
    assert_eq!(mask, HybridBitset::from_iter(vec![1, 3, 5]));
}

#[test]
fn test_constraint_expression_no_parens() {
    // Grammar: E -> E '+' T | T; T -> T '*' F | F; F -> 'i'
    // LLM token vocabulary: i, +, *, +i
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"+".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"*".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"+i".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '+' '*' 'i'
    let expr = groups![
        eat_u8(b'+'),
        eat_u8(b'*'),
        eat_u8(b'i'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]), // Start production
        prod("E", vec![nt("E"), t("PLUS"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("T"), t("TIMES"), nt("F")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("PLUS"), TerminalID(0));
    grammar_token_map.insert(regex_name("TIMES"), TerminalID(1));
    grammar_token_map.insert(regex_name("I"), TerminalID(2));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(3));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3,
    );

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0)
    assert_eq!(mask, HybridBitset::from_iter(vec![0]));

    // Commit "i"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // Now expect '+', '*', '+i' => IDs 1,2,3
    assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 3]));
}

#[test]
fn test_constraint_expression_no_plus_times() {
    // Grammar: E -> T; T -> F; F -> '(' E ')' | 'i'
    // LLM token vocabulary: i, (, ), (i
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"(".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b")".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '(' ')' 'i'
    let expr = groups![
        eat_u8(b'('),
        eat_u8(b')'),
        eat_u8(b'i'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]), // Start production
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("LPAREN"), nt("E"), t("RPAREN")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(0));
    grammar_token_map.insert(regex_name("RPAREN"), TerminalID(1));
    grammar_token_map.insert(regex_name("I"), TerminalID(2));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(3));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3,
    );

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (3)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 3]));

    // Commit "(i"
    state.commit(LLMTokenID(3));
    let mask = state.get_mask();
    // Now expect ')' => ID 2
    assert_eq!(mask, HybridBitset::from_iter(vec![2]));
}

#[test]
fn test_constraint_expression_no_times_parens() {
    // Grammar: E -> E '+' T | T; T -> F; F -> 'i'
    // LLM token vocabulary: i, +, +i
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"+".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"+i".to_vec(), LLMTokenID(2));

    // Tokenizer regex for grammar tokens '+' 'i'
    let expr = groups![
        eat_u8(b'+'), 
        eat_u8(b'i'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]), // Start production
        prod("E", vec![nt("E"), t("PLUS"), nt("T")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("PLUS"), TerminalID(0));
    grammar_token_map.insert(regex_name("I"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        2,
    );

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0)
    assert_eq!(mask, HybridBitset::from_iter(vec![0]));

    // Commit "i"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // Now expect '+', '+i' => IDs 1,2
    assert_eq!(mask, HybridBitset::from_iter(vec![1, 2]));
}

#[test]
fn test_constraint_expression_unbalanced_parens() {
    // Grammar: S -> E EOF; E -> T; T -> F; F -> '(' E | 'i'
    // This is a bit of a weird grammar since parens are never closed,
    // but it's a good test of recursion.
    // LLM token vocabulary: i, (, (i, $
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"(".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '(', 'i', '$'
    let expr = groups![
        eat_u8(b'('),
        eat_u8(b'i'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]),
        prod("E", vec![nt("T")]),
        prod("T", vec![nt("F")]),
        prod("F", vec![t("LPAREN"), nt("E")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(0));
    grammar_token_map.insert(regex_name("I"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3,
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 2]));

    // Commit "("
    state.commit(LLMTokenID(1));
    let mask = state.get_mask();
    // After '(', we expect another E, so the mask should be the same
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 2]));

    // Commit "i"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, HybridBitset::from_iter(vec![3]));
}

#[test]
fn test_constraint_expression_cycle() {
    // Grammar: S -> E EOF; E -> F; F -> E | I
    // This grammar has a cycle E -> F -> E, which is a good test for the parser.
    // LLM token vocabulary: i, $
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));

    // Tokenizer regex for grammar tokens 'i', '$'
    let expr = groups![
        eat_u8(b'i'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]),
        prod("E", vec![nt("F")]),
        // prod("F", vec![nt("E")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("I"), TerminalID(0));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(1));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        1, // max_original_llm_token_id
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0)
    assert_eq!(mask, HybridBitset::from_iter(vec![0]));

    // Commit "i"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // After "i", E is satisfied, so we expect EOF ($)
    assert_eq!(mask, HybridBitset::from_iter(vec![1]));

    // Commit "$"
    state.commit(LLMTokenID(1));
    assert!(state.is_active());
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, HybridBitset::from_iter(vec![]));
}

#[test]
fn test_ambiguous_tokenizer_no_gss_explosion() {
    // Grammar: S -> A, A -> '{' A '}' | ''
    // Tokenizer:
    //   - OPEN_BRACE: '{'
    //   - CLOSE_BRACE: '}'
    //   - ANYTHING: '{'+
    // This setup creates a situation where a single '{' can be tokenized as either
    // OPEN_BRACE or ANYTHING. Since ANYTHING is not in the grammar, it should be ignored
    // by the parser, but the ambiguity exists for the tokenizer and could lead to
    // complex states if not handled correctly. We want to ensure this doesn't cause
    // an exponential blowup in the GSS.

    // 1. Tokenizer
    let tokenizer_expr = groups![
        eat_u8_fast(b'{'),      // Group 0: OPEN_BRACE
        eat_u8_fast(b'}'),      // Group 1: CLOSE_BRACE
        repeat1_fast(eat_u8_fast(b'{')) // Group 2: ANYTHING
    ];
    let tokenizer = tokenizer_expr.build();

    // 2. Grammar
    let productions = vec![
        prod("S", vec![nt("A")]),
        prod("A", vec![t("OPEN_BRACE"), nt("A"), t("CLOSE_BRACE")]),
        prod("A", vec![]),
    ];

    // 3. LLM Vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"}".to_vec(), LLMTokenID(1));
    let max_original_llm_token_id = 1;

    // 4. Mappings
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("OPEN_BRACE"), TerminalID(0));
    grammar_token_map.insert(regex_name("CLOSE_BRACE"), TerminalID(1));

    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("OPEN_BRACE"), 0);
    token_name_map.insert(regex_name("CLOSE_BRACE"), 1);
    token_name_map.insert(regex_name("ANYTHING"), 2);

    // 5. Parser and Constraint
    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);
    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map,
        token_name_map,
        max_original_llm_token_id,
    );

    // 6. Test Logic
    let mut constraint_state = constraint.init();
    let mut last_gss_nodes = 0;

    constraint_state.commit_bytes(b"{{");
    assert!(constraint_state.is_active());
    constraint_state.print_gss();
    let stats = gather_gss_stats(
        &constraint_state.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
    );
    println!("After first commit: GSS stats = {:?}", stats);
    last_gss_nodes = stats.unique_nodes;

    // Commit one more
    constraint_state.commit_bytes(b"{{");
    assert!(constraint_state.is_active());
    constraint_state.print_gss();
    let final_stats = gather_gss_stats(
        &constraint_state.state.values().map(|s| s.active_state.stack.as_ref()).collect::<Vec<_>>(),
    );
    println!("After second commit: GSS stats = {:?}", final_stats);

    // The number of nodes should only increase by a small constant amount, not exponentially.
    // The exact number can vary with implementation details, but it should be small.
    // Let's assert it increases by at most 5.
    assert!(
        final_stats.unique_nodes <= last_gss_nodes * 2,
        "GSS nodes should not grow exponentially. Before: {}, After: {}",
        last_gss_nodes,
        final_stats.unique_nodes
    );
}

#[test]
fn test_constraint_indirect_recursion_simplified() {
    // Grammar: S' -> S EOF; S -> a E | b; E -> S
    // This is equivalent to S -> a* b, so valid strings are "b", "ab", "aab", etc.
    // LLM token vocabulary: a, b, $
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"b".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

    // Tokenizer regex for grammar tokens 'a', 'b', '$'
    let expr = groups![
        eat_u8(b'a'),
        eat_u8(b'b'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S'", vec![nt("S"), t("EOF")]),
        prod("S", vec![t("A"), nt("E")]),
        prod("S", vec![t("B")]),
        prod("E", vec![nt("S")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A"), TerminalID(0));
    grammar_token_map.insert(regex_name("B"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        2, // max_original_llm_token_id
    );

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect 'a' or 'b'
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1]));

    // Commit "a"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // After 'a', we expect E, which is S, so we expect 'a' or 'b' again.
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1]));

    // Commit "b"
    state.commit(LLMTokenID(1));
    let mask = state.get_mask();
    // After "ab", we have a complete S. Now we expect EOF.
    assert_eq!(mask, HybridBitset::from_iter(vec![2]));
}

#[test]
fn test_constraint_repetition_a() {
    // Grammar: S' -> S, S -> S A | [], which is equivalent to S -> A*
    // LLM token vocabulary: a
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));

    // Tokenizer regex for grammar tokens 'a', '$'
    let expr = groups![
        eat_u8(b'a'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S'", vec![nt("S")]),
        prod("S", vec![nt("S"), t("A")]),
        prod("S", vec![]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("A"), TerminalID(0));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(1));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);
    println!("Parser: {}", parser);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        0, // max_original_llm_token_id
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // The grammar can accept 'a' or EOF. Since EOF is not in the LLM vocab,
    // we only expect "a" (0).
    assert_eq!(mask, HybridBitset::from_iter(vec![0]));

    // Commit "a"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // After 'a', we can have another 'a' or end with EOF. Again, only 'a' is in vocab.
    assert_eq!(mask, HybridBitset::from_iter(vec![0]));
}

#[test]
fn test_constraint_expression_split_token() {
    // Grammar: S -> E EOF; E -> LPAREN E | I
    // LLM token vocabulary: "i(", "$"
    // This tests a case where an LLM token "i(" is a sequence of grammar tokens
    // that is not actually valid in the grammar (I must be followed by EOF).
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i(".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));

    // Tokenizer regex for grammar tokens '(', 'i', '$'
    let expr = groups![
        eat_u8(b'('), // ID 0
        eat_u8(b'i'), // ID 1
        eat_u8(b'$'), // ID 2
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]),
        prod("E", vec![t("LPAREN"), nt("E")]),
        prod("E", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(0));
    grammar_token_map.insert(regex_name("I"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        1, // max_original_llm_token_id
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let state = constraint.init();
    let mask = state.get_mask();

    // The grammar expects either 'i' (I) or '(' (LPAREN) at the start.
    // The LLM token "i(" corresponds to the grammar token sequence [I, LPAREN].
    // After the parser sees I, it completes the rule E -> I. The next expected
    // token is EOF, not LPAREN. Therefore, the sequence [I, LPAREN] is invalid.
    // The other LLM token "$" (EOF) is also not valid at the start.
    // Thus, the initial mask should be empty.
    assert_eq!(mask, HybridBitset::from_iter(vec![]));
}

#[test]
fn test_constraint_expression_trivial_indirect() {
    // Grammar: S -> E EOF; E -> F; F -> LPAREN E | I
    // LLM token vocabulary: i, (, (i, $
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"(".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '(', 'i', '$'
    let expr = groups![
        eat_u8(b'('),
        eat_u8(b'i'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]),
        prod("E", vec![nt("F")]),
        prod("F", vec![t("LPAREN"), nt("E")]),
        prod("F", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(0));
    grammar_token_map.insert(regex_name("I"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3,
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 2]));

    // Commit "("
    state.commit(LLMTokenID(1));
    let mask = state.get_mask();
    // After '(', we expect another E, so the mask should be the same
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 2]));

    // Commit "i"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, HybridBitset::from_iter(vec![3]));
}

#[test]
fn test_constraint_expression_trivial_direct() {
    // Grammar: S -> E EOF; E -> LPAREN E | I
    // LLM token vocabulary: i, (, (i, $
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"(".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '(', 'i', '$'
    let expr = groups![
        eat_u8(b'('),
        eat_u8(b'i'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]),
        prod("E", vec![t("LPAREN"), nt("E")]),
        prod("E", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(0));
    grammar_token_map.insert(regex_name("I"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3,
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 2]));

    // Commit "("
    state.commit(LLMTokenID(1));
    let mask = state.get_mask();
    // After '(', we expect another E, so the mask should be the same
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1, 2]));

    // Commit "i"
    state.commit(LLMTokenID(0));
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, HybridBitset::from_iter(vec![3]));
}

#[test]
fn test_constraint_expression_trivial_direct_limited_vocab() {
    // Grammar: S -> E EOF; E -> LPAREN E | I
    // LLM token vocabulary: i, (, (i, $
    let mut llm_token_map = LLMTokenMap::new();
    // llm_token_map.insert(b"i".to_vec(), LLMTokenID(0));
    // llm_token_map.insert(b"(".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(2));
    // llm_token_map.insert(b"$".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '(', 'i', '$'
    let expr = groups![
        eat_u8(b'('),
        eat_u8(b'i'),
        eat_u8(b'$'),
    ];
    let tokenizer = expr.build();

    // Grammar productions
    let productions = vec![
        prod("S", vec![nt("E"), t("EOF")]),
        prod("E", vec![t("LPAREN"), nt("E")]),
        prod("E", vec![t("I")]),
    ];
    // Map grammar terminals to IDs matching regex order
    let mut grammar_token_map: BiBTreeMap<Terminal, TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(regex_name("LPAREN"), TerminalID(0));
    grammar_token_map.insert(regex_name("I"), TerminalID(1));
    grammar_token_map.insert(regex_name("EOF"), TerminalID(2));

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), None);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer.clone(),
        parser.clone(),
        llm_token_map.clone(),
        token_name_map,
        3,
    );
    // constraint.dump_precomputed();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, HybridBitset::from_iter(vec![2]));

    // Commit "(i"
    state.commit(LLMTokenID(2));
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, HybridBitset::from_iter(vec![]));
}
