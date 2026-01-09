// src/test_constraint_basic.rs
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use std::time::Instant;

use bimap::BiBTreeMap;
use indoc::indoc;

use crate::constraint::{GrammarConstraint, GrammarConstraintConfig, LLMTokenBV};
use crate::datastructures::bitset::Bitset;
use crate::datastructures::hybrid_bitset::RangeSet;
use crate::finite_automata::{eat_u8, rep1};
use crate::glr::grammar::{nt, prod, regex_name, t, Terminal};
use crate::glr::parser::{
    GLRParserState,
    ParseState,
};
use crate::glr::table::generate_glr_parser_with_terminal_map;
use crate::interface::{
    eat_any_fast,
    eat_u8_fast,
    eat_u8_negation_fast,
    eat_u8_range_fast,
    repeat0_fast,
    repeat1_fast,
    CompiledGrammar,
    GrammarDefinition,
};
use crate::json_serialization::JSONConvertible;
use crate::interface::json_schema::json_schema_to_ebnf;
use crate::dfa_u8::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::TerminalID;
use crate::{choice_fast, groups, seq_fast};

#[test]
fn test_trivial() {
    // Grammar: S -> "a" "$"
    // Tokenizer: "a", "$"
    // LLM Vocab: "a", "$"

    let ebnf_grammar = indoc! {r#"
        s ::= A EOF;
        A ::= 'a';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        1, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    println!("Parser: {}", constraint.parser);
    // constraint.dump_precomputed2();
    constraint.dump_parser_dwa();

    println!("Initializing constraint state...");
    let mut state = constraint.init();
    println!("Initialized constraint state.");

    // Initial mask should allow "a"
    let mask1 = state.get_mask();
    assert_eq!(mask1, Bitset::from_iter(vec![0]));

    // Commit "a"
    state.commit(LLMTokenID(0)).unwrap();
    assert!(state.is_active());

    // Mask should now allow "$"
    let mask2 = state.get_mask();
    assert_eq!(mask2, Bitset::from_iter(vec![1]));

    // Commit "$"
    state.commit(LLMTokenID(1)).unwrap();
    assert!(state.is_active());

    // Mask should now be empty as we've reached the end of a valid parse
    let mask3 = state.get_mask();
    assert_eq!(mask3, Bitset::from_iter(vec![]));
}

/// Test that x;x is correctly parsed as two expression statements.
/// This is a minimal reproduction of a bug where semicolon is not allowed after x.
#[test]
fn test_x_semicolon_x() {
    // Grammar: program ::= expression_statement expression_statement? EOF
    //          expression_statement ::= expression ';'?
    //          expression ::= 'x'
    
    let ebnf_grammar = indoc! {r#"
        program ::= expression_statement expression_statement? EOF;
        expression_statement ::= expression ';'? ;
        expression ::= 'x' ;
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    println!("Grammar: {}", grammar_definition);

    // LLM tokens: "x" -> 0, ";" -> 1, "$" -> 2
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"x".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b";".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        2, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    println!("Tokenizer: {}", constraint.tokenizer);
    constraint.dump_vocab();
    constraint.dump_parser_dwa();

    let mut state = constraint.init();
    
    // Initial mask should allow "x"
    println!("\n--- Initial state ---");
    let mask0 = state.get_mask();
    println!("Initial mask: {:?}", mask0);
    assert!(mask0.contains(0), "x should be allowed initially");
    
    // Commit "x"
    println!("\n--- After committing 'x' ---");
    state.commit(LLMTokenID(0)).unwrap();
    let mask1 = state.get_mask();
    state.print_gss();
    println!("Mask after x: {:?}", mask1);
    
    // After x, semicolon should be allowed (for x;)
    assert!(mask1.contains(1), "semicolon should be allowed after x");
    // Also x should be allowed (for xx)
    assert!(mask1.contains(0), "x should be allowed after x (for second expression_statement)");
    // EOF should be allowed (for x$)
    assert!(mask1.contains(2), "EOF should be allowed after x");
    
    // Commit ";"
    println!("\n--- After committing ';' ---");
    state.commit(LLMTokenID(1)).unwrap();
    let mask2 = state.get_mask();
    println!("Mask after x;: {:?}", mask2);
    
    // After x;, x should be allowed (for x;x)
    assert!(mask2.contains(0), "x should be allowed after x;");
    // EOF should be allowed (for x;$)
    assert!(mask2.contains(2), "EOF should be allowed after x;");
    
    // Commit "x"
    println!("\n--- After committing second 'x' ---");
    state.commit(LLMTokenID(0)).unwrap();
    let mask3 = state.get_mask();
    println!("Mask after x;x: {:?}", mask3);
    
    // After x;x, EOF should be allowed
    assert!(mask3.contains(2), "EOF should be allowed after x;x");
}

#[ignore]
#[test]
fn test_constraint_simple() {
    // LLM tokens: "ab", "ac", "$"
    // Grammar tokens: "a", "ab", "b|c", "$" (EOF)
    // Grammar: S -> X $ ; X -> "a" ("b|c") | "ab"
    let ebnf_grammar = indoc! {r#"
        s ::= x EOF;
        x ::= A B_OR_C | AB;
        A ::= 'a';
        AB ::= 'ab';
        B_OR_C ::= 'b' | 'c';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"ab".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"ac".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        2, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    // constraint.dump_precomputed2();

    let mut constraint_state = constraint.init();

    // Initial mask
    let mask = constraint_state.get_mask();
    println!("Initial mask: {:?}", mask);
    assert_eq!(mask, Bitset::from_iter(vec![0, 1])); // Expect "ab" or "ac"

    // Commit "ab" (LLMTokenID 0)
    println!("{}", &constraint_state);
    constraint_state.commit(LLMTokenID(0)).unwrap();
    assert!(constraint_state.is_active());

    // Mask after committing "ab"
    println!("Constraint state:\n{}", &constraint_state);
    let mask_after_commit = constraint_state.get_mask();
    assert_eq!(mask_after_commit, Bitset::from_iter(vec![2])); // Expect "$" (EOF)

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method

    // Ensure the parse state after stepping the constraint with all LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
    // In general, this should be true if all LLM tokens cleanly match grammar tokens (or, equivalently, if the only non-empty entry in the precompute tree is under the initial tokenizer state).
    let llm_token = b"ab".to_vec();
    let grammar_tokenss = vec![vec!["A", "B_OR_C"], vec!["AB"]];
    let llm_token_id_for_comp = llm_token_map.get(&llm_token).unwrap();

    let mut constraint_state_for_comp = constraint.init(); // This is fine, it's a comment
    let parser = &constraint.parser;
    let grammar_token_map = &parser.terminal_map;
    // Mask before commit (optional, for debugging)
    let _mask_before = constraint_state_for_comp.get_mask();
    constraint_state_for_comp.commit(*llm_token_id_for_comp).unwrap();

    let mut parser_state_for_comp = parser.init_glr_parser_null(None);
    for grammar_tokens in grammar_tokenss {
        let mut parser_state = parser.init_glr_parser(None);
        for grammar_token in grammar_tokens {
            let grammar_token_id = grammar_token_map.get_by_left(&regex_name(grammar_token)).unwrap();
            parser_state.step(*grammar_token_id);
        }
        parser_state_for_comp.merge_with(parser_state);
    }

    assert_eq!(constraint_state_for_comp.state().len(), 1, "Constraint state should have one tokenizer state after commit");
    let (tokenizer_state_id_comp, actual_constraint_parser_state) = constraint_state_for_comp.state().iter().next().unwrap();
    let mut actual_constraint_parser_state = actual_constraint_parser_state.clone();

    // // For comparison, parser_state_for_comp's GSS acc needs to be "all_ones" like commit does.
    // let mut comparable_parser_gss = (*parser_state_for_comp.active_state.stack).clone();
    // let mut comparable_parser_active_state = ParseState::with_stack(Arc::new(comparable_parser_gss));
    //
    // Arc::make_mut(&mut comparable_parser_active_state.stack).reset_llm_tokens();
    // Arc::make_mut(&mut actual_constraint_parser_state.active_state.stack).reset_llm_tokens();
    //
    // assert_eq!(*tokenizer_state_id_comp, constraint.tokenizer.initial_state_id(), "Tokenizer should be in initial state");
    // assert_eq!(actual_constraint_parser_state.active_state, comparable_parser_active_state, "GSS structures should match");
}

#[test]
fn test_constraint_simple_minimized() {
    // LLM tokens: "a", "$"
    // Grammar tokens: "a", "$" (EOF)
    // Grammar: S -> X $ ; X -> "a"
    let ebnf_grammar = indoc! {r#"
        s ::= x EOF;
        x ::= A;
        A ::= 'a';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(1));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        1, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    // constraint.dump_precomputed1();
    // constraint.dump_precomputed2();
    constraint.dump_parser_dwa();

    let mut constraint_state = constraint.init();

    // Initial mask
    let mask = constraint_state.get_mask();
    println!("Initial mask: {:?}", mask);
    assert_eq!(mask, Bitset::from_iter(vec![0])); // Expect "a"

    // // Commit "a" (LLMTokenID 0)
    // println!("{}", &constraint_state);
    // constraint_state.commit(LLMTokenID(0)).unwrap();
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
    // let llm_token_id_for_comp = llm_token_map.get(&llm_token).unwrap();
    //
    // let mut constraint_state_for_comp = constraint.init();
    // constraint_state_for_comp.commit(*llm_token_id_for_comp).unwrap();
    //
    // let mut parser_state_for_comp = parser.init_glr_parser(None);
    // let grammar_token_id = grammar_token_map.get_by_left(&regex_name("A")).unwrap();
    // parser_state_for_comp.step(*grammar_token_id);
    //
    // assert_eq!(constraint_state_for_comp.state().len(), 1, "Constraint state should have one tokenizer state after commit");
    // let (_tokenizer_state_id_comp, actual_constraint_parser_state) = constraint_state_for_comp.state().iter().next().unwrap();
    //
    // assert_eq!(actual_constraint_parser_state.active_state, parser_state_for_comp.active_state, "GSS structures should match");
}

#[ignore]
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

    let ebnf_grammar = indoc! {r#"
        s ::= e;
        e ::= e PLUS t | t;
        t ::= t TIMES f | f;
        f ::= LPAREN e RPAREN | I;
        PLUS ::= '+';
        TIMES ::= '*';
        LPAREN ::= '(';
        RPAREN ::= ')';
        I ::= 'i';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        6, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    // constraint.dump_precomputed2();
    constraint.dump_parser_dwa();
    // constraint.dump_precomputed_special();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (3), "(i" (5)
    assert_eq!(mask, Bitset::from_iter(vec![0, 3, 5]));

    // Commit "(i"
    state.commit(LLMTokenID(5)).unwrap();
    let mask = state.get_mask();
    // Now expect '+', '*', ')', '+i' => IDs 1,2,4,6
    assert_eq!(mask, Bitset::from_iter(vec![1, 2, 4, 6]));

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method

    // Ensure the parse state after stepping the constraint with all LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
    // In general, this should be true if all LLM tokens cleanly match grammar tokens (or, equivalently, if the only non-empty entry in the precompute tree is under the initial tokenizer state).
    let llm_token = b"(i".to_vec();
    let grammar_tokens = vec!["LPAREN", "I"];
    let llm_token_id_for_comp = llm_token_map.get(&llm_token).unwrap();
    let parser = &constraint.parser;
    let grammar_token_map = &parser.terminal_map;
    let grammar_token_ids = grammar_tokens.iter().map(|token| grammar_token_map.get_by_left(&regex_name(token)).unwrap()).collect::<Vec<_>>();

    let mut constraint_state_for_comp = constraint.init();
    let _mask_before = constraint_state_for_comp.get_mask(); // Optional, for debugging
    constraint_state_for_comp.commit(*llm_token_id_for_comp).unwrap();

    let mut parser_state_for_comp = parser.init_glr_parser(None);
    for grammar_token_id in grammar_token_ids {
        parser_state_for_comp.step(*grammar_token_id);
    }

    assert_eq!(constraint_state_for_comp.state().len(), 1);
    let (tokenizer_state_id_comp, actual_constraint_parser_state) = constraint_state_for_comp.state().iter().next().unwrap();
    let mut actual_constraint_parser_state = actual_constraint_parser_state.clone();

    // // For comparison, parser_state_for_comp's GSS acc needs to be "all_ones" like commit does.
    // let mut comparable_parser_gss = (*parser_state_for_comp.active_state.stack).clone();
    // let mut comparable_parser_active_state = ParseState::with_stack(Arc::new(comparable_parser_gss));
    //
    // Arc::make_mut(&mut comparable_parser_active_state.stack).reset_llm_tokens();
    // Arc::make_mut(&mut actual_constraint_parser_state.active_state.stack).reset_llm_tokens();
    //
    // assert_eq!(*tokenizer_state_id_comp, constraint.tokenizer.initial_state_id(), "Tokenizer should be in initial state");
    // assert_eq!(actual_constraint_parser_state.active_state, comparable_parser_active_state, "GSS structures should match");
}

#[test]
fn test_constraint_expression_minimized_06_11_25() {
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"+".to_vec(), LLMTokenID(0));

    let ebnf_grammar = indoc! {r#"
        s ::= e;
        e ::= e '+' | t;
        t ::= t '*' | I;
        I ::= 'i';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        1, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    println!("Parser: {}", constraint.parser);
    constraint.dump_parser_dwa();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    assert_eq!(mask, Bitset::from_iter(vec![]));

    // Commit "(i"
    state.commit_bytes(b"i");
    state.print_gss();
    let mask = state.get_mask();
    assert_eq!(mask, Bitset::from_iter(vec![0]));
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


    // let _precomputed = GrammarConstraint::precompute1(
    //     &tokenizer,
    //     None,
    //     None,
    //     &internal_llm_token_map_for_precompute, // Use the manually created internal map
    //     &BiBTreeMap::new(), // empty name‐map
    //     internal_llm_token_map_for_precompute.iter().map(|(_, id)| id.0).max().unwrap_or(0),
    //     &BTreeMap::new(), // empty terminal_follow_map
    //     None,
    //     &mut BTreeMap::new(),
    // );
    // print_precomputed(&_precomputed);
    // println!("Done precomputing");
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

    // let _precomputed = GrammarConstraint::precompute1(
    //     &tokenizer,
    //     None,
    //     None,
    //     &internal_llm_token_map_for_precompute, // Use the manually created internal map
    //     &BiBTreeMap::new(), // empty name‐map
    //     internal_llm_token_map_for_precompute.iter().map(|(_, id)| id.0).max().unwrap_or(0),
    //     &BTreeMap::new(), // empty terminal_follow_map
    //     None,
    //     &mut BTreeMap::new(),
    // );
    // println!("Done precomputing");
}

#[test]
fn test_aborted_tokenizer_restart_equivalence() {
    // Tokenizer:
    // Group 0: "a" (A_T)
    // Group 1: "#" followed by an optional "a" (HASH_OPT_A_T)
    let ebnf_grammar = indoc! {r#"
        s ::= HASH_OPT_A | HASH_OPT_A A;
        A ::= 'a';
        HASH_OPT_A ::= '#' 'a'?;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    // LLM Tokens
    let mut llm_token_map = LLMTokenMap::new();
    let llm_hash = LLMTokenID(0);
    let llm_a = LLMTokenID(1);
    let llm_hash_a = LLMTokenID(2);
    llm_token_map.insert(b"#".to_vec(), llm_hash);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"#a".to_vec(), llm_hash_a);

    let max_original_llm_token_id = 2;

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );
    println!("parser: {}", constraint.parser);
    println!("Commit vocab: {:?}", constraint.commit_vocab);

    // Scenario 1: Commit "#", then "a"
    let mut constraint_state1 = constraint.init();
    println!("Scenario 1: Committing LLM Token '#' (ID {})", llm_hash.0);
    constraint_state1.commit(llm_hash).unwrap();
    println!("Scenario 1: State after committing '#': {:?}", constraint_state1.state().keys().map(|k|k.0).collect::<Vec<_>>());
    for (tid, glr_state) in constraint_state1.state() {
        // glr_state.log_gss(&format!("Scenario 1, after '#', GSS for tokenizer state {}", tid.0), TerminalID(0), false, false);
    }

    println!("\nScenario 1: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state1.commit(llm_a).unwrap();
    println!("Scenario 1: State after committing 'a': {:?}", constraint_state1.state().keys().map(|k|k.0).collect::<Vec<_>>());
     for (tid, glr_state) in constraint_state1.state() {
        // glr_state.log_gss(&format!("Scenario 1, after 'a', GSS for tokenizer state {}", tid.0), TerminalID(0), false, false);
    }

    // Scenario 2: Commit "#a"
    let mut constraint_state2 = constraint.init();
    println!("\nScenario 2: Committing LLM Token '#a' (ID {})", llm_hash_a.0);
    constraint_state2.commit(llm_hash_a).unwrap();
    println!("Scenario 2: State after committing '#a': {:?}", constraint_state2.state().keys().map(|k|k.0).collect::<Vec<_>>());
    for (tid, glr_state) in constraint_state2.state() {
        // glr_state.log_gss(&format!("Scenario 2, after '#a', GSS for tokenizer state {}", tid.0), TerminalID(0), false, false);
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
    let ebnf_grammar = indoc! {r#"
        s ::= HASH_OPT_AA | HASH_OPT_AA A A;
        A ::= 'a';
        HASH_OPT_AA ::= '#' ('a' 'a')?;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    // LLM Tokens
    let mut llm_token_map = LLMTokenMap::new();
    let llm_hash = LLMTokenID(0);    // "#"
    let llm_a = LLMTokenID(1);       // "a"
    let llm_hash_aa = LLMTokenID(2); // "#aa"
    llm_token_map.insert(b"#".to_vec(), llm_hash);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"#aa".to_vec(), llm_hash_aa);

    let max_original_llm_token_id = 2;

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );

    // Scenario 1: Commit "#", then "a"
    let mut constraint_state3 = constraint.init();
    println!("Scenario 1: Committing LLM Token '#' (ID {})", llm_hash.0);
    constraint_state3.commit(llm_hash).unwrap();
    println!("{}", &constraint_state3);

    println!("\nScenario 1: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state3.commit(llm_a).unwrap();
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
    constraint_state1.commit(llm_hash).unwrap();
    println!("{}", &constraint_state1);

    println!("\nScenario 3: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state1.commit(llm_a).unwrap();
    println!("{}", &constraint_state1);

    println!("\nScenario 3: Committing LLM Token 'a' (ID {})", llm_a.0);
    constraint_state1.commit(llm_a).unwrap();
    println!("{}", &constraint_state1);

    // Scenario 4: Commit "#aa"
    let mut constraint_state2 = constraint.init();
    println!("\nScenario 4: Committing LLM Token '#aa' (ID {})", llm_hash_aa.0);
    constraint_state2.commit(llm_hash_aa).unwrap();
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
    let ebnf_grammar = indoc! {r#"
        s ::= A;
        A ::= 'a'+;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    // 5. LLM vocabulary: "a" and "aaa"
    let mut llm_token_map = LLMTokenMap::new();
    let llm_a = LLMTokenID(0);
    let llm_aaa = LLMTokenID(1);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"aaa".to_vec(), llm_aaa);
    let max_original_llm_token_id = 1;

    // 7. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );

    // Scenario 1: Commit "a" three times
    let mut state1 = constraint.init();
    println!("Scenario 1: Committing 'a' (ID {})", llm_a.0);
    state1.commit(llm_a).unwrap();
    println!("{}", &state1);
    println!("Scenario 1: Committing 'a' (ID {}) again", llm_a.0);
    state1.commit(llm_a).unwrap();
    println!("{}", &state1);
    println!("Scenario 1: Committing 'a' (ID {}) a third time", llm_a.0);
    state1.commit(llm_a).unwrap();
    println!("{}", &state1);

    // Scenario 2: Commit "aaa" once
    let mut state2 = constraint.init();
    println!("\nScenario 2: Committing 'aaa' (ID {})", llm_aaa.0);
    state2.commit(llm_aaa).unwrap();
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

    let ebnf_grammar = indoc! {r#"
        #![ignore(WS)]
        s ::= A B;
        A ::= 'a';
        B ::= 'b';
        WS ::= ' ';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    println!("Grammar Definition: {:?}", grammar_definition);

    let mut llm_token_map = LLMTokenMap::new();
    let llm_a = LLMTokenID(0);
    let llm_b = LLMTokenID(1);
    let llm_ws = LLMTokenID(2);
    let llm_a_b = LLMTokenID(3);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"b".to_vec(), llm_b);
    llm_token_map.insert(b" ".to_vec(), llm_ws);
    llm_token_map.insert(b"a b".to_vec(), llm_a_b);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        3, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    println!("Tokenizer: {}", constraint.tokenizer);
    println!("Parser: {}", constraint.parser);
    // constraint.dump_precomputed1();
    constraint.dump_parser_dwa();

    // --- Runtime check ---
    // Scenario 1: commit "a", then " ", then "b"
    let mut state1 = constraint.init();
    assert_eq!(state1.get_mask(), Bitset::from_iter(vec![llm_a.0, llm_ws.0, llm_a_b.0]), "Initial mask should allow 'a' or 'a b'");
    state1.commit(llm_a).unwrap();
    assert_eq!(state1.get_mask(), Bitset::from_iter(vec![llm_b.0, llm_ws.0, llm_ws.0]), "After 'a', mask should allow 'b' or ' '");
    state1.commit(llm_ws).unwrap();
    assert_eq!(state1.get_mask(), Bitset::from_iter(vec![llm_b.0, llm_ws.0]), "After 'a ', mask should allow 'b'");
    state1.commit(llm_b).unwrap();
    // assert_eq!(state1.get_mask(), HybridBitset::from_iter(vec![llm_ws.0]), "After 'a b', mask should be empty (complete parse).");

    // --- Equivalence check ---
    let mut state2 = constraint.init();
    state2.commit(llm_a_b).unwrap();
    // assert_eq!(state2.get_mask(), HybridBitset::from_iter(vec![llm_ws.0]), "After committing 'a b', mask should be empty (complete parse).");
    assert_eq!(state1.state(), state2.state(), "States from ('a',' ','b') and ('a b') should be equivalent.");
}

#[test]
fn test_hideous_ambiguity() {
    // 1. Define the grammar
    let ebnf_grammar = indoc! {r#"
        s ::= FSTRING_MIDDLE FSTRING_MIDDLE;
        FSTRING_MIDDLE ::= 'a'+;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    // 3. LLM Token Map
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));

    // 6. Create the Constraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        0,
        &GrammarConstraintConfig::default(),
    );

    // 7. Initialize the Constraint State
    let mut constraint_state = constraint.init();

    // 8. Step with LLM Token "a" repeatedly
    let a_id = llm_token_map.get(&b"a"[..]).unwrap().0;
    for i in 0..10 { // Reduced iterations for faster test, was 1000
        println!("{}. Committing LLM token ID {}", i, a_id);
        let mask = constraint_state.get_mask();
        if !mask.contains(a_id) {
            println!("Token 'a' (ID {}) not in mask. Mask: {:?}. Stopping.", a_id, mask);
            break;
        }
        constraint_state.commit(LLMTokenID(a_id)).unwrap();
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
    let ebnf_grammar = indoc! {r#"
        s ::= DEF_T;
        DEF_T ::= "def";
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    // 2. LLM vocabulary: only "def", but with a non-zero original ID
    let mut llm_token_map = LLMTokenMap::new();
    // let def_original_llm_id = 750; // Using the ID from your Python script's log
    let def_original_llm_id = 0;
    llm_token_map.insert(b"def".to_vec(), LLMTokenID(def_original_llm_id));
    let max_original_llm_token_id = def_original_llm_id;

    // 6. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(), // Original LLMTokenID map
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );

    // constraint.dump_precomputed1(); // Optional: for debugging precomputation

    // 7. Initialize the constraint state.
    //    This calls constraint.init() internally.
    let mut constraint_state = constraint.init();
    let mask = constraint_state.get_mask();

    // 9. Define the expected mask.
    //    It should contain the original LLMTokenID for "def".
    let mut expected_mask = Bitset::zeros();
    expected_mask.insert(def_original_llm_id as usize); // Expecting the original LLM ID

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
    use crate::constraint_precompute::run_precompute1;
    use crate::dwa_i32::common::Label;
    use crate::dfa_u8::TokenizerStateID;

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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

    // In this test, original and internal are the same.
    let internal_llm_token_map: BTreeMap<_, _> =
        llm_token_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let internal_max_llm_token = max_original_llm_token_id;

    let terminals_count = parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::from([
        (tokenizer.initial_state_id(), tokenizer.initial_state_id())
    ]);

    let dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        state_to_rep,
    );

    // --- Verification ---
    // In weight-heavy mode, the terminal DWA has:
    // - Start state with tsid-labeled transitions (tsid info encoded in weights)
    // - Expanded weights in N×M space
    
    let start_state_id = dwa.body.start_state;
    // Start state should have tsid transitions (labels >= terminals_count)
    assert!(!dwa.states[start_state_id].transitions.is_empty(), 
        "Terminal DWA start state should have tsid transitions");
    
    // The DWA should have states and be non-trivial
    assert!(dwa.states.len() > 1, "DWA should have multiple states");
    
    // Check that weights are expanded (in N×M space)
    // Expanded weights have max position around internal_max_llm_token * num_tsids
    let mut found_expanded_weight = false;
    for state in &dwa.states.0 {
        if let Some(ref w) = state.final_weight {
            if let Some(max_pos) = w.rsb.last() {
                if max_pos > internal_max_llm_token {
                    found_expanded_weight = true;
                    break;
                }
            }
        }
    }
    assert!(found_expanded_weight, "DWA should have expanded weights (N×M space)");
}

#[ignore]
#[test]
fn test_precompute_x_eq() {
    use crate::constraint_precompute::run_precompute1;
    use crate::dwa_i32::common::Label;
    use crate::dfa_u8::TokenizerStateID;

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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

    let internal_llm_token_map: BTreeMap<_, _> =
        llm_token_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
    let internal_max_llm_token = max_original_llm_token_id;

    let terminals_count = parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = BTreeMap::from([
        (tokenizer.initial_state_id(), tokenizer.initial_state_id())
    ]);
    
    let dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        state_to_rep,
    );

    // --- Verification (weight-heavy mode) ---
    // In weight-heavy mode, the terminal DWA has:
    // - Start state with tsid-labeled transitions (tsid info encoded in weights)
    // - Expanded weights in N×M space
    
    let start_state_id = dwa.body.start_state;
    // Start state should have tsid transitions (labels >= terminals_count)
    assert!(!dwa.states[start_state_id].transitions.is_empty(), 
        "Terminal DWA start state should have tsid transitions");
    
    // The DWA should have states and be non-trivial
    assert!(dwa.states.len() > 1, "DWA should have multiple states");
    
    // Check that weights are expanded (in N×M space)
    let mut found_expanded_weight = false;
    for state in &dwa.states.0 {
        if let Some(ref w) = state.final_weight {
            if let Some(max_pos) = w.rsb.last() {
                if max_pos > internal_max_llm_token {
                    found_expanded_weight = true;
                    break;
                }
            }
        }
    }
    assert!(found_expanded_weight, "DWA should have expanded weights (N×M space)");
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

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
    assert_eq!(mask, Bitset::from_iter(vec![0, 2, 4]));

    // Commit "(i"
    state.commit(LLMTokenID(4)).unwrap();
    let mask = state.get_mask();
    // Now expect '+', ')', '+i' => IDs 1,3,5
    assert_eq!(mask, Bitset::from_iter(vec![1, 3, 5]));
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

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
    assert_eq!(mask, Bitset::from_iter(vec![0]));

    // Commit "i"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // Now expect '+', '*', '+i' => IDs 1,2,3
    assert_eq!(mask, Bitset::from_iter(vec![1, 2, 3]));
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

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
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 3]));

    // Commit "(i"
    state.commit(LLMTokenID(3)).unwrap();
    let mask = state.get_mask();
    // Now expect ')' => ID 2
    assert_eq!(mask, Bitset::from_iter(vec![2]));
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

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
    assert_eq!(mask, Bitset::from_iter(vec![0]));

    // Commit "i"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // Now expect '+', '+i' => IDs 1,2
    assert_eq!(mask, Bitset::from_iter(vec![1, 2]));
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

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
    // constraint.dump_precomputed1();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 2]));

    // Commit "("
    state.commit(LLMTokenID(1)).unwrap();
    let mask = state.get_mask();
    // After '(', we expect another E, so the mask should be the same
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 2]));

    // Commit "i"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, Bitset::from_iter(vec![3]));
}

#[test]
fn test_constraint_expression_unbalanced_parens2() {
    let mut llm_token_map = LLMTokenMap::new();
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());
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
        3,
    );

    // Initial state and step
    let mut state = constraint.init();

    // Commit "(i"
    state.commit_bytes(b"(i");
    println!("state: {}", state);
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, Bitset::from_iter(vec![3]));
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

    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());

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
    // constraint.dump_precomputed1();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0)
    assert_eq!(mask, Bitset::from_iter(vec![0]));

    // Commit "i"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // After "i", E is satisfied, so we expect EOF ($)
    assert_eq!(mask, Bitset::from_iter(vec![1]));

    // Commit "$"
    state.commit(LLMTokenID(1)).unwrap();
    assert!(state.is_active());
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, Bitset::from_iter(vec![]));
}

fn load_gpt2_vocab() -> Option<(LLMTokenMap, usize)> {
    use std::io::BufReader;
    use std::fs;

    // Attempt to load gpt2 vocab from various paths
    let paths = vec![
        "vocab.json", 
        "src/tests/data/vocab.json", 
        "gpt2_vocab.json",
        "benchmarking/gpt2_vocab.json",
        "python/.cache/py_benchmark_vocabs/gpt2_vocab.json",
    ];
    
    for p in &paths {
        if let Ok(file) = fs::File::open(p) {
            let reader = BufReader::new(file);
            if let Ok(vocab_json) = serde_json::from_reader::<_, serde_json::Value>(reader) {
                let vocab_map = match vocab_json.as_object() {
                    Some(m) => m,
                    None => continue,
                };
                
                let mut llm_token_map = LLMTokenMap::new();
                let mut max_id = 0;
                let mut valid = true;

                for (token_str, id_val) in vocab_map {
                    let id = match id_val.as_u64() {
                        Some(id) => id as usize,
                        None => { valid = false; break; }
                    };
                    if id > max_id { max_id = id; }

                    // Minimal Byte-Pair Encoding reversal for GPT-2:
                    // Map 'Ġ' (U+0120) to space, 'Ċ' (U+010A) to newline.
                    let bytes: Vec<u8> = token_str.chars().flat_map(|c| {
                        match c {
                            'Ġ' => vec![b' '],
                            'Ċ' => vec![b'\n'],
                            c if c.is_ascii() => vec![c as u8],
                            _ => c.to_string().into_bytes(), // Fallback
                        }
                    }).collect();
                    llm_token_map.insert(bytes, LLMTokenID(id));
                }
                
                if !valid {
                    continue;
                }
                
                // Verify this is a real GPT-2 vocab (should have thousands of tokens)
                if llm_token_map.len() < 1000 {
                    eprintln!("Warning: {} has only {} tokens, not a real GPT-2 vocab", p, llm_token_map.len());
                    continue;
                }
                
                // Verify it has basic tokens
                if !llm_token_map.contains_key(&vec![b'{']) {
                    eprintln!("Warning: {} missing '{{' token, not a real GPT-2 vocab", p);
                    continue;
                }
                
                println!("Loaded vocab from {}", p);
                return Some((llm_token_map, max_id));
            }
        }
    }

    None
}

#[test]
fn test_json_gpt2_initial_mask_bruteforce() -> Result<(), Box<dyn std::error::Error>> {
    let ebnf_grammar = indoc! {r#"
        #![ignore(WS)]
        value ::= object | array | STRING | NUMBER | 'true' | 'false' | 'null' ;
        object ::= '{' pairs '}' ;
        pairs ::= pair (',' pair)* | ;
        pair ::= STRING ':' value ;
        array ::= '[' items ']' ;
        items ::= value (',' value)* | ;
        STRING ::= '"' [^"]* '"' ;
        NUMBER ::= '-'? [0-9]+ ;
        WS ::= [ \t\n\r]+ ;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar)?;

    let (llm_token_map, max_id) = match load_gpt2_vocab() {
        Some(v) => v,
        None => {
            println!("Skipping test_json_gpt2_initial_mask_bruteforce: vocab.json not found.");
            println!("To run, download https://huggingface.co/openai-community/gpt2/raw/main/vocab.json to project root.");
            return Ok(());
        }
    };

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_id,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    let mask = state.get_mask();
    println!("Initial mask size: {} / {}", mask.len(), llm_token_map.len());

    // Brute force verification
    println!("Starting brute-force verification of {} tokens...", llm_token_map.len());
    let mut errors = 0;
    for (bytes, id) in &llm_token_map {
        let mut temp_state = constraint.init();
        temp_state.commit(*id).unwrap();
        let is_valid = temp_state.is_valid();
        let allowed = mask.contains(id.0);

        if is_valid != allowed {
             let s = String::from_utf8_lossy(bytes);
             println!("Mismatch! Token ID {}: Mask={}, Valid={}. Token: {:?}", id.0, allowed, is_valid, s);
             errors += 1;
             if errors > 20 { panic!("Too many mismatches."); }
        }
    }
    assert_eq!(errors, 0, "Initial mask does not match brute-force validity check.");

    Ok(())
}

#[test]
fn test_js_minimized_ebnf_string() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load and compile the grammar from the EBNF file
    let ebnf_grammar = indoc! {r#"
        program ::= (expression ';')* EOF;
        expression ::= '!'? (IDENTIFIER | STRING_LITERAL) ;
        EOF ::= '$';

        STRING_LITERAL ::= '"' [^"]* '"' ;
        IDENTIFIER ::= 'a' ;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(&ebnf_grammar)?;

    // 2. Define the LLM vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    let llm_a = LLMTokenID(0);
    let llm_not_quote = LLMTokenID(1);
    let llm_quote = LLMTokenID(2);
    llm_token_map.insert(b"a".to_vec(), llm_a);
    llm_token_map.insert(b"!\"".to_vec(), llm_not_quote);
    llm_token_map.insert(b"\"".to_vec(), llm_quote);
    let max_original_llm_token_id = 2;

    // 3. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );
    // println!("Tokenizer: {}", constraint.tokenizer);
    // println!("Parser: {}", constraint.parser);
    // constraint.dump_precomputed0();

    // 4. Initialize state and get the initial mask
    let mut state = constraint.init();
    let mask1 = state.get_mask();

    // The grammar can start with an IDENTIFIER ("a"), a unary '!' ("!\""), or a STRING_LITERAL ("\"").
    // It can also start with other things like 'let', 'if', 'while', '{', '(', 'true', 'false', a number, or a unary '-'.
    // The LLM tokens provided match the start of IDENTIFIER, unary '!', and STRING_LITERAL.
    let expected_mask1 = Bitset::from_iter(vec![llm_a.0, llm_not_quote.0, llm_quote.0]);
    assert_eq!(
        mask1,
        expected_mask1,
        "Initial mask should allow 'a', '!\"', and '\"'"
    );

    // 5. Commit "a" and get the next mask
    state.commit(llm_a).unwrap();
    state.print_gss();
    let mask2 = state.get_mask();

    let expected_mask2 = Bitset::from_iter(vec![]);
    assert_eq!(
        mask2,
        expected_mask2,
    );

    Ok(())
}

#[test]
fn test_js_like_grammar_initial_mask() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the EBNF grammar
    // Note: Use lowercase for non-terminals (s, x) since uppercase names are treated as terminals.
    let ebnf_grammar = indoc! {r#"
        s ::= x x '$';
        x ::= ( '!' x | 'a' ) ';'?;
    "#};

    // 2. Parse and compile the grammar
    let grammar_definition = GrammarDefinition::from_ebnf(&ebnf_grammar)?;
    println!("Grammar: {}", grammar_definition);
    let _compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    println!("Parser: {}", _compiled_grammar.glr_parser);

    // 3. Define the LLM vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    let llm_semicolons = LLMTokenID(0);
    let llm_empty_string_semicolon = LLMTokenID(1);
    llm_token_map.insert(b";;;".to_vec(), llm_semicolons);
    llm_token_map.insert(b"a;".to_vec(), llm_empty_string_semicolon);
    let max_original_llm_token_id = 1;

    // 4. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );
    constraint.dump_parser_dwa();

    // 5. Initialize state and get the initial mask
    let mut state = constraint.init();
    state.commit_bytes(b"a");
    state.print_gss();
    let mask2 = state.get_mask();

    assert!(state.is_active(), "State should be active after committing 'a'");
    let expected_mask2 = Bitset::from_iter(vec![llm_empty_string_semicolon.0]);
    assert_eq!(
        mask2,
        expected_mask2,
    );

    Ok(())
}

#[test]
fn test_js_like_grammar_initial_mask_minimized() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the EBNF grammar
    let ebnf_grammar = indoc! {r#"
        program ::= unary_expression unary_expression '$';
        unary_expression ::= ( '!' unary_expression | 'X' ) ';'?;
    "#};

    // 2. Parse and compile the grammar
    let grammar_definition = GrammarDefinition::from_ebnf(&ebnf_grammar)?;
    println!("Grammar: {}", grammar_definition);
    let _compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));

    // 3. Define the LLM vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    let llm_semicolons = LLMTokenID(0);
    llm_token_map.insert(b";;".to_vec(), llm_semicolons);
    let max_original_llm_token_id = 0;

    // 4. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );
    println!("Parser: {}", constraint.parser);

    // 5. Initialize state and get the initial mask
    let mut state = constraint.init();
    state.commit_bytes(b"X");
    state.print_gss();
    let mask2 = state.get_mask();

    assert!(state.is_active(), "State should be active after committing 'X'");
    let expected_mask2 = Bitset::from_iter(vec![]);
    assert_eq!(
        mask2,
        expected_mask2,
    );

    Ok(())
}

#[test]
fn test_ebnf_ignore_directive_with_partial_match() -> Result<(), Box<dyn std::error::Error>> {
    // This test checks the behavior of the #![ignore(...)] directive with
    // LLM tokens that are either fully ignored or partially ignored.

    // 1. Define the EBNF grammar. 'IGNORE' matches one or more spaces or "/*" sequences.
    // We use a regex to combine them into a single terminal to avoid a panic,
    // as the current implementation only supports one ignore terminal ID.
    let ebnf_grammar = indoc! {r#"
        // Instruct the parser to ignore Whitespace and single-line Comments.
        #![ignore(IGNORE)]

        program ::= 'x' ;

        // --- Lexical Grammar (Minimal) ---
        IGNORE ::= ( ' ' | '/*' )+ ;
    "#};

    // 2. Parse and compile the grammar
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar)?;

    // 3. Define the LLM vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    let llm_space_eq = LLMTokenID(0); // " =" - starts with an ignored char, but '=' is invalid
    let llm_comment = LLMTokenID(1);  // "/*" - is a full ignored token
    llm_token_map.insert(b" =".to_vec(), llm_space_eq);
    llm_token_map.insert(b"/*".to_vec(), llm_comment);
    let max_original_llm_token_id = 1;

    // 4. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );
    println!("Parser: {}", constraint.parser);
    constraint.dump_parser_dwa();

    // 5. Initialize state and get the initial mask
    let mut state = constraint.init();
    let mask1 = state.get_mask();

    // 6. Assert the initial mask
    // The grammar can start with 'x' or any number of IGNORE tokens.
    // "/*" is a full IGNORE token and should be allowed.
    // " =" starts with " ", which matches IGNORE, but is followed by "=", which is not
    // a valid token. Therefore, the entire LLM token " =" does not form a valid
    // sequence of grammar tokens and should not be in the mask.
    let expected_mask1 = Bitset::from_iter(vec![llm_comment.0]);
    assert_eq!(
        mask1,
        expected_mask1,
        "Initial mask should allow '/*' and 'x', but not ' ='"
    );

    Ok(())
}

// #[test]
// fn test_gss_structural_sharing_factor() -> Result<(), Box<dyn std::error::Error>> {
//     // This test verifies that for a grammar with a known ambiguity that can cause
//     // GSS explosion, the structural sharing remains effective. A low sharing factor
//     // indicates that many structurally identical sub-graphs are being correctly
//     // deduplicated.
//
//     // 1. Minimal grammar that causes GSS explosion without proper sharing.
//     //    See `test_js_if_statement_gss_explosion` for a detailed explanation.
//     let js_grammar_ebnf = indoc! {r#"
//         program ::= statement* EOF;
//         EOF ::= '<|EOF|>';
//
//         statement ::= if_statement | expression | block ;
//         block ::= '{' statement* '}' ;
//         if_statement ::= 'if' expression statement ;
//
//         expression ::= IDENTIFIER IDENTIFIER | IDENTIFIER ;
//         IDENTIFIER ::= [a-zA-Z_] [a-zA-Z0-9_]* ;
//     "#};
//     let grammar_definition = GrammarDefinition::from_ebnf(js_grammar_ebnf)?;
//     let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
//     let parser = compiled_grammar.glr_parser;
//
//     // 2. Replicate the GSS setup from `precompute3` to test a single token step.
//     //    We are interested in the terminal for 'if', which is TerminalID(1) in this compiled grammar.
//
//     let tid = 1; // Terminal ID for 'if'
//     let terminal = TerminalID(tid);
//
//     let mut glr_state = parser.init_glr_parser_with_acc();
//
//     const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = BelowBottomReductionMode::ContinueFromAll;
//     glr_state.process_token_advanced(terminal, &ProcessTokenAdvancedConfig { below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE, current_token: None, ..Default::default() });
//
//     // 3. Get stats and assert on the structural sharing factor.
//     let stats = glr_state.active_state.stack.inner.stats();
//     println!("Stats for terminal ID {}: {:?}", tid, stats);
//
//     let THRESHOLD = 0.49; // A reasonably high sharing factor
//     if !(stats.structural_sharing_factor > THRESHOLD) {
//         // Print the GSS structure before and after normalization for debugging.
//         println!("GSS (low sharing factor):");
//         println!("{}", glr_state.active_state.stack.inner.to_graph_string(false));
//         println!("GSS after normalization (what it ideally should be):");
//         println!("{}", glr_state.active_state.stack.inner.normalize().to_graph_string(false));
//     }
//     assert!(
//         stats.structural_sharing_factor > THRESHOLD,
//         "Structural sharing factor ({}) was not greater than {}, indicating poor GSS node sharing",
//         stats.structural_sharing_factor,
//         THRESHOLD
//     );
//
//     Ok(())
// }

// #[test]
// fn test_gss_structural_sharing_factor2() -> Result<(), Box<dyn std::error::Error>> {
//     // This test verifies that for a grammar with a known ambiguity that can cause
//     // GSS explosion, the structural sharing remains effective. A low sharing factor
//     // indicates that many structurally identical sub-graphs are being correctly
//     // deduplicated.
//
//     // 1. Minimal grammar that causes GSS explosion without proper sharing.
//     //    See `test_js_if_statement_gss_explosion` for a detailed explanation.
//     let js_grammar_ebnf = indoc! {r#"
//         program ::= (statement ';')* EOF;
//
//         statement ::=
//             'p01' VALUE
//           | 'p02' VALUE
//           | 'p03' VALUE
//           | 'p04' VALUE
//           | 'p05' VALUE
//         ;
//
//         VALUE ::= IDENTIFIER;
//
//         EOF ::= '$';
//
//         IDENTIFIER ::= 'a';
//     "#};
//     let grammar_definition = GrammarDefinition::from_ebnf(js_grammar_ebnf)?;
//     let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
//     let parser = compiled_grammar.glr_parser;
//     println!("Parser: {}", parser);
//
//     // 2. Replicate the GSS setup from `precompute3` to test a single token step.
//     //    We are interested in the terminal for 'if', which is TerminalID(1) in this compiled grammar.
//     use crate::datastructures::gss_leveled_adapter::Acc;
//     use crate::glr::parser::{BelowBottomReductionMode, ProcessTokenAdvancedConfig};
//
//     let mut glr_state = parser.init_glr_parser_with_acc();
//
//     const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = BelowBottomReductionMode::ContinueFromAll;
//     // for tid in parser.terminal_map.right_values() {
//     let terminal = Terminal::terminal("VALUE");
//     let tid = *parser.terminal_map.get_by_left(&terminal).unwrap();
//     let mut glr_state = glr_state.clone();
//     glr_state.process_token_advanced(tid, &ProcessTokenAdvancedConfig { below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE, current_token: None, ..Default::default() });
//
//     // 3. Get stats and assert on the structural sharing factor.
//     let stats = glr_state.active_state.stack.inner.stats();
//     println!("Stats for terminal ID {}: {:?}", tid.0, stats);
//
//     let THRESHOLD = 0.49; // A reasonably high sharing factor
//     if !(stats.structural_sharing_factor > THRESHOLD) {
//         // Print the GSS structure before and after normalization for debugging.
//         println!("GSS (low sharing factor):");
//         println!("{}", glr_state.active_state.stack.inner.to_graph_string(false));
//         println!("GSS after normalization (what it ideally should be):");
//         println!("{}", glr_state.active_state.stack.inner.normalize().to_graph_string(false));
//     }
//     assert!(
//         stats.structural_sharing_factor > THRESHOLD,
//         "Structural sharing factor ({}) was not greater than {}, indicating poor GSS node sharing",
//         stats.structural_sharing_factor,
//         THRESHOLD
//     );
//
//     Ok(())
// }

// #[test]
// fn test_gss_structural_sharing_factor3() -> Result<(), Box<dyn std::error::Error>> {
//     // This test verifies that for a grammar with a known ambiguity that can cause
//     // GSS explosion, the structural sharing remains effective. A low sharing factor
//     // indicates that many structurally identical sub-graphs are being correctly
//     // deduplicated.
//
//     // 1. Minimal grammar that causes GSS explosion without proper sharing.
//     //    See `test_js_if_statement_gss_explosion` for a detailed explanation.
//     let js_grammar_ebnf = indoc! {r#"
//         program ::= (statement ';')* EOF;
//
//         statement ::=
//             'p01' VALUE POST
//           | 'p02' VALUE POST
//           | 'p03' VALUE POST
//           | 'p04' VALUE POST
//           | 'p05' VALUE POST
//           | 'p06' VALUE POST
//           | 'p07' VALUE POST
//           | 'p08' VALUE POST
//           | 'p09' VALUE POST
//           | 'p10' VALUE POST
//           | 'p11' VALUE POST
//           | 'p12' VALUE POST
//           | 'p13' VALUE POST
//           | 'p14' VALUE POST
//           | 'p15' VALUE POST
//           | 'p16' VALUE POST
//           | 'p17' VALUE POST
//           | 'p18' VALUE POST
//           | 'p19' VALUE POST
//           | 'p20' VALUE POST
//           | 'p21' VALUE POST
//           | 'p22' VALUE POST
//           | 'p23' VALUE POST
//           | 'p24' VALUE POST
//           | 'p25' VALUE POST
//           | 'p26' VALUE POST
//           | 'p27' VALUE POST
//           | 'p28' VALUE POST
//           | 'p29' VALUE POST
//           | 'p30' VALUE POST
//           | 'p31' VALUE POST
//           | 'p32' VALUE POST
//         ;
//
//         VALUE ::= IDENTIFIER;
//
//         POST ::=
//             't01' | 't02' | 't03' | 't04' | 't05' | 't06' | 't07' | 't08'
//           | 't09' | 't10' | 't11' | 't12' | 't13' | 't14' | 't15' | 't16'
//           | 't17' | 't18' | 't19' | 't20' | 't21' | 't22' | 't23' | 't24'
//           | 't25' | 't26' | 't27' | 't28' | 't29' | 't30' | 't31' | 't32'
//           | 't33' | 't34' | 't35' | 't36' | 't37' | 't38' | 't39' | 't40'
//           | 't41' | 't42' | 't43' | 't44' | 't45' | 't46' | 't47' | 't48'
//           | 't49' | 't50'
//         ;
//
//         EOF ::= '$';
//
//         IDENTIFIER ::= 'a';
//     "#};
//     let grammar_definition = GrammarDefinition::from_ebnf(js_grammar_ebnf)?;
//     let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
//     let parser = compiled_grammar.glr_parser;
//     println!("Parser: {}", parser);
//
//     // 2. Replicate the GSS setup from `precompute3` to test a single token step.
//     //    We are interested in the terminal for 'if', which is TerminalID(1) in this compiled grammar.
//     use crate::datastructures::gss_leveled_adapter::Acc;
//     use crate::glr::parser::{BelowBottomReductionMode, ProcessTokenAdvancedConfig};
//
//     let mut glr_state = parser.init_glr_parser_with_acc();
//
//     const BELOW_BOTTOM_REDUCE_MODE: BelowBottomReductionMode = BelowBottomReductionMode::ContinueFromAll;
//     for tid in parser.terminal_map.right_values() {
//         let tid = tid.0;
//         let terminal = TerminalID(tid);
//         let mut glr_state = glr_state.clone();
//         glr_state.process_token_advanced(terminal, &ProcessTokenAdvancedConfig { below_bottom_mode: BELOW_BOTTOM_REDUCE_MODE, current_token: None, ..Default::default() });
//
//         // 3. Get stats and assert on the structural sharing factor.
//         let stats = glr_state.active_state.stack.inner.stats();
//         println!("Stats for terminal ID {}: {:?}", tid, stats);
//
//         let THRESHOLD = 0.49; // A reasonably high sharing factor
//         if !(stats.structural_sharing_factor > THRESHOLD) {
//             // Print the GSS structure before and after normalization for debugging.
//             println!("GSS (low sharing factor):");
//             println!("{}", glr_state.active_state.stack.inner.to_graph_string(false));
//             println!("GSS after normalization (what it ideally should be):");
//             println!("{}", glr_state.active_state.stack.inner.normalize().to_graph_string(false));
//         }
//         assert!(
//             stats.structural_sharing_factor > THRESHOLD,
//             "Structural sharing factor ({}) was not greater than {}, indicating poor GSS node sharing",
//             stats.structural_sharing_factor,
//             THRESHOLD
//         );
//     }
//
//     Ok(())
// }

#[test]
fn test_ebnf_grammar_initial_mask() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the EBNF grammar string
    let ebnf_grammar = indoc! {r#"
        program ::= IGNORE;
        IGNORE ::= ' ' | '$@';
    "#};

    // 2. Parse and compile the grammar
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar)?;

    // 3. Define the LLM vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    let space_token_id = LLMTokenID(0);
    let at_token_id = LLMTokenID(1);
    llm_token_map.insert(b" ".to_vec(), space_token_id);
    llm_token_map.insert(b"@".to_vec(), at_token_id);
    let max_original_llm_token_id = 1;

    // 4. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &GrammarConstraintConfig::off(),
    );
    println!("Tokenizer: {}", constraint.tokenizer);
    println!("Parser: {}", constraint.parser);

    // 5. Initialize state and get the initial mask
    let mut state = constraint.init();
    let mask = state.get_mask();

    // 6. Assert the expected mask
    let expected_mask = Bitset::from_iter(vec![space_token_id.0]);
    assert_eq!(
        mask,
        expected_mask,
        "Mask should only allow the ignored space token"
    );

    Ok(())
}

#[test]
fn test_ebnf_grammar_initial_mask_mandatory_pass() -> Result<(), Box<dyn std::error::Error>> {
    // This test is a minimal pair to the failing `test_ebnf_grammar_initial_mask`.
    let ebnf_grammar = indoc! {r#"
        program ::= IGNORE;
        IGNORE ::= ' ';
    "#};

    // 2. Parse and compile the grammar
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar)?;

    // 3. Define the LLM vocabulary
    let mut llm_token_map = LLMTokenMap::new();
    let space_token_id = LLMTokenID(0);
    let at_token_id = LLMTokenID(1);
    llm_token_map.insert(b" ".to_vec(), space_token_id);
    llm_token_map.insert(b"@".to_vec(), at_token_id);
    let max_original_llm_token_id = 1;

    // 4. Create the GrammarConstraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &GrammarConstraintConfig::off(),
    );
    println!("Tokenizer: {}", constraint.tokenizer);
    println!("Parser: {}", constraint.parser);

    // 5. Initialize state and get the initial mask
    let mut state = constraint.init();
    let mask = state.get_mask();

    // 6. Assert the expected mask
    let expected_mask = Bitset::from_iter(vec![space_token_id.0]);
    assert_eq!(
        mask,
        expected_mask,
        "Mask should only allow the ignored space token"
    );

    Ok(())
}


#[test]
fn test_precompute_self_loop_from_shared_states() {
    // This test is designed to reproduce a panic in `Precomputer0::dfs` where
    // the algorithm attempts to create a self-loop on a trie node.
    //
    // The scenario is based on the user's description:
    // - A vocabulary with shared prefixes ("za", "zaabm", "zaabn").
    // - A stateful tokenizer (e.g., for `a+`) that can lead to complex
    //   sharing of trie nodes across different tokenizer states during precomputation.
    //
    // The combination can lead to a situation where a trie node `N` is being
    // processed as a source for a new edge, while `N` itself is already present
    // in the queue of candidate destination nodes for the next position in the
    // vocabulary segment. This causes the algorithm to select `N` as its own
    // child, triggering an `assert_ne!` panic.

    // 1. Tokenizer with a stateful component (`a+`)
    let ebnf_grammar = indoc! {r#"
        s ::= Z_T A_PLUS_T B_T M_T | Z_T A_PLUS_T B_T N_T;
        Z_T ::= 'z';
        A_PLUS_T ::= 'a'+;
        B_T ::= 'b';
        M_T ::= 'm';
        N_T ::= 'n';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    // 2. LLM Vocabulary with shared prefixes
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"za".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"zaabm".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"zaabn".to_vec(), LLMTokenID(2));
    let max_original_llm_token_id = 2;

    // 5. Run precomputation and check for panics
    // We expect this to fail with a panic until the bug is fixed.
    // The test asserts that it *should not* panic.
    // let result = std::panic::catch_unwind(|| {
    //     let _ = GrammarConstraint::new_from_grammar_definition(
    //         Arc::new(grammar_definition),
    //         llm_token_map,
    //         max_original_llm_token_id,
    //         &GrammarConstraintConfig::default(),
    //         None,
    //     );
    // });
    //
    // assert!(
    //     result.is_ok(),
    //     "The precomputation should not have panicked. Panic info: {:?}",
    //     result.err()
    // );
}

#[ignore]
#[test]
fn test_js_full_grammar_gss_explosion() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Setting up for Full JS Grammar GSS Explosion Test ---");
    let js_grammar_ebnf = fs::read_to_string("src/js.ebnf")?;
    let grammar_definition = GrammarDefinition::from_ebnf(&js_grammar_ebnf)?;
    println!("Full JS grammar compiled successfully.");

    // LLM vocab: single-byte tokens to compose the repeating chunk.
    let mut llm_token_map = LLMTokenMap::new();
    let mut max_id = 0;
    let repeating_chunk = b"if(true){";
    let mut vocab = BTreeSet::new();
    for byte in repeating_chunk {
        vocab.insert(*byte);
    }

    for (i, byte_val) in vocab.iter().enumerate() {
        llm_token_map.insert(vec![*byte_val], LLMTokenID(i));
        max_id = i;
    }

    let mut config = crate::constraint::GrammarConstraintConfig::default();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_id,
        &config,
    );
    println!("GrammarConstraint constructed successfully.");

    let mut constraint_state = constraint.init();

    // Warm-up with 2 chunks
    for _ in 0..2 {
        for byte in repeating_chunk {
            constraint_state.commit_bytes(&[*byte]);
        }
    }
    assert!(constraint_state.is_active());
    println!("After warm-up of 2 chunks of '{}'", String::from_utf8_lossy(repeating_chunk));

    // Measure time for the 3rd chunk
    let start1 = Instant::now();
    for byte in repeating_chunk {
        constraint_state.commit_bytes(&[*byte]);
    }
    let time1 = start1.elapsed();
    assert!(constraint_state.is_active());
    println!("\n3rd chunk '{}' took {:?}", String::from_utf8_lossy(repeating_chunk), time1);

    // Measure time for the 4th chunk
    let start2 = Instant::now();
    for byte in repeating_chunk {
        constraint_state.commit_bytes(&[*byte]);
    }
    let time2 = start2.elapsed();
    assert!(constraint_state.is_active());
    println!("\n4th chunk '{}' took {:?}", String::from_utf8_lossy(repeating_chunk), time2);

    // The execution time should not accelerate significantly. If it does, it indicates
    // an exponential blowup. We allow some tolerance.
    let tolerance_factor = 2.0;
    assert!(
        time2.as_secs_f64() <= time1.as_secs_f64() * tolerance_factor,
        "Execution time is accelerating, indicating an explosion. Time for 3rd chunk: {:?}, Time for 4th chunk: {:?}",
        time1,
        time2
    );

    Ok(())
}

#[test]
fn test_js_if_statement_gss_explosion() -> Result<(), Box<dyn std::error::Error>> {
    // This test reproduces the GSS explosion seen in `test_js_constraint_integration`
    // with the input "if(1){if(1){...". The grammar has a recursive structure for
    // statements (Statement -> IfStatement, IfStatement -> 'if' '(' Expression ')' Statement)
    // which can lead to exponential growth in GSS nodes if states are not merged properly.
    println!("--- Setting up for JS GSS Explosion Test ---");
    /*
    Essence of this test

    We deliberately construct the smallest grammar that forces a GLR parser to keep two equally plausible parses alive for the same input prefix and then continue parsing the exact same remainder of the input in both parses. With repetition, these branches multiply, causing accelerating growth in the Graph-Structured Stack (GSS).

    Minimal ingredients (all are necessary here):
    1) Lexical overlap: the literal keyword 'if' and IDENTIFIER both match "if".
       This makes the prefix "if a" ambiguous:
       - Path A (if-statement): 'if' expression statement
       - Path B (expression-as-statement): expression ::= IDENTIFIER IDENTIFIER
         In Path B, we let the expression consume "if a" as two identifiers.

    2) Shared continuation: a block "{" statement* "}" that both paths accept.
       After both parses consume "if a", they see the same next token '{' and
       both recurse into the block. Repeating the ambiguous unit (ifa{) causes
       the number of simultaneous parses to grow combinatorially, which shows up
       as accelerating GSS node growth.

    What removes the explosion (any one of these):
    - Reserve the keyword: make IDENTIFIER not match "if" (removes lexical overlap).
    - Remove the 'block' alternative (or do not allow 'block' as a statement)
      so the two branches do not share the same continuation.
    - Remove 'statement ::= expression' or make 'expression' not consume both
      identifiers (e.g., expression ::= IDENTIFIER). Then the two branches do
      not align on the same next token '{'.
    - Force parentheses around either the if or the call-like expression:
      if '(' expr ')' stmt or IDENTIFIER '(' expr ')' so the ambiguous prefix
      disappears in this minimalist setup.

    Why the LLM token set is ['i','f','a','{']:
    We use single-byte LLM tokens to keep the constraint layer simple while still
    forming grammar terminals ('if', IDENTIFIER, '{'). The sequence "ifa{" is the
    smallest repeatable chunk that produces the child-vs-sibling ambiguity which
    triggers the blowup.

    How the assertion detects the explosion:
    We commit the same chunk three times and measure unique GSS node counts.
    If growth is linear, increases between samples should not accelerate.
    We fail the test when the second increase is larger than the first,
    indicating combinatorial (explosive) growth.
    */
    // Minimal grammar for explosion:
    // - 'if' literal overlaps with IDENTIFIER
    // - expression can be IDENTIFIER IDENTIFIER (so "if a" is a valid expression)
    // - block '{' statement* '}' is a shared continuation for both parses
    let js_grammar_ebnf = indoc! {r#"
        program ::= statement* EOF;
        EOF ::= '<|EOF|>';

        statement ::= if_statement | expression | block ;
        block ::= '{' statement* '}' ;
        if_statement ::= 'if' expression statement ;

        expression ::= IDENTIFIER IDENTIFIER | IDENTIFIER ;
        IDENTIFIER ::= [a-zA-Z_] [a-zA-Z0-9_]* ;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(js_grammar_ebnf)?;
    println!("Grammar compiled successfully.");

    // Minimal LLM vocab: single-byte tokens that compose "ifa{"
    // This keeps the constraint path simple while still forming the needed terminals.
    let mut llm_token_map = LLMTokenMap::new();
    let mut max_id = 0;
    for (i, s) in ["i", "f", "a", "{"].iter().enumerate() {
        llm_token_map.insert(s.as_bytes().to_vec(), LLMTokenID(i));
        max_id = i;
    }

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_id,
        &GrammarConstraintConfig::default(),
    );
    println!("GrammarConstraint constructed successfully.");
    println!("{}", constraint.parser);

    let mut constraint_state = constraint.init();
    // "ifa{" is the minimal ambiguous unit:
    // - Can be parsed as an if-statement ('if' + expr "a" + child block "{")
    // - Or as an expression statement ("if a") followed by a sibling block "{"
    // Repeating it forces both branches to re-encounter the same continuation,
    // producing accelerating GSS growth.
    let repeating_chunk = b"ifa{";

    // First chunk
    for _ in 0..1 {
        for byte in repeating_chunk {
            constraint_state.commit_bytes(&[*byte]);
        }
    }
    assert!(constraint_state.is_active());
    println!("After first chunk '{}'", String::from_utf8_lossy(repeating_chunk));
    constraint_state.print_gss_stats();
    let nodes1 = constraint_state.num_unique_nodes();
    constraint_state.print_gss();

    // Second chunk
    for byte in repeating_chunk {
        constraint_state.commit_bytes(&[*byte]);
    }
    assert!(constraint_state.is_active());
    println!("\nAfter second chunk '{}'", String::from_utf8_lossy(repeating_chunk));
    constraint_state.print_gss_stats();
    let nodes2 = constraint_state.num_unique_nodes();
    constraint_state.print_gss();

    // Third chunk
    for byte in repeating_chunk {
        constraint_state.commit_bytes(&[*byte]);
    }
    assert!(constraint_state.is_active());
    println!("\nAfter third chunk '{}'", String::from_utf8_lossy(repeating_chunk));
    constraint_state.print_gss_stats();
    let nodes3 = constraint_state.num_unique_nodes();
    constraint_state.print_gss();

    let increase1 = nodes2 - nodes1;
    let increase2 = nodes3 - nodes2;

    println!("\nNode counts: {}, {}, {}", nodes1, nodes2, nodes3);
    println!("Increases: {}, {}", increase1, increase2);

    // Check for accelerating growth: nodes3 - nodes2 > nodes2 - nodes1
    // If true, parsing states are multiplying (combinatorial blowup), as intended.
    // The increase in nodes should not accelerate. If it does, it indicates
    // an exponential blowup.
    assert!(
        increase2 <= increase1,
        "GSS node growth is accelerating, indicating an explosion. First increase: {}, Second increase: {}",
        increase1,
        increase2
    );

    Ok(())
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
    grammar_token_map.insert(regex_name("ANYTHING"), TerminalID(2));

    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert(regex_name("OPEN_BRACE"), 0);
    token_name_map.insert(regex_name("CLOSE_BRACE"), 1);
    token_name_map.insert(regex_name("ANYTHING"), 2);

    // 5. Parser and Constraint
    let parser = generate_glr_parser_with_terminal_map(&productions, grammar_token_map.clone(), &HashSet::new(), HashSet::new());
    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map,
        token_name_map,
        max_original_llm_token_id,
    );

    // 6. Test Logic
    let mut constraint_state = constraint.init();

    // Warm-up commit
    constraint_state.commit_bytes(b"{{");
    assert!(constraint_state.is_active());
    println!("After warm-up '{{': {} states", constraint_state.state.len());
    constraint_state.print_gss();

    // First single '{' commit
    constraint_state.commit_bytes(b"{");
    assert!(constraint_state.is_active());
    println!("After first single '{{': {} states", constraint_state.state.len());
    constraint_state.print_gss();
    let nodes1 = constraint_state.num_unique_nodes();

    // Second single '{' commit$
    constraint_state.commit_bytes(b"{");
    assert!(constraint_state.is_active());
    println!("After second single '{{': {} states", constraint_state.state.len());
    constraint_state.print_gss();
    let nodes2 = constraint_state.num_unique_nodes();

    // Third single '{' commit
    constraint_state.commit_bytes(b"{");
    assert!(constraint_state.is_active());
    println!("After third single '{{': {} states", constraint_state.state.len());
    constraint_state.print_gss();
    let nodes3 = constraint_state.num_unique_nodes();

    let increase1 = nodes2 - nodes1;
    let increase2 = nodes3 - nodes2;

    // The increase in nodes should stabilize or decrease, not grow.
    assert!(
        increase2 <= increase1,
        "GSS node growth should not accelerate. First increase: {}, Second increase: {}",
        increase1,
        increase2
    );
}

#[test]
fn test_constraint_indirect_recursion_minimized() {
    // Grammar: S' -> S EOF; S -> a E | b; E -> S
    // This is equivalent to S -> a* b, so valid strings are "b", "ab", "aab", etc.
    // LLM token vocabulary: a, b, $
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"b".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));

    let ebnf_grammar = indoc! {r#"
        s_prime ::= s EOF;
        s ::= A e | B;
        e ::= s;
        A ::= 'a';
        B ::= 'b';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        2, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect 'a' or 'b'
    assert_eq!(mask, Bitset::from_iter(vec![0, 1]));

    // Commit "a"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // After 'a', we expect E, which is S, so we expect 'a' or 'b' again.
    assert_eq!(mask, Bitset::from_iter(vec![0, 1]));

    // Commit "b"
    state.commit(LLMTokenID(1)).unwrap();
    let mask = state.get_mask();
    // After "ab", we have a complete S. Now we expect EOF.
    assert_eq!(mask, Bitset::from_iter(vec![2]));
}

#[test]
fn test_constraint_repetition_a() {
    // Grammar: S' -> S, S -> S A | [], which is equivalent to S -> A*
    // LLM token vocabulary: a
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));

    let ebnf_grammar = indoc! {r#"
        s_prime ::= s;
        s ::= s A | ;
        A ::= 'a';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        0, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    println!("Parser: {}", constraint.parser);
    constraint.dump_parser_dwa();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // The grammar can accept 'a' or EOF. Since EOF is not in the LLM vocab,
    // we only expect "a" (0).
    assert_eq!(mask, Bitset::from_iter(vec![0]));

    // Commit "a"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // After 'a', we can have another 'a' or end with EOF. Again, only 'a' is in vocab.
    assert_eq!(mask, Bitset::from_iter(vec![0]));
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

    let ebnf_grammar = indoc! {r#"
        s ::= e EOF;
        e ::= LPAREN e | I;
        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        1, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );
    // constraint.dump_precomputed1();
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
    assert_eq!(mask, Bitset::from_iter(vec![]));
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

    let ebnf_grammar = indoc! {r#"
        s ::= e EOF;
        e ::= f;
        f ::= LPAREN e | I;
        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        3,
        &GrammarConstraintConfig::default(),
    );
    // constraint.dump_precomputed1();
    // constraint.dump_precomputed2();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 2]));

    // Commit "("
    state.commit(LLMTokenID(1)).unwrap();
    let mask = state.get_mask();
    // After '(', we expect another E, so the mask should be the same
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 2]));

    // Commit "i"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, Bitset::from_iter(vec![3]));
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

    let ebnf_grammar = indoc! {r#"
        s ::= e EOF;
        e ::= LPAREN e | I;
        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        3,
        &GrammarConstraintConfig::default(),
    );
    println!("Parser: {}", constraint.parser);
    constraint.dump_parser_dwa();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (1), "(i" (2)
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 2]));

    // Commit "("
    state.commit(LLMTokenID(1)).unwrap();
    let mask = state.get_mask();
    state.print_gss();
    // After '(', we expect another E, so the mask should be the same
    assert_eq!(mask, Bitset::from_iter(vec![0, 1, 2]));

    // Commit "i"
    state.commit(LLMTokenID(0)).unwrap();
    let mask = state.get_mask();
    state.print_gss();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    assert_eq!(mask, Bitset::from_iter(vec![3]));
}

#[test]
fn test_constraint_expression_trivial_direct_limited_vocab() {
    // Grammar: S -> E EOF; E -> LPAREN E | I
    // LLM token vocabulary: only "(i"
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"(i".to_vec(), LLMTokenID(2));

    let ebnf_grammar = indoc! {r#"
        s ::= e EOF;
        e ::= LPAREN e | I;
        LPAREN ::= '(';
        I ::= 'i';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    println!("{}", grammar_definition);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        3,
        &GrammarConstraintConfig::default(),
    );
    println!("Tokenizer: {}", constraint.tokenizer);
    println!("Parser: {}", constraint.parser);
    constraint.dump_parser_dwa();

    // Initial state and step
    let mut state = constraint.init();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: "(i" (2) is the only token
    assert_eq!(mask, Bitset::from_iter(vec![2]));

    // Commit "(i"
    state.commit(LLMTokenID(2)).unwrap();
    println!("After committing (i):");
    state.print_gss();
    
    let mask = state.get_mask();
    // After "(i", the inner E is satisfied. The outer E is satisfied. We now expect EOF.
    // But there's no EOF token in the vocab, so mask should be empty.
    assert_eq!(mask, Bitset::from_iter(vec![]));
}

/// Test that building the terminal DWA from a tokenizer and LLM vocabulary
/// correctly results in a DWA that traces through the segments of an LLM token.
#[ignore]
#[test]
fn test_tokenizer_vocab_to_terminal_dwa_aa() {
    use crate::constraint_precompute::run_precompute1;
    use crate::finite_automata::{Expr, ExprGroups, ExprGroup};
    use crate::dwa_i32::{DWA, Weight};
    
    // Build tokenizer: just terminal 0 = 'a'
    let tokenizer = ExprGroups {
        groups: vec![ExprGroup {
            expr: Expr::U8Seq(b"a".to_vec()),
            is_non_greedy: false,
        }],
    }.build();
    
    println!("Tokenizer DFA:\n{}", tokenizer);

    // LLM vocab: "aa" -> 0
    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"aa".to_vec(), LLMTokenID(0));

    let terminals_count = 1; // Just terminal 'a' (id=0)
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();
    
    // Number of tokenizer states for weight-heavy encoding
    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        0, // max internal token id
        terminals_count,
        state_to_rep,
    );
    
    println!("Actual Terminal DWA:\n{}", terminal_dwa);
    
    // In weight-heavy mode, the terminal DWA has:
    // - Start state with tsid-labeled transitions (tsid info encoded in weights)
    // - Weights are in N×M space
    
    let start_state_id = terminal_dwa.body.start_state;
    // Start state should have tsid transitions (labels >= terminals_count)
    assert!(!terminal_dwa.states[start_state_id].transitions.is_empty(), 
        "Terminal DWA start state should have tsid transitions");
    
    // The DWA should have states and be non-trivial
    assert!(terminal_dwa.states.len() > 1, "DWA should have multiple states");
}

// #[ignore]
// #[test]
// fn test_gss_explosion_from_ambiguity() -> Result<(), Box<dyn std::error::Error>> {
//     // This test uses the grammar from `test_js_minimized_ebnf_string` to reproduce
//     // a low structural sharing factor, which is a symptom of GSS explosion.
//     // When parsing from a combined GSS state, processing a common token like an
//     // identifier can lead to many structurally similar but distinct GSS paths,
//     // revealing poor node sharing if the GSS is not normalized.
//
//     // 1. Grammar from `test_js_minimized_ebnf_string`
//     let ebnf_grammar = indoc! {r#"
//         program ::= (expression ';')* EOF;
//         expression ::= '!'? (IDENTIFIER | STRING_LITERAL) ;
//         EOF ::= '$';
//
//         STRING_LITERAL ::= '"' [^"]* '"' ;
//         IDENTIFIER ::= 'a' ;
//     "#};
//     let grammar_definition = GrammarDefinition::from_ebnf(&ebnf_grammar)?;
//     let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
//     let parser = compiled_grammar.glr_parser;
//     println!("Parser: {}", parser);
//
//     // 2. Replicate the GSS setup from `precompute3`
//     let mut glr_state = parser.init_glr_parser_with_acc();
//     Arc::make_mut(&mut glr_state.active_state.stack).inner = glr_state.active_state.stack.inner.apply(|acc| {
//         let mut acc = acc.clone();
//         acc.llm_tokens_union = HybridBitset::max_ones();
//         acc
//     });
//
//     for i in 0..50 {
//         let mut next_glr_state: Option<GLRParserState> = None;
//         for (terminal, terminal_id) in &parser.terminal_map {
//             let mut glr_state_copy = glr_state.clone();
//             glr_state_copy.process_token_advanced(*terminal_id, &ProcessTokenAdvancedConfig { below_bottom_mode: BelowBottomReductionMode::ContinueFromAll, current_token: None, ..Default::default() });
//             let edge_bv = HybridBitset::from_iter(vec![i, terminal_id.0]);
//             allow_only_llm_tokens_on_stored_trie_nodes_and_prune_arc(&mut glr_state_copy.active_state.stack, &edge_bv, &mut HashMap::new());
//             if glr_state_copy.is_ok() {
//                 if let Some(existing) = &mut next_glr_state {
//                     existing.merge_with(glr_state_copy);
//                 } else {
//                     next_glr_state = Some(glr_state_copy);
//                 }
//             }
//         }
//         glr_state = next_glr_state.expect("At least one terminal should be processable");
//     }
//
//     // 4. Check stats before normalization. A low sharing factor indicates the problem.
//     let stats_before = glr_state.active_state.stack.inner.stats();
//     println!("Stats before normalization: {:?}", stats_before);
//
//     // 5. Normalize the GSS and check stats again. Normalization should fix the issue.
//     let normalized_gss = glr_state.active_state.stack.inner.normalize();
//     let stats_after = normalized_gss.stats();
//     println!("Stats after normalization: {:?}", stats_after);
//
//     // 6. Assertions
//     // The key issue is a low structural sharing factor before normalization.
//     // We expect it to be less than some threshold, indicating redundancy.
//     assert!(stats_before.structural_sharing_factor < 0.8, "Expected a relatively low structural sharing factor (< 0.8) before normalization, but got {}", stats_before.structural_sharing_factor);
//
//     // Normalization should significantly reduce the number of total nodes and thus increase the sharing factor.
//     assert!(stats_after.total_unique_nodes < stats_before.total_unique_nodes, "Normalization should reduce the total number of unique nodes. Before: {}, After: {}", stats_before.total_unique_nodes, stats_after.total_unique_nodes);
//     assert!(stats_after.structural_sharing_factor > stats_before.structural_sharing_factor, "Normalization should improve the structural sharing factor. Before: {}, After: {}", stats_before.structural_sharing_factor, stats_after.structural_sharing_factor);
//
//     Ok(())
// }

#[test]
fn test_json_schema_mask_generation() {
    // 1. Define minimal JSON schema mimicking PackageJson structure
    let schema_json = r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "additionalProperties": true
    }"#;
    
    // 2. Convert to EBNF
    let ebnf = json_schema_to_ebnf(schema_json).unwrap();
    println!("Generated EBNF:\n{}", ebnf);
    
    let grammar_definition = GrammarDefinition::from_ebnf(&ebnf).unwrap();

    // 3. Setup Token Map (mimic PackageJson failure trace)
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{".to_vec(), LLMTokenID(90));
    llm_token_map.insert(b"\n".to_vec(), LLMTokenID(198));
    llm_token_map.insert(b" ".to_vec(), LLMTokenID(220));
    llm_token_map.insert(b" \"".to_vec(), LLMTokenID(366));
    llm_token_map.insert(b"name".to_vec(), LLMTokenID(3672));
    
    // 4. Init Constraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        5000,
        &GrammarConstraintConfig::default(),
    );
    
    let mut state = constraint.init();
    
    // 5. Commit sequence: { \n "
    println!("Commit '{{'");
    state.commit(LLMTokenID(90)).expect("Commit {");
    
    println!("Commit '\\n'");
    state.commit(LLMTokenID(198)).expect("Commit \\n");
    
    println!("Commit ' '");
    state.commit(LLMTokenID(220)).expect("Commit space");
    
    println!("Commit ' \"'");
    state.commit(LLMTokenID(366)).expect("Commit \""); 
    
    // 6. Check if "name" is allowed
    let mask = state.get_mask();
    println!("Mask contains 3672 ('name')? {}", mask.contains(3672));
    assert!(mask.contains(3672), "Token 'name' (3672) should be allowed!");
}

#[test]
fn test_json_schema_gpt2_real_vocab() {
    // 1. Define minimal JSON schema
    let schema_json = r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        },
        "required": ["name"],
        "additionalProperties": true
    }"#;
    let ebnf = json_schema_to_ebnf(schema_json).unwrap();
    let grammar_definition = GrammarDefinition::from_ebnf(&ebnf).unwrap();

    // 2. Load REAL GPT-2 Vocab
    let (llm_token_map, max_id) = load_gpt2_vocab()
        .expect("No valid GPT-2 vocab found! This test requires a real GPT-2 vocab with thousands of tokens. \
                 Try: wget -O benchmarking/gpt2_vocab.json https://huggingface.co/openai-community/gpt2/raw/main/vocab.json");

    println!("Loaded real vocab with {} tokens. Max ID: {}", llm_token_map.len(), max_id);

    // 3. Init Constraint
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map.clone(),
        max_id,
        &GrammarConstraintConfig::default(),
    );
     
    let mut state = constraint.init();
    
    // 4. Commit sequence: { \n "
    // Note: GPT-2 tokens might be different than single bytes!
    // We should use the IDs that correspond to the bytes we want, if possible.
    // However, we are testing IF the sequence of TOKENS is valid.
    // The previous trace said:
    // { -> 90
    // \n -> 198 (Ċ)
    // " " -> 220 (Ġ)
    // " \"" -> 366 (Ġ")
    
    // Let's Verify these IDs exist in map and contain correct bytes
    assert!(llm_token_map.values().any(|id| id.0 == 90), "ID 90 must exist");
    assert!(llm_token_map.values().any(|id| id.0 == 3672), "ID 3672 (name) must exist");

    println!("Commit '{{' (ID 90)");
    state.commit(LLMTokenID(90)).expect("Commit {");
    
    println!("Commit '\\n' (ID 198)");
    state.commit(LLMTokenID(198)).expect("Commit \\n");
    
    println!("Commit ' ' (ID 220)");
    state.commit(LLMTokenID(220)).expect("Commit space");
    
    println!("Commit ' \"' (ID 366)");
    state.commit(LLMTokenID(366)).expect("Commit \""); 
    
    // 5. Verify 'name' (3672) is allowed
    let mask = state.get_mask();
    println!("Real GPT-2 Mask contains 3672? {}", mask.contains(3672));
    assert!(mask.contains(3672), "'name' token (3672) should be allowed in real GPT-2 vocab!");
}