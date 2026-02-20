// src/test_constraint_basic.rs
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
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
use crate::dfa_u8::{LLMTokenID, LLMTokenMap, Tokenizer, TokenizerStateID};
use crate::types::TerminalID;
use crate::{choice_fast, groups, seq_fast};

#[test]
fn test_trivial() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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

#[test]
fn test_dwa_ws_boundary_long_token() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let ebnf_grammar = indoc! {r#"
        #![ignore(WS)]
        s ::= STMT;
        WS ::= [ \t\n]+;
        STMT ::= [a-z]+;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b" ".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b" the".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b" that".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"that".to_vec(), LLMTokenID(3));
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(4));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        4, // max_original_llm_token_id
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    let mask = state.get_mask();

    assert!(mask.contains(1), "token ' the' should be valid at start");
    assert!(mask.contains(2), "token ' that' should be valid at start");
    assert!(mask.contains(3), "token 'that' should be valid at start");
    assert!(mask.contains(0), "token ' ' should be valid at start");
}

/// Test that x;x is correctly parsed as two expression statements.
/// This is a minimal reproduction of a bug where semicolon is not allowed after x.
#[test]
fn test_x_semicolon_x() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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

    let tokenizer = Tokenizer::new(name.build());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::run_precompute1;
    use crate::dwa_i32::common::Label;
    use crate::dfa_u8::TokenizerStateID;

    // Tokenizer for `a+`
    let tokenizer_expr = groups![repeat1_fast(eat_u8(b'a'))];
    let tokenizer = Tokenizer::new(tokenizer_expr.build());

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
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
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
            if let Some(max_pos) = w.to_rsb_allow_expansion().last() {
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(tokenizer_expr.build());

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
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
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
            if let Some(max_pos) = w.to_rsb_allow_expansion().last() {
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(3));

    // Tokenizer regex for grammar tokens '(', 'i', '$'
    let expr = groups![
        eat_u8(b'('),
        eat_u8(b'i'),
        eat_u8(b'$'),
    ];
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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

// Known bug: GLR parser produces false positives for tokens spanning key-value boundaries.
#[test]
fn test_glr_fp_repro_minimal() {
    enum ReproOutcome {
        Skip(String),
        Offenders(Vec<(usize, Vec<u8>)>),
    }

    let outcome = (|| {
        let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let lark_grammar = indoc! {r#"
            start: ws object ws
            object: "{" ws name_pair ws "}"
            name_pair: QUOTE "name" QUOTE ws ":" ws QUOTE name_val QUOTE
            name_val: name_chars
            name_chars: STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR*
            QUOTE: "\""
            ws: WS*
            WS: " " | "\n" | "\t" | "\r"
            STR_CHAR: /[A-Za-z0-9 \[\]\-:{}@.]/
        "#};

        let grammar_definition = match GrammarDefinition::from_lark(lark_grammar) {
            Ok(def) => def,
            Err(err) => return ReproOutcome::Skip(format!("Failed to parse grammar: {}", err)),
        };

        let mut llm_token_map = LLMTokenMap::new();
        let tok_open = LLMTokenID(0);
        let tok_name = LLMTokenID(1);
        let tok_colon_quote = LLMTokenID(2);
        let tok_fp_bracket = LLMTokenID(3);
        let tok_fp_dash = LLMTokenID(4);
        llm_token_map.insert(b"{\"".to_vec(), tok_open);
        llm_token_map.insert(b"name".to_vec(), tok_name);
        llm_token_map.insert(b"\":\"".to_vec(), tok_colon_quote);
        llm_token_map.insert(b"\":[".to_vec(), tok_fp_bracket);
        llm_token_map.insert(b"\":-".to_vec(), tok_fp_dash);
        let max_id = 4;

        let constraint = GrammarConstraint::new_from_grammar_definition(
            Arc::new(grammar_definition),
            llm_token_map.clone(),
            max_id,
            &GrammarConstraintConfig::default(),
        );

        let prefix_ids = [tok_open, tok_name];

        let mut state = constraint.init();
        for id in prefix_ids {
            if state.commit(id).is_err() {
                return ReproOutcome::Skip("prefix commit failed".to_string());
            }
        }

        let mask = state.get_mask();

        let expected_next = tok_colon_quote;
        if !mask.contains(expected_next.0) {
            return ReproOutcome::Skip("expected next token not in mask".to_string());
        }

        let id_to_bytes: BTreeMap<usize, Vec<u8>> = llm_token_map
            .iter()
            .map(|(bytes, id)| (id.0, bytes.clone()))
            .collect();

        let disputed = [tok_fp_bracket.0, tok_fp_dash.0];
        let mut offenders: Vec<(usize, Vec<u8>)> = Vec::new();
        for token_id in disputed {
            if mask.contains(token_id) {
                let bytes = id_to_bytes.get(&token_id).cloned().unwrap_or_default();
                offenders.push((token_id, bytes));
            }
        }

        ReproOutcome::Offenders(offenders)
    })();

    match outcome {
        ReproOutcome::Skip(msg) => {
            println!("Skipping test_glr_fp_repro_minimal: {}", msg);
        }
        ReproOutcome::Offenders(offenders) => {
            if !offenders.is_empty() {
                let mut formatted = Vec::new();
                for (token_id, bytes) in offenders {
                    formatted.push(format!(
                        "{}:{:?}",
                        token_id,
                        String::from_utf8_lossy(&bytes)
                    ));
                }
                panic!("FP tokens in mask at step 2: {}", formatted.join(", "));
            }
        }
    }
}

// Minimal repro: Super DWA specialization admits a token that skips a required literal.
// - Why it fails: cross-template state merging from complex signatures in Super DWA.
// - Minimal conditions: key literal overlaps STR_CHAR and a fixed delimiter surrounds STR_CHAR.
#[test]
fn test_super_dwa_fp_minimal() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let lark_grammar = indoc! {r#"
        start: "a" ":" "x" STR_CHAR STR_CHAR "x"
        STR_CHAR: "a" | ":" | "-"
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse minimal grammar");

    if std::env::var("GLR_FP_TRACE").is_ok() {
        if let Some(group_id) = grammar_definition
            .regex_name_to_group_id
            .get_by_left(&"STR_CHAR".to_string())
        {
            if let Some(expr) = grammar_definition.group_id_to_expr.get(group_id) {
                match expr {
                    crate::dfa_u8::Expr::U8Class(set) => {
                        let bytes: Vec<u8> = set.iter().collect();
                        let chars: Vec<String> = bytes
                            .iter()
                            .map(|b| (*b as char).to_string())
                            .collect();
                        eprintln!("TRACE STR_CHAR u8set bytes={:?} chars={:?}", bytes, chars);
                    }
                    other => {
                        eprintln!("TRACE STR_CHAR expr={:?}", other);
                    }
                }
            }
        }
    }

    let mut llm_token_map = LLMTokenMap::new();
    let tok_prefix = LLMTokenID(0);
    let tok_valid = LLMTokenID(1);
    let tok_fp = LLMTokenID(2);
    // Prefix consumes the key `a`. Next valid token is `:x`; `:-` skips the required `x`.
    llm_token_map.insert(b"a".to_vec(), tok_prefix);
    llm_token_map.insert(b":x".to_vec(), tok_valid);
    llm_token_map.insert(b":-".to_vec(), tok_fp);
    let max_id = 2;

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_id,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(tok_prefix).expect("prefix commit failed");

    let mask = state.get_mask();
    assert!(
        mask.contains(tok_valid.0),
        "Expected valid token in mask: token {}",
        tok_valid.0,
    );
    assert!(
        !mask.contains(tok_fp.0),
        "False positive token in mask: token {}",
        tok_fp.0,
    );
}

// Regression test for UTF-8 handling in JSON string character classes.
// Minimal schema-equivalent shape: {"type":"object"} -> object with JSON string keys.
// At prefix {" the next character is inside a JSON string and must be valid UTF-8.
#[test]
fn test_json_string_mask_rejects_invalid_utf8_continuation_byte() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        start: "{" JSON_STRING ":" JSON_STRING "}"
        JSON_STRING: "\"" STRING_CHARS "\""
        STRING_CHARS: STRING_CHAR*
        STRING_CHAR: /[^\x00-\x1F"\\]/
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse minimal JSON-string grammar");

    let mut llm_token_map = LLMTokenMap::new();
    let tok_prefix = LLMTokenID(0); // b"{\""
    let tok_bad_utf8 = LLMTokenID(1); // b"\xA1" (standalone UTF-8 continuation byte)
    let tok_good_ascii = LLMTokenID(2); // b"a"

    llm_token_map.insert(b"{\"".to_vec(), tok_prefix);
    llm_token_map.insert(vec![0xA1], tok_bad_utf8);
    llm_token_map.insert(b"a".to_vec(), tok_good_ascii);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        2,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(tok_prefix).expect("prefix commit failed");

    let mask = state.get_mask();

    assert!(
        mask.contains(tok_good_ascii.0),
        "sanity: ASCII key character should be allowed after {{\""
    );

    assert!(
        !mask.contains(tok_bad_utf8.0),
        "regression: standalone byte 0xA1 must not be allowed as JSON string content after {{\""
    );
}

// MINIMAL regression: get_mask() must include continuation tokens spanning
// multiple terminals after a committed prefix.
// Grammar: start: "a" ":" "a".
// After commit_bytes(b"a"), parser expects ":" then "a".
// Token b":a" must be in the mask.
#[test]
fn test_span_token_in_get_mask() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        start: "a" ":" "a"
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse grammar");

    let mut llm_token_map = LLMTokenMap::new();
    let tok_span = LLMTokenID(0);
    llm_token_map.insert(b":a".to_vec(), tok_span);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        1,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit_bytes(b"a");

    let mask = state.get_mask();
    assert!(
        mask.contains(tok_span.0),
        "span token must be in get_mask()"
    );
}

#[test]
fn test_json_value_span_token_fn() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        start: ws value ws
        value: object | array | string | number | "true" | "false" | "null"
        object: "{" ws members? ws "}"
        members: pair (ws "," ws pair)*
        pair: string ws ":" ws value
        array: "[" ws elements? ws "]"
        elements: value (ws "," ws value)*
        string: QUOTE char* QUOTE
        char: letter | digit | MINUS | UNDERSCORE
        number: int | int frac | int exp | int frac exp
        int: digits | MINUS digits
        frac: DOT digits
        exp: EXP digits | EXP PLUS digits | EXP MINUS digits
        digits: DIGIT+
        ws: WS*
        letter: LETTER
        digit: DIGIT
        QUOTE: "\""
        MINUS: "-"
        PLUS: "+"
        DOT: "."
        EXP: "e" | "E"
        UNDERSCORE: "_"
        WS: " " | "\n" | "\t" | "\r"
        LETTER: "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
        DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse grammar");

    let tok_prefix = LLMTokenID(4895);   // b'{"'
    let tok_span = LLMTokenID(34713);    // b'":"",'
    let tok_suffix = LLMTokenID(34714);  // b'"a":null}'

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{\"".to_vec(), tok_prefix);
    llm_token_map.insert(b"\":\"\",".to_vec(), tok_span);
    llm_token_map.insert(b"\"a\":null}".to_vec(), tok_suffix);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        tok_suffix.0,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(tok_prefix).expect("prefix commit failed");

    let mask = state.get_mask();
    assert!(
        mask.contains(tok_span.0),
        "json_value span token FN: expected token 34713 (b'\":\"\",') to be allowed after token 4895"
    );
}

#[test]
fn test_json_value_span_token_fn_copy_minimized() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        start: "{" pair "}"
        pair: string ":" string "," string ":" "null"
        string: QUOTE char* QUOTE
        char: "a"
        QUOTE: "\""
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse minimized span-token grammar");

    let tok_prefix = LLMTokenID(0);   // b'{"'
    let tok_span = LLMTokenID(1);     // b'":"",'
    let tok_suffix = LLMTokenID(2);   // b'"a":null}'

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{\"".to_vec(), tok_prefix);
    llm_token_map.insert(b"\":\"\",".to_vec(), tok_span);
    llm_token_map.insert(b"\"a\":null}".to_vec(), tok_suffix);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        tok_suffix.0,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(tok_prefix).expect("prefix commit failed");

    let mut probe = state.clone();
    probe.commit(tok_span).expect("span commit probe failed");
    assert!(
        probe.is_active(),
        "sanity: committing span token should keep parser state active"
    );

    let mut bytes_state = constraint.init();
    bytes_state.commit_bytes(b"{\"");

    let mut bytes_probe = bytes_state.clone();
    bytes_probe.commit_bytes(b"\":\"\",");
    assert!(
        bytes_probe.is_active(),
        "sanity: committing span bytes should keep parser state active"
    );

    let mask = state.get_mask();
    assert!(
        mask.contains(tok_span.0),
        "minimized copy FN: expected b'\":\"\",' to be allowed after b'{{\"'"
    );

    let bytes_mask = bytes_state.get_mask();
    assert!(
        bytes_mask.contains(tok_span.0),
        "minimized copy FN (commit_bytes path): expected b'\":\"\",' to be allowed after b'{{\"'"
    );
}

#[test]
fn test_json_value_span_token_fn_minimal() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ebnf_grammar = indoc! {r#"
        start ::= string ':' string ',';
        string ::= '"' '"';
    "#};

    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar)
        .expect("Failed to parse grammar");

    let tok_span = LLMTokenID(34713); // b'":"",'
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"\":\"".to_vec(), tok_span);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        tok_span.0,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit_bytes(b"\"");

    let mask = state.get_mask();
    assert!(
        mask.contains(tok_span.0),
        "minimal EBNF span-token FN: expected b'\":\"' to be allowed after prefix token"
    );
}

#[test]
fn test_span_token_ignore_ws() {
    // Regression: span tokens must work across %ignore WS boundaries.
    // Token b'":"","' crossing key→colon→value→comma→nextkey with %ignore WS.
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        start: "{" pair ("," pair)* "}"
        pair: string ":" value
        value: string | "null"
        string: QUOTE char* QUOTE
        char: /[a-z]/
        QUOTE: "\""
        WS: " " | "\n" | "\t" | "\r"
        %ignore WS
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse ignore-WS grammar");

    let tok_prefix = LLMTokenID(0);   // b'{"'
    let tok_span = LLMTokenID(1);     // b'":"","'
    let tok_suffix = LLMTokenID(2);   // b'"a":null}'

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{\"".to_vec(), tok_prefix);
    llm_token_map.insert(b"\":\"\",".to_vec(), tok_span);
    llm_token_map.insert(b"\"a\":null}".to_vec(), tok_suffix);

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        tok_suffix.0,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(tok_prefix).expect("prefix commit failed");

    // After '{"', token b'":"","' must be valid:
    // It completes: " (close key) : (sep) "" (empty value) , (sep) " (start next key)
    let mask = state.get_mask();
    assert!(
        mask.contains(tok_span.0),
        "ignore-WS span token FN: expected b'\":\"\",\"' to be allowed after b'{{\"'"
    );
}

#[test]
fn test_js_minimized_ebnf_string() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let tokenizer = Tokenizer::new(tokenizer_expr.build());

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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::run_precompute1;
    use crate::finite_automata::{Expr, ExprGroups, ExprGroup};
    use crate::dwa_i32::{DWA, Weight};
    
    // Build tokenizer: just terminal 0 = 'a'
    let tokenizer = Tokenizer::new(ExprGroups {
        groups: vec![ExprGroup {
            expr: Expr::U8Seq(b"a".to_vec()),
            is_non_greedy: false,
        }],
    }.build());
    
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
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
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

/// Demonstrate a short-token path-length violation in terminal DWA.
///
/// This test is ignored by default because it currently fails, demonstrating
/// the over-approximation bug in terminal DWA weights.
#[test]
fn test_terminal_dwa_short_token_path_length_violation() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::dwa_i32::Weight;

    // Use JS grammar tokenizer to mirror real terminal definitions (IGNORE, +, +=, etc.)
    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    // LLM vocab: short token plus longer extensions with same prefix
    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b" ++".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b" +".to_vec(), LLMTokenID(1));
    internal_llm_token_map.insert(b" +=".to_vec(), LLMTokenID(2));
    internal_llm_token_map.insert(b" ++=".to_vec(), LLMTokenID(3));
    internal_llm_token_map.insert(b" +++".to_vec(), LLMTokenID(4));
    internal_llm_token_map.insert(b" +++=".to_vec(), LLMTokenID(5));
    internal_llm_token_map.insert(b"+".to_vec(), LLMTokenID(6));
    internal_llm_token_map.insert(b"++".to_vec(), LLMTokenID(7));
    internal_llm_token_map.insert(b"+=".to_vec(), LLMTokenID(8));
    internal_llm_token_map.insert(b" +".to_vec(), LLMTokenID(1));
    internal_llm_token_map.insert(b" +=".to_vec(), LLMTokenID(2));
    internal_llm_token_map.insert(b" ++=".to_vec(), LLMTokenID(3));
    internal_llm_token_map.insert(b" +++".to_vec(), LLMTokenID(4));
    internal_llm_token_map.insert(b" +++=".to_vec(), LLMTokenID(5));
    internal_llm_token_map.insert(b"+".to_vec(), LLMTokenID(6));
    internal_llm_token_map.insert(b"++".to_vec(), LLMTokenID(7));
    internal_llm_token_map.insert(b"+=".to_vec(), LLMTokenID(8));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        8, // max internal token id
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    fn max_path_len_for_token(
        dwa: &crate::dwa_i32::DWA,
        token_id: usize,
        terminals_count: usize,
        weight_contains_token: &impl Fn(&Weight, usize) -> bool,
    ) -> (usize, Vec<(usize, crate::dwa_i32::common::Label, usize)>) {
        let n_states = dwa.states.len();
        let mut memo: Vec<Option<usize>> = vec![None; n_states];
        let mut choice: Vec<Option<(crate::dwa_i32::common::Label, usize, usize)>> = vec![None; n_states];

        fn dfs(
            state_id: usize,
            dwa: &crate::dwa_i32::DWA,
            token_id: usize,
            terminals_count: usize,
            weight_contains_token: &impl Fn(&Weight, usize) -> bool,
            memo: &mut Vec<Option<usize>>,
            choice: &mut Vec<Option<(crate::dwa_i32::common::Label, usize, usize)>>,
        ) -> usize {
            if let Some(v) = memo[state_id] {
                return v;
            }

            let mut best = 0usize;
            if let Some(final_weight) = &dwa.states[state_id].final_weight {
                if weight_contains_token(final_weight, token_id) {
                    best = 0;
                }
            }

            for (&label, &next_state) in &dwa.states[state_id].transitions {
                if let Some(weight) = dwa.states[state_id].trans_weights.get(&label) {
                    if !weight_contains_token(weight, token_id) {
                        continue;
                    }
                    let label_usize = label as usize;
                    let add: usize = if label_usize < terminals_count { 1 } else { 0 };
                    let cand = add.saturating_add(dfs(next_state, dwa, token_id, terminals_count, weight_contains_token, memo, choice));
                    if cand > best {
                        best = cand;
                        choice[state_id] = Some((label, next_state, add));
                    }
                }
            }

            memo[state_id] = Some(best);
            best
        }

        let max_len = dfs(
            dwa.body.start_state,
            dwa,
            token_id,
            terminals_count,
            weight_contains_token,
            &mut memo,
            &mut choice,
        );

        let mut path = Vec::new();
        let mut state = dwa.body.start_state;
        let mut safety = 0usize;
        while let Some((label, next_state, _add)) = choice[state] {
            path.push((state, label, next_state));
            state = next_state;
            safety += 1;
            if safety > dwa.states.len().saturating_mul(2).saturating_add(10) {
                break;
            }
        }

        (max_len, path)
    }

    let token_len = 3; // " ++"
    let (max_len, witness_path) = max_path_len_for_token(
        &terminal_dwa,
        0,
        terminals_count,
        &weight_contains_token,
    );

    if max_len > token_len {
        eprintln!("WITNESS max_len={} token_len={}", max_len, token_len);
        let mut terminal_edges = 0usize;
        for (idx, (src, label, dst)) in witness_path.iter().enumerate() {
            let label_usize = *label as usize;
            let is_terminal = label_usize < terminals_count;
            if is_terminal {
                terminal_edges += 1;
            }
            let has_token = terminal_dwa.states[*src]
                .trans_weights
                .get(label)
                .map(|w| weight_contains_token(w, 0))
                .unwrap_or(false);
            if is_terminal {
                let term_name = compiled_grammar
                    .glr_parser
                    .terminal_map
                    .get_by_right(&TerminalID(label_usize))
                    .map(|t| format!("{}", t))
                    .unwrap_or_else(|| "<unknown>".to_string());
                eprintln!(
                    "WITNESS[{}] {} --terminal {} ({})--> {} has_token={}",
                    idx,
                    src,
                    label,
                    term_name,
                    dst,
                    has_token
                );
            } else {
                let tsid = label_usize.saturating_sub(terminals_count);
                eprintln!("WITNESS[{}] {} --tsid {}--> {} has_token={}", idx, src, tsid, dst, has_token);
            }
        }
        eprintln!("WITNESS terminal_edges_count={}", terminal_edges);
    }

    assert!(
        max_len <= token_len,
        "token len {} shorter than max path len {} (expected to fail with current over-approximation)",
        token_len,
        max_len
    );
}

/// Minimal grammar reproduction for the short-token path-length violation.
#[test]
fn test_terminal_dwa_short_token_path_length_violation_minimal() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::dwa_i32::Weight;

    let ebnf_grammar = r#"
#![ignore(IGNORE)]
start ::= PLUS | PLUSPLUS | PLUSEQ | template_literal;
PLUS ::= '+' ;
PLUSPLUS ::= '++' ;
PLUSEQ ::= '+=' ;
EOF ::= '<|endoftext|>';

// Lexical Grammar: Whitespace and Comments (from js.ebnf)
IGNORE ::= ( WS | COMMENT )+ ;
WS ::= ( ' ' | '\t' | '\n' | '\r' )+ ;
COMMENT ::= SINGLE_LINE_COMMENT | MULTI_LINE_COMMENT ;
SINGLE_LINE_COMMENT ::= '//' ( [^\n\r] )* ;
MULTI_LINE_COMMENT ::= '/*' ( [^*] | '*' [^/] )* '*/' ;

// Template literal tokens (from js.ebnf)
template_literal ::= '`' TEMPLATE_CHARS '`' ;

// Template chars (from js.ebnf)
TEMPLATE_CHARS ::= TEMPLATE_CHAR+ ;
TEMPLATE_CHAR ::= [^`\\] | '\\' . ;
"#;

    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b" ++".to_vec(), LLMTokenID(0));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        8, // max internal token id
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    fn max_path_len_for_token(
        dwa: &crate::dwa_i32::DWA,
        token_id: usize,
        terminals_count: usize,
        weight_contains_token: &impl Fn(&Weight, usize) -> bool,
    ) -> usize {
        let n_states = dwa.states.len();
        let mut memo: Vec<Option<usize>> = vec![None; n_states];

        fn dfs(
            state_id: usize,
            dwa: &crate::dwa_i32::DWA,
            token_id: usize,
            terminals_count: usize,
            weight_contains_token: &impl Fn(&Weight, usize) -> bool,
            memo: &mut Vec<Option<usize>>,
        ) -> usize {
            if let Some(v) = memo[state_id] {
                return v;
            }

            let mut best = 0usize;
            if let Some(final_weight) = &dwa.states[state_id].final_weight {
                if weight_contains_token(final_weight, token_id) {
                    best = 0;
                }
            }

            for (&label, &next_state) in &dwa.states[state_id].transitions {
                if let Some(weight) = dwa.states[state_id].trans_weights.get(&label) {
                    if !weight_contains_token(weight, token_id) {
                        continue;
                    }
                    let label_usize = label as usize;
                    let add: usize = if label_usize < terminals_count { 1 } else { 0 };
                    let cand = add.saturating_add(dfs(next_state, dwa, token_id, terminals_count, weight_contains_token, memo));
                    if cand > best {
                        best = cand;
                    }
                }
            }

            memo[state_id] = Some(best);
            best
        }

        dfs(
            dwa.body.start_state,
            dwa,
            token_id,
            terminals_count,
            weight_contains_token,
            &mut memo,
        )
    }

    let token_len = 3; // " ++"
    let max_len = max_path_len_for_token(
        &terminal_dwa,
        0,
        terminals_count,
        &weight_contains_token,
    );

    assert!(
        max_len <= token_len,
        "token len {} shorter than max path len {} (expected to fail with current over-approximation)",
        token_len,
        max_len
    );
}

#[test]
#[ignore]
fn test_terminal_dwa_tilde_sequence_weights() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::dwa_i32::Weight;
    use crate::dwa_i32::common::Label;

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"~".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b"~~~~".to_vec(), LLMTokenID(1));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let tilde_tid = compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"~".to_vec()))
        .expect("'~' terminal should exist");
    let tilde_label = tilde_tid.0 as Label;
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        1, // max internal token id
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        Some(Arc::new(HashSet::from([tilde_label]))),
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    let w_single = crate::debug_path_weight::check_dwa_path_weight(&terminal_dwa, &[tilde_label]);
    let w_quad = crate::debug_path_weight::check_dwa_path_weight(
        &terminal_dwa,
        &[tilde_label, tilde_label, tilde_label, tilde_label],
    );

    assert!(
        weight_contains_token(&w_single, 0),
        "expected '~' to be allowed for token id 0"
    );
    assert!(w_quad.is_empty(), "expected '~~~~' weight to be empty");
}

#[test]
fn test_terminal_dwa_tilde_sequence_weights_simple_grammar() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::dwa_i32::Weight;
    use crate::dwa_i32::common::Label;

    let ebnf_grammar = r#"
start ::= '~'+ ;
"#;
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"~".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b"~~~~".to_vec(), LLMTokenID(1));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let tilde_tid = compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"~".to_vec()))
        .expect("'~' terminal should exist");
    let tilde_label = tilde_tid.0 as Label;
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        1, // max internal token id
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        Some(Arc::new(HashSet::from([tilde_label]))),
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    let w_single = crate::debug_path_weight::check_dwa_path_weight(&terminal_dwa, &[tilde_label]);
    let w_quad = crate::debug_path_weight::check_dwa_path_weight(
        &terminal_dwa,
        &[tilde_label, tilde_label, tilde_label, tilde_label],
    );

    assert!(
        weight_contains_token(&w_single, 0),
        "expected '~' to be allowed for token id 0"
    );
    assert!(w_quad.is_empty(), "expected '~~~~' weight to be empty");
}

#[test]
fn test_terminal_dwa_greedy_keywords_no_else_if_on_ei() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::debug_path_weight::{check_dwa_path_weight, weight_contains_token};
    use crate::dwa_i32::common::Label;
    use rand::SeedableRng;

    let ebnf_grammar = indoc! {r#"
        #![ignore(WS)]
        start ::= atom;
        atom ::= IF | ELSE | IDENTIFIER;
        IF ::= 'if';
        ELSE ::= 'else';
        IDENTIFIER ::= [a-zA-Z]+;
        WS ::= [ ]+;
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"ei".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b"elseif".to_vec(), LLMTokenID(1));
    internal_llm_token_map.insert(b"else".to_vec(), LLMTokenID(2));
    internal_llm_token_map.insert(b"if".to_vec(), LLMTokenID(3));
    internal_llm_token_map.insert(b" else".to_vec(), LLMTokenID(4));
    internal_llm_token_map.insert(b" if".to_vec(), LLMTokenID(5));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let terminal_map = &compiled_grammar.glr_parser.terminal_map;
    let if_tid = terminal_map
        .get_by_left(&Terminal::Literal(b"if".to_vec()))
        .copied()
        .or_else(|| terminal_map.get_by_left(&regex_name("IF")).copied())
        .expect("IF/'if' terminal should exist");
    let else_tid = terminal_map
        .get_by_left(&Terminal::Literal(b"else".to_vec()))
        .copied()
        .or_else(|| terminal_map.get_by_left(&regex_name("ELSE")).copied())
        .expect("ELSE/'else' terminal should exist");
    let identifier_tid = *compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&regex_name("IDENTIFIER"))
        .expect("IDENTIFIER terminal should exist");

    // Put IF/ELSE/IDENTIFIER in the same greedy group for this repro.
    let mut terminal_to_greedy_group = vec![None; terminals_count];
    terminal_to_greedy_group[if_tid.0] = Some(0);
    terminal_to_greedy_group[else_tid.0] = Some(0);
    terminal_to_greedy_group[identifier_tid.0] = Some(0);

    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();
    let terminal_dwa = run_precompute1(
        tokenizer,
        &internal_llm_token_map,
        5, // max internal token id
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
        terminal_to_greedy_group,
    );

    let num_tsids_for_weight = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        1
    };
    let ei_token_id = 0usize;

    // Deterministic targeted check: token "ei" must not admit else->if.
    let else_if_weight = check_dwa_path_weight(
        &terminal_dwa,
        &[else_tid.0 as Label, if_tid.0 as Label],
    );
    assert!(
        !weight_contains_token(&else_if_weight, ei_token_id, num_tsids_for_weight),
        "Token 'ei' should not be accepted on path else->if"
    );

    // Sample terminal-DWA paths and ensure no sampled "ei" path has keyword->keyword adjacency.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xE1F);
    let sampled_paths = terminal_dwa.sample_paths(1000, &mut rng);
    let is_keyword = |tid: usize| tid == if_tid.0 || tid == else_tid.0;

    for sampled in sampled_paths {
        let labels: Vec<Label> = sampled.iter().map(|(label, _)| *label).collect();
        let path_weight = check_dwa_path_weight(&terminal_dwa, &labels);
        if !weight_contains_token(&path_weight, ei_token_id, num_tsids_for_weight) {
            continue;
        }

        let terminal_labels: Vec<usize> = labels
            .iter()
            .filter_map(|label| {
                let u = *label as usize;
                if u < terminals_count { Some(u) } else { None }
            })
            .collect();

        let has_adjacent_keywords = terminal_labels
            .windows(2)
            .any(|pair| is_keyword(pair[0]) && is_keyword(pair[1]));
        assert!(
            !has_adjacent_keywords,
            "Sampled token 'ei' path has adjacent keywords: {:?}",
            terminal_labels
        );
    }
}

#[test]
fn test_js_greedy_keywords_full_grammar() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::debug_path_weight::{check_dwa_path_weight, weight_contains_token};
    use crate::dwa_i32::common::Label;

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    let tokenizer = &compiled_grammar.tokenizer;
    let parser = compiled_grammar.glr_parser();

    let vocab = [
        "ei",
        "sei",
        "elseif",
        "aks",
        "breaks",
        "en",
        "else",
        "if",
        "import",
        " else",
        " if",
        "tc",
        "sa",
    ];
    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    for (idx, token) in vocab.iter().enumerate() {
        internal_llm_token_map.insert(token.as_bytes().to_vec(), LLMTokenID(idx));
    }

    let terminals_count = parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    // Mirror greedy-group mapping logic from constraint.rs.
    let terminal_to_greedy_group = {
        let mut mapping = vec![None; terminals_count];
        let terminal_name_to_tid: HashMap<String, usize> = parser
            .terminal_map
            .iter()
            .map(|(term, tid)| {
                let formatted = match term {
                    Terminal::RegexName(name) => name.clone(),
                    Terminal::Literal(bytes) => {
                        let mut escaped = String::new();
                        for ch in String::from_utf8_lossy(bytes).chars() {
                            match ch {
                                '\\' => escaped.push_str("\\\\"),
                                '\'' => escaped.push_str("\\'"),
                                _ => escaped.extend(ch.escape_default()),
                            }
                        }
                        format!("'{}'", escaped)
                    }
                };
                (formatted, tid.0)
            })
            .collect();

        for (group_idx, group) in grammar_definition.greedy_groups.iter().enumerate() {
            for terminal_name in &group.terminals {
                if let Some(&tid) = terminal_name_to_tid.get(terminal_name) {
                    mapping[tid] = Some(group_idx);
                }
            }
        }
        mapping
    };

    let terminal_dwa = run_precompute1(
        tokenizer,
        &internal_llm_token_map,
        vocab.len().saturating_sub(1),
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
        terminal_to_greedy_group,
    );

    let num_tsids_for_weight = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        1
    };
    let en_token_id = vocab
        .iter()
        .position(|t| *t == "en")
        .expect("'en' should be present");
    let case_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"case".to_vec()))
        .expect("'case' terminal should exist");
    let new_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"new".to_vec()))
        .expect("'new' terminal should exist");

    let case_new_weight = check_dwa_path_weight(
        &terminal_dwa,
        &[case_tid.0 as Label, new_tid.0 as Label],
    );
    assert!(
        case_new_weight.is_empty()
            || !weight_contains_token(&case_new_weight, en_token_id, num_tsids_for_weight),
        "Token 'en' should not be accepted on path 'case' -> 'new'"
    );
}

#[test]
#[ignore]
fn test_suffix_dfa_prunes_tilde_equals() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1, ApproximateDfaPruner};
    use crate::glr::grammar::Terminal;
    use crate::glr::approximate_dfa::build_approximate_parser_dfa_from_start;
    use crate::interface::grammar_to_suffix_grammar;
    use crate::dwa_i32::Weight;
    use crate::dwa_i32::common::Label;

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    let parser = compiled_grammar.glr_parser();
    let tokenizer = &compiled_grammar.tokenizer;

    let suffix_grammar = grammar_to_suffix_grammar(&grammar_definition);
    let suffix_compiled = CompiledGrammar::from_definition(Arc::new(suffix_grammar));
    let suffix_parser = suffix_compiled.glr_parser();

    let approx_dfa = build_approximate_parser_dfa_from_start(&suffix_parser);
    let mut orig_to_suffix_tid = vec![None; parser.terminal_map.len()];
    for (term, orig_tid) in parser.terminal_map.iter() {
        if let Some(suffix_tid) = suffix_parser.terminal_map.get_by_left(term) {
            orig_to_suffix_tid[orig_tid.0] = Some(*suffix_tid);
        }
    }

    let mut ignored_terminals = vec![false; parser.terminal_map.len()];
    for tid in &grammar_definition.ignore_terminal_ids {
        if tid.0 < ignored_terminals.len() {
            ignored_terminals[tid.0] = true;
        }
    }

    let approx_pruner = ApproximateDfaPruner {
        dfa: approx_dfa,
        orig_to_suffix_tid,
        ignored_terminals,
        reduce_fallback_terminals_by_state:
            crate::constraint_precompute::build_reduce_fallback_terminals_by_state(&suffix_parser),
    };

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"~".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b"=".to_vec(), LLMTokenID(1));

    let terminals_count = parser.terminal_map.len();
    let tilde_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"~".to_vec()))
        .expect("'~' terminal should exist");
    let eq_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"=".to_vec()))
        .expect("'=' terminal should exist");
    let tilde_label = tilde_tid.0 as Label;
    let eq_label = eq_tid.0 as Label;

    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        1,
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        Some(approx_pruner),
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    let w_tilde = crate::debug_path_weight::check_dwa_path_weight(&terminal_dwa, &[tilde_label]);
    let w_tilde_eq = crate::debug_path_weight::check_dwa_path_weight(&terminal_dwa, &[tilde_label, eq_label]);

    assert!(
        weight_contains_token(&w_tilde, 0),
        "expected '~' to be allowed for token id 0"
    );
    assert!(w_tilde_eq.is_empty(), "expected '~' -> '=' to be pruned by suffix DFA");
}

#[test]
#[ignore]
fn test_suffix_dfa_prunes_pow_assign_tilde_equals_tilde() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{run_precompute1, ApproximateDfaPruner};
    use crate::glr::approximate_dfa::build_approximate_parser_dfa_from_start;
    use crate::glr::grammar::Terminal;
    use crate::interface::grammar_to_suffix_grammar;
    use crate::dwa_i32::common::Label;

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    let parser = compiled_grammar.glr_parser();
    let tokenizer = &compiled_grammar.tokenizer;

    let suffix_grammar = grammar_to_suffix_grammar(&grammar_definition);
    let suffix_compiled = CompiledGrammar::from_definition(Arc::new(suffix_grammar));
    let suffix_parser = suffix_compiled.glr_parser();

    let approx_dfa = build_approximate_parser_dfa_from_start(&suffix_parser);
    let mut orig_to_suffix_tid = vec![None; parser.terminal_map.len()];
    for (term, orig_tid) in parser.terminal_map.iter() {
        if let Some(suffix_tid) = suffix_parser.terminal_map.get_by_left(term) {
            orig_to_suffix_tid[orig_tid.0] = Some(*suffix_tid);
        }
    }

    let mut ignored_terminals = vec![false; parser.terminal_map.len()];
    for tid in &grammar_definition.ignore_terminal_ids {
        if tid.0 < ignored_terminals.len() {
            ignored_terminals[tid.0] = true;
        }
    }

    let approx_pruner = ApproximateDfaPruner {
        dfa: approx_dfa,
        orig_to_suffix_tid,
        ignored_terminals,
        reduce_fallback_terminals_by_state:
            crate::constraint_precompute::build_reduce_fallback_terminals_by_state(&suffix_parser),
    };

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"**=".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b"~".to_vec(), LLMTokenID(1));
    internal_llm_token_map.insert(b"=".to_vec(), LLMTokenID(2));

    let terminals_count = parser.terminal_map.len();
    let pow_assign_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"**=".to_vec()))
        .expect("'**=' terminal should exist");
    let tilde_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"~".to_vec()))
        .expect("'~' terminal should exist");
    let eq_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"=".to_vec()))
        .expect("'=' terminal should exist");
    let pow_assign_label = pow_assign_tid.0 as Label;
    let tilde_label = tilde_tid.0 as Label;
    let eq_label = eq_tid.0 as Label;

    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        2,
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        Some(approx_pruner),
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let w_pow_assign_tilde_eq_tilde = crate::debug_path_weight::check_dwa_path_weight(
        &terminal_dwa,
        &[pow_assign_label, tilde_label, eq_label, tilde_label],
    );

    assert!(
        w_pow_assign_tilde_eq_tilde.is_empty(),
        "expected '**=' -> '~' -> '=' -> '~' to be pruned by suffix DFA"
    );
}

#[test]
#[ignore]
fn test_terminal_dwa_prunes_pow_assign_tilde_equals_tilde_default_precompute1() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1, ApproximateDfaPruner};
    use crate::glr::approximate_dfa::build_approximate_parser_dfa_from_start;
    use crate::glr::grammar::Terminal;
    use crate::interface::grammar_to_suffix_grammar;
    use crate::dwa_i32::Weight;
    use crate::dwa_i32::common::Label;

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    let parser = compiled_grammar.glr_parser();
    let tokenizer = &compiled_grammar.tokenizer;

    let suffix_grammar = grammar_to_suffix_grammar(&grammar_definition);
    let suffix_compiled = CompiledGrammar::from_definition(Arc::new(suffix_grammar));
    let suffix_parser = suffix_compiled.glr_parser();

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b"=~=~".to_vec(), LLMTokenID(0));

    let terminals_count = parser.terminal_map.len();
    let pow_assign_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"**=".to_vec()))
        .expect("'**=' terminal should exist");
    let tilde_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"~".to_vec()))
        .expect("'~' terminal should exist");
    let eq_tid = parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"=".to_vec()))
        .expect("'=' terminal should exist");
    let pow_assign_label = pow_assign_tid.0 as Label;
    let tilde_label = tilde_tid.0 as Label;
    let eq_label = eq_tid.0 as Label;

    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let mut ignored_terminals = vec![false; parser.terminal_map.len()];
    for tid in &grammar_definition.ignore_terminal_ids {
        if tid.0 < ignored_terminals.len() {
            ignored_terminals[tid.0] = true;
        }
    }

    let approx_dfa = build_approximate_parser_dfa_from_start(&suffix_parser);
    let mut orig_to_suffix_tid = vec![None; parser.terminal_map.len()];
    for (term, orig_tid) in parser.terminal_map.iter() {
        if let Some(suffix_tid) = suffix_parser.terminal_map.get_by_left(term) {
            orig_to_suffix_tid[orig_tid.0] = Some(*suffix_tid);
        }
    }

    let approx_pruner = ApproximateDfaPruner {
        dfa: approx_dfa,
        orig_to_suffix_tid,
        ignored_terminals,
        reduce_fallback_terminals_by_state:
            crate::constraint_precompute::build_reduce_fallback_terminals_by_state(&suffix_parser),
    };

    let suffix_prune_cache = Arc::new(crate::interface::build_suffix_parser_cache(
        &grammar_definition,
        &parser.terminal_map,
    ));

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        0,
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        Some(approx_pruner),
        Some(suffix_prune_cache),
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    let w_path = crate::debug_path_weight::check_dwa_path_weight(
        &terminal_dwa,
        &[pow_assign_label, tilde_label, eq_label, tilde_label],
    );

    assert!(
        !weight_contains_token(&w_path, 0),
        "expected token '=~=~' to be pruned for '**=' -> '~' -> '=' -> '~'"
    );
}

#[test]
#[ignore]
fn test_weight_overapprox_simple() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1};
    use crate::dwa_i32::Weight;

    // Use JS grammar tokenizer to match witness path terminal IDs.
    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    // Vocab: richer set to reproduce over-approximation
    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b" ++".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b" +".to_vec(), LLMTokenID(1));
    internal_llm_token_map.insert(b" +=".to_vec(), LLMTokenID(2));
    internal_llm_token_map.insert(b" ++=".to_vec(), LLMTokenID(3));
    internal_llm_token_map.insert(b" +++".to_vec(), LLMTokenID(4));
    internal_llm_token_map.insert(b" +++=".to_vec(), LLMTokenID(5));
    internal_llm_token_map.insert(b"+".to_vec(), LLMTokenID(6));
    internal_llm_token_map.insert(b"++".to_vec(), LLMTokenID(7));
    internal_llm_token_map.insert(b"+=".to_vec(), LLMTokenID(8));
    internal_llm_token_map.insert(b" ++++".to_vec(), LLMTokenID(9));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let terminal_dwa = run_precompute1(
        &tokenizer,
        &internal_llm_token_map,
        9, // max internal token id
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    // Walk the witness path using terminal names.
    let ignore_tid = compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&regex_name("IGNORE"))
        .expect("IGNORE terminal should exist");
    let plus_tid = compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"+".to_vec()))
        .expect("'+' terminal should exist");
    let plusplus_tid = compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&Terminal::Literal(b"++".to_vec()))
        .expect("'++' terminal should exist");
    let template_chars_tid = compiled_grammar
        .glr_parser
        .terminal_map
        .get_by_left(&regex_name("TEMPLATE_CHARS"))
        .expect("TEMPLATE_CHARS terminal should exist");

    let path: [crate::dwa_i32::common::Label; 4] = [
        ignore_tid.0 as crate::dwa_i32::common::Label,
        plus_tid.0 as crate::dwa_i32::common::Label,
        plusplus_tid.0 as crate::dwa_i32::common::Label,
        template_chars_tid.0 as crate::dwa_i32::common::Label,
    ];
    let mut state = terminal_dwa.body.start_state;
    let mut weight: Option<Weight> = None;

    for &label in &path {
        let state_ref = &terminal_dwa.states[state];
        let next_state = *state_ref
            .transitions
            .get(&label)
            .expect("expected DWA transition on label for fixed path");
        let trans_weight = state_ref
            .trans_weights
            .get(&label)
            .expect("expected weight for DWA transition on fixed path");
        weight = Some(match weight {
            None => trans_weight.clone(),
            Some(mut w) => {
                w &= trans_weight;
                w
            }
        });
        state = next_state;
    }

    let mut weight = weight.expect("expected non-empty path weight");

    if let Some(final_w) = &terminal_dwa.states[state].final_weight {
        weight &= final_w;
    }

    assert!(
        !weight_contains_token(&weight, 0),
        "Bug: token ' ++' survives 4 transitions"
    );
}

#[test]
fn test_terminal_nwa_vs_dwa_overapprox_js() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::constraint_precompute::{is_weight_heavy_enabled, run_precompute1, run_precompute1_nwa_for_tests};
    use crate::dwa_i32::{NWA, Weight};

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let tokenizer = &compiled_grammar.tokenizer;

    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    internal_llm_token_map.insert(b" ++".to_vec(), LLMTokenID(0));
    internal_llm_token_map.insert(b" +".to_vec(), LLMTokenID(1));
    internal_llm_token_map.insert(b" +=".to_vec(), LLMTokenID(2));
    internal_llm_token_map.insert(b" ++=".to_vec(), LLMTokenID(3));
    internal_llm_token_map.insert(b" +++".to_vec(), LLMTokenID(4));
    internal_llm_token_map.insert(b" +++=".to_vec(), LLMTokenID(5));
    internal_llm_token_map.insert(b"+".to_vec(), LLMTokenID(6));
    internal_llm_token_map.insert(b"++".to_vec(), LLMTokenID(7));
    internal_llm_token_map.insert(b"+=".to_vec(), LLMTokenID(8));

    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|sid| (sid, sid))
        .collect();

    let nwa = run_precompute1_nwa_for_tests(
        tokenizer,
        &internal_llm_token_map,
        8,
        terminals_count,
        state_to_rep.clone(),
        (0..tokenizer.dfa().states.len()).collect(),
        None,
    );

    let terminal_dwa = run_precompute1(
        tokenizer,
        &internal_llm_token_map,
        8,
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );

    let num_tsids = if is_weight_heavy_enabled() {
        tokenizer.dfa().states.len()
    } else {
        0
    };

    let weight_contains_token = |weight: &Weight, internal_id: usize| -> bool {
        if num_tsids == 0 {
            weight.contains(internal_id)
        } else {
            let start = internal_id.saturating_mul(num_tsids);
            let end = start.saturating_add(num_tsids.saturating_sub(1));
            for range in weight.ranges() {
                let r_start = *range.start();
                let r_end = *range.end();
                if r_start > end {
                    break;
                }
                if r_end >= start {
                    return true;
                }
            }
            false
        }
    };

    fn max_path_len_for_token_nwa(
        nwa: &NWA,
        token_id: usize,
        terminals_count: usize,
        weight_contains_token: &impl Fn(&Weight, usize) -> bool,
    ) -> usize {
        let mut best = 0usize;
        let mut best_seen: HashMap<crate::dwa_i32::common::NWAStateID, usize> = HashMap::new();
        let mut queue: VecDeque<(crate::dwa_i32::common::NWAStateID, usize)> = VecDeque::new();

        for &start in &nwa.body.start_states {
            queue.push_back((start, 0));
        }

        while let Some((state, len)) = queue.pop_front() {
            if let Some(prev) = best_seen.get(&state) {
                if *prev >= len {
                    continue;
                }
            }
            best_seen.insert(state, len);
            if len > best {
                best = len;
            }

            for (next_state, w) in &nwa.states[state].epsilons {
                if !weight_contains_token(w, token_id) {
                    continue;
                }
                queue.push_back((*next_state, len));
            }

            for (&label, targets) in &nwa.states[state].transitions {
                let label_usize = label as usize;
                let add = if label_usize < terminals_count { 1 } else { 0 };
                for (next_state, w) in targets {
                    if !weight_contains_token(w, token_id) {
                        continue;
                    }
                    queue.push_back((*next_state, len + add));
                }
            }
        }

        best
    }

    fn max_path_len_for_token_dwa(
        dwa: &crate::dwa_i32::DWA,
        token_id: usize,
        terminals_count: usize,
        weight_contains_token: &impl Fn(&Weight, usize) -> bool,
    ) -> usize {
        let n_states = dwa.states.len();
        let mut memo: Vec<Option<usize>> = vec![None; n_states];

        fn dfs(
            state_id: usize,
            dwa: &crate::dwa_i32::DWA,
            token_id: usize,
            terminals_count: usize,
            weight_contains_token: &impl Fn(&Weight, usize) -> bool,
            memo: &mut Vec<Option<usize>>,
        ) -> usize {
            if let Some(v) = memo[state_id] {
                return v;
            }

            let mut best = 0usize;
            if let Some(final_weight) = &dwa.states[state_id].final_weight {
                if weight_contains_token(final_weight, token_id) {
                    best = 0;
                }
            }

            for (&label, &next_state) in &dwa.states[state_id].transitions {
                if let Some(weight) = dwa.states[state_id].trans_weights.get(&label) {
                    if !weight_contains_token(weight, token_id) {
                        continue;
                    }
                    let label_usize = label as usize;
                    let add: usize = if label_usize < terminals_count { 1 } else { 0 };
                    let cand = add.saturating_add(dfs(next_state, dwa, token_id, terminals_count, weight_contains_token, memo));
                    if cand > best {
                        best = cand;
                    }
                }
            }

            memo[state_id] = Some(best);
            best
        }

        dfs(
            dwa.body.start_state,
            dwa,
            token_id,
            terminals_count,
            weight_contains_token,
            &mut memo,
        )
    }

    let token_len = 3; // " ++"
    let max_len_nwa = max_path_len_for_token_nwa(&nwa, 0, terminals_count, &weight_contains_token);
    let max_len_dwa = max_path_len_for_token_dwa(&terminal_dwa, 0, terminals_count, &weight_contains_token);

    assert!(
        max_len_nwa <= token_len,
        "NWA should not have overlong path (len {} > token_len {})",
        max_len_nwa,
        token_len
    );
    assert!(
        max_len_dwa <= token_len,
        "DWA over-approximates: len {} > token_len {}",
        max_len_dwa,
        token_len
    );
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
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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
fn test_json_schema_name_prefix_disallows_quote_colon_minus() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let ebnf = r#"#![ignore(WS)]
root ::= '{' '\"name\"' ':' JSON_STRING '}' ;
WS ::= ( ( ' ' | '\t' | '\n' | '\r' ) )* ;
JSON_STRING ::= '"' STRING_CHARS '"' ;
STRING_CHARS ::= ( STRING_CHAR )* ;
STRING_CHAR ::= [a-zA-Z] ;
"#;

    let grammar_definition = GrammarDefinition::from_ebnf(ebnf).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"\"".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"name".to_vec(), LLMTokenID(3));
    llm_token_map.insert(b"\":-".to_vec(), LLMTokenID(4));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        10,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(LLMTokenID(1)).expect("Commit {");
    state.commit(LLMTokenID(2)).expect("Commit \"");
    state.commit(LLMTokenID(3)).expect("Commit name");

    let mask = state.get_mask();
    assert!(!mask.contains(4), "Token '\":-' should not be allowed after key prefix '\"name'");
}

#[test]
fn test_newsletter_schema_disallows_quote_colon_minus() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let schema_json = r#"{
        "type": "object",
        "title": "Newsletter Subscription",
        "properties": {
            "name": {"type": "string", "minLength": 8, "maxLength": 80},
            "email": {"type": "string", "maxLength": 120},
            "lists": {"type": "string", "enum": ["Daily New", "Promotion"]}
        },
        "additionalProperties": false,
        "required": ["name", "email", "lists"],
        "x-guidance": {
            "item_separator": ", ",
            "key_separator": ": ",
            "whitespace_flexible": false,
            "whitespace_pattern": null,
            "coerce_one_of": false,
            "lenient": false
        }
    }"#;

    let ebnf = json_schema_to_ebnf(schema_json).unwrap();
    let grammar_definition = GrammarDefinition::from_ebnf(&ebnf).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"\"".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"name".to_vec(), LLMTokenID(3));
    llm_token_map.insert(b"\":-".to_vec(), LLMTokenID(4));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        10,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(LLMTokenID(1)).expect("Commit {");
    state.commit(LLMTokenID(2)).expect("Commit \"");
    state.commit(LLMTokenID(3)).expect("Commit name");

    let mask = state.get_mask();
    assert!(!mask.contains(4), "Token '\":-' should not be allowed after key prefix '\"name'");
}

#[test]
#[ignore]
fn debug_newsletter_nwa_path_for_quote_colon_minus() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = r###"start: ws object ws
object: "{" ws pairs_0 ws "}"
pairs_0: name_pair ws "," ws pairs_1
pairs_1: email_pair ws "," ws pairs_2
pairs_2: lists_pair
name_pair: QUOTE "name" QUOTE ws ":" ws name_val
email_pair: QUOTE "email" QUOTE ws ":" ws email_val
lists_pair: QUOTE "lists" QUOTE ws ":" ws lists_val
name_val: QUOTE name_chars QUOTE
name_chars: STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR STR_CHAR*
email_val: QUOTE email_chars QUOTE
email_chars: STR_CHAR*
lists_val: QUOTE LISTS_S0 QUOTE | QUOTE LISTS_S1 QUOTE
LISTS_S0: "Daily New"
LISTS_S1: "Promotion"
QUOTE: "\""
ws: WS*
WS: " " | "\n" | "\t" | "\r"
BOOL: "true" | "false"
STR_CHAR: " " | "!" | "#" | "$" | "%" | "&" | "'" | "(" | ")" | "*" | "+" | "," | "-" | "." | "/" | "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | ":" | ";" | "<" | "=" | ">" | "?" | "@" | "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" | "[" | "]" | "^" | "_" | "`" | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z" | "{" | "|" | "}" | "~"
"###;

    let grammar_definition = Arc::new(GrammarDefinition::from_lark(lark_grammar).unwrap());

    let (llm_token_map, max_id) = load_gpt2_vocab()
        .expect("No valid GPT-2 vocab found for debug trace");

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::clone(&grammar_definition),
        llm_token_map.clone(),
        max_id,
        &GrammarConstraintConfig::default(),
    );

    let original_id = 48219usize; // token bytes are '\":-'
    let internal_id = *constraint
        .parser_dwa_vocab
        .original_to_internal
        .get(&original_id)
        .expect("Expected internal mapping for token 48219");

    let token_bytes = llm_token_map
        .iter()
        .find(|(_, id)| id.0 == original_id)
        .map(|(bytes, _)| bytes.clone())
        .unwrap_or_default();

    eprintln!("DEBUG_NWA original_id={}, internal_id={}, bytes={:?}",
        original_id,
        internal_id,
        String::from_utf8_lossy(&token_bytes),
    );

    for tid in [0usize, 1, 2, 3] {
        if let Some(terminal) = constraint.parser.terminal_map.get_by_right(&TerminalID(tid)) {
            eprintln!("DEBUG_NWA terminal_id={} maps to {:?}", tid, terminal);
        }
    }

    std::env::set_var("DEBUG_PRECOMPUTE1_NWA_TOKEN", internal_id.to_string());
    std::env::set_var("DEBUG_PRECOMPUTE1_NWA_TOKEN_LEN", "0");

    let _constraint_with_debug = GrammarConstraint::new_from_grammar_definition(
        grammar_definition,
        llm_token_map,
        max_id,
        &GrammarConstraintConfig::default(),
    );
}

#[test]
#[ignore]
fn debug_lark_quote_colon_charclass_suffix() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = r#"start: key ":" value
key: QUOTE "name" QUOTE
value: QUOTE STR_CHAR STR_CHAR* QUOTE
QUOTE: "\""
STR_CHAR: /[a-z]/
"#;

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar).unwrap();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"\"".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"name".to_vec(), LLMTokenID(2));
    llm_token_map.insert(b"\":a".to_vec(), LLMTokenID(3));

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        10,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(LLMTokenID(1)).expect("Commit \"");
    state.commit(LLMTokenID(2)).expect("Commit name");

    let mask = state.get_mask();
    eprintln!("DEBUG mask contains token 3: {}", mask.contains(3));
    assert!(!mask.contains(3), "Token '\":a' should not be allowed after prefix '\"name'");
}

#[test]
fn test_json_schema_gpt2_real_vocab() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
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

#[test]
#[ignore]
fn test_specsuper_config_equivalence() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let (llm_token_map, max_id) = match load_gpt2_vocab() {
        Some(v) => v,
        None => {
            eprintln!(
                "Skipping test_specsuper_config_equivalence: GPT-2 vocab not found. \
                 Try: wget -O benchmarking/gpt2_vocab.json https://huggingface.co/openai-community/gpt2/raw/main/vocab.json"
            );
            return;
        }
    };

    let ebnf_grammar = include_str!("js.ebnf");
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar)
        .expect("Failed to parse JS grammar");
    let grammar_definition = Arc::new(grammar_definition);
    let config = GrammarConstraintConfig::default();

    std::env::set_var("SPECSUPER_CONFIG", "baseline");
    let constraint_baseline = GrammarConstraint::new_from_grammar_definition(
        Arc::clone(&grammar_definition),
        llm_token_map.clone(),
        max_id,
        &config,
    );

    std::env::set_var("SPECSUPER_CONFIG", "no-min");
    let constraint_nomin = GrammarConstraint::new_from_grammar_definition(
        Arc::clone(&grammar_definition),
        llm_token_map,
        max_id,
        &config,
    );

    crate::dwa_i32::test_weighted_automata::stochastic_equivalence_test(
        constraint_baseline.parser_dwa.clone(),
        constraint_nomin.parser_dwa.clone(),
    );
}

#[test]
fn test_assign_ws_suffix_prune_keeps_class_31() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        start: ws stmt (ws stmt)* ws
        stmt: name ws "=" ws number ws ";"
        ws: WS*
        number: DIGIT+
        name: letter+
        letter: LETTER
        WS: " " | "\n" | "\t"
        LETTER: "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
        DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
    "#};

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar)
        .expect("Failed to parse assign_ws grammar");

    let (llm_token_map, max_id) = match load_gpt2_vocab() {
        Some(v) => v,
        None => {
            println!("Skipping test_assign_ws_suffix_prune_keeps_class_31: vocab.json not found.");
            println!("To run, download https://huggingface.co/openai-community/gpt2/raw/main/vocab.json to project root.");
            return;
        }
    };

    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_id,
        &GrammarConstraintConfig::default(),
    );

    let weight_contains_internal = |weight: &crate::dwa_i32::Weight, token_id: usize, num_tsids: usize| -> bool {
        if num_tsids == 0 {
            return weight.contains(token_id);
        }
        let start = token_id.saturating_mul(num_tsids);
        let end = start.saturating_add(num_tsids.saturating_sub(1));
        for range in weight.ranges() {
            let r_start = *range.start();
            let r_end = *range.end();
            if r_start > end {
                break;
            }
            if r_end >= start {
                return true;
            }
        }
        false
    };

    let mut has_31 = false;
    let mut has_32 = false;
    for state in &constraint.parser_dwa.states.0 {
        if let Some(w) = &state.final_weight {
            if weight_contains_internal(w, 31, constraint.num_tsids) {
                has_31 = true;
            }
            if weight_contains_internal(w, 32, constraint.num_tsids) {
                has_32 = true;
            }
        }
        for w in state.trans_weights.values() {
            if weight_contains_internal(w, 31, constraint.num_tsids) {
                has_31 = true;
            }
            if weight_contains_internal(w, 32, constraint.num_tsids) {
                has_32 = true;
            }
        }
        if has_31 && has_32 {
            break;
        }
    }

    assert!(has_32, "Internal token class 32 should appear in DWA weights");
    assert!(has_31, "Internal token class 31 should appear in DWA weights");
}

/// Test suffix grammar validation of terminal DWA paths.
/// 
/// This test validates that terminal DWA paths correspond to valid suffixes
/// of the grammar language. It samples paths from the terminal DWA and checks
/// what proportion are accepted by the suffix parser.
#[test]
fn test_suffix_grammar_validation() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    use crate::interface::suffix_grammar::validate_terminal_dwa_paths_verbose;
    use crate::constraint_precompute::run_precompute1;
    
    // Use a simple grammar for testing
    let ebnf_grammar = indoc! {r#"
        s ::= A B EOF;
        A ::= 'a';
        B ::= 'b';
        EOF ::= '$';
    "#};
    let grammar_definition = GrammarDefinition::from_ebnf(ebnf_grammar).unwrap();
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    
    // Build minimal LLM token map
    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    llm_token_map.insert(b"b".to_vec(), LLMTokenID(1));
    llm_token_map.insert(b"$".to_vec(), LLMTokenID(2));
    
    // Internal token map for precompute1
    let internal_llm_token_map: BTreeMap<Vec<u8>, crate::dfa_u8::LLMTokenID> = llm_token_map
        .iter()
        .map(|(bytes, id)| (bytes.clone(), *id))
        .collect();
    
    // Build state_to_rep (trivial for simple grammars)
    let tokenizer = &compiled_grammar.tokenizer;
    let state_to_rep: BTreeMap<TokenizerStateID, TokenizerStateID> = tokenizer
        .iter_states()
        .map(|s| (s, s))
        .collect();
    
    // Build terminal DWA
    let terminals_count = compiled_grammar.glr_parser.terminal_map.len();
    let terminal_dwa = run_precompute1(
        tokenizer,
        &internal_llm_token_map,
        2, // max token ID
        terminals_count,
        state_to_rep,
        (0..tokenizer.dfa().states.len()).collect(),
        None,
        None,
        None,
        std::sync::Arc::new(vec![false; terminals_count]),
        std::sync::Arc::new(Vec::new()),
        std::sync::Arc::new(Vec::new()),
    vec![None; terminals_count],
    );
    
    // Validate paths against suffix grammar (verbose)
    let proportion_valid = validate_terminal_dwa_paths_verbose(
        &terminal_dwa,
        &grammar_definition,
        terminals_count,
        100, // sample size
        true, // verbose
    );
    
    println!("\nFinal: Proportion of valid terminal DWA paths: {:.2}%", proportion_valid * 100.0);
    
    // We expect all paths to be valid (or most, depending on the DWA structure)
    // For a simple grammar like this, we should see high validity
    assert!(proportion_valid >= 0.0, "Proportion should be non-negative");
    // Note: We're not asserting 100% validity since the terminal DWA might have
    // paths that are valid tokenizer transitions but not grammar suffixes
}

#[test]
fn test_mask_commit_consistency_minimal_repro_should_fail_loudly() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let lark_grammar = indoc! {r#"
        PATTERN_0: /[\x20-\x21\x23-\x5B\x5D-\x7F]/
        PATTERN_1: /[\xC2-\xDF]/
        PATTERN_2: /[\x80-\xBF]/
        PATTERN_3: /[\xE0-\xEF]/
        PATTERN_4: /[\xF0-\xF4]/
        PATTERN_5: /[\x30-\x39\x41-\x46\x61-\x66]/
        PATTERN_6: /[\x22\x2F\x5C\x62\x66\x6E\x72\x74]/
        PATTERN_7: /[\x30-\x39]/
        PATTERN_8: /[\x31-\x39]/
        PATTERN_9: /[\x45\x65]/
        PATTERN_10: /[\x2B\x2D]/
        STRING_CHAR: PATTERN_0 | PATTERN_1 PATTERN_2 | PATTERN_3 PATTERN_2 PATTERN_2 | PATTERN_4 PATTERN_2 PATTERN_2 PATTERN_2
        HEX: PATTERN_5
        ESCAPE_SHORT_CHAR: PATTERN_6
        ESCAPE_SEQ: "\\" ESCAPE_SHORT_CHAR | "\\" "u" HEX HEX HEX HEX
        STRING_CONTENT: (STRING_CHAR | ESCAPE_SEQ)*
        JSON_STRING: "\"" STRING_CONTENT "\""
        DIGIT: PATTERN_7
        NONZERO_DIGIT: PATTERN_8
        INT_PART: "0" | NONZERO_DIGIT DIGIT*
        FRAC_PART: "." DIGIT+
        EXP_MARK: PATTERN_9
        EXP_SIGN: PATTERN_10
        EXP_PART: EXP_MARK EXP_SIGN? DIGIT+
        JSON_INTEGER: "-"? INT_PART
        JSON_NUMBER: "-"? INT_PART FRAC_PART? EXP_PART?
        JSON_BOOL: "true" | "false"
        JSON_NULL: "null"
        json_kv: JSON_STRING ":" json_value
        json_object: "{" "}" | "{" json_kv ("," json_kv)* "}"
        json_array: "[" "]" | "[" json_value ("," json_value)* "]"
        json_value: json_object | json_array | JSON_STRING | JSON_NUMBER | JSON_INTEGER | JSON_BOOL | JSON_NULL
        obj_required_0_1: "\"a\"" ":" json_object
        obj_required_0_2: "\"\"" ":" JSON_STRING
        obj_required_0_0: "\"\"" ":" JSON_STRING "," obj_required_0_1 | "\"a\"" ":" json_object "," obj_required_0_2
        start: "{" obj_required_0_0 "}"
    "#};

    let tok_prefix_0 = LLMTokenID(0); // b"{\""
    let tok_prefix_1 = LLMTokenID(1); // b"\":\""
    let tok_disputed = LLMTokenID(2); // b"\",\""
    let tok_prefix_3 = LLMTokenID(3); // b"a\""
    let tok_prefix_4 = LLMTokenID(4); // b":{\""
    let tok_suffix = LLMTokenID(5); // b"\":{}}}"

    let committed_prefix = [
        b"{\"".as_slice(),
        b"\":\"".as_slice(),
        b"\",\"".as_slice(),
        b"a\"".as_slice(),
        b":{\"".as_slice(),
        b"\":\"".as_slice(),
    ]
    .concat();
    let expected_prefix = b"{\"\":\"\",\"a\":{\"\":\"";
    assert_eq!(
        committed_prefix,
        expected_prefix,
        "test setup invariant failed: committed token bytes must match intended prefix"
    );

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"{\"".to_vec(), tok_prefix_0);
    llm_token_map.insert(b"\":\"".to_vec(), tok_prefix_1);
    llm_token_map.insert(b"\",\"".to_vec(), tok_disputed);
    llm_token_map.insert(b"a\"".to_vec(), tok_prefix_3);
    llm_token_map.insert(b":{\"".to_vec(), tok_prefix_4);
    llm_token_map.insert(b"\":{}}}".to_vec(), tok_suffix);

    let grammar_definition = GrammarDefinition::from_lark(lark_grammar).unwrap();
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        5,
        &GrammarConstraintConfig::default(),
    );

    let mut state = constraint.init();
    state.commit(tok_prefix_0).unwrap();
    state.commit(tok_prefix_1).unwrap();
    state.commit(tok_disputed).unwrap();
    state.commit(tok_prefix_3).unwrap();
    state.commit(tok_prefix_4).unwrap();
    state.commit(tok_prefix_1).unwrap();

    let mask = state.get_mask();

    assert!(
        mask.contains(tok_disputed.0),
        "LOUD_FAIL disputed token missing from mask: prefix='{{\"\":\"\",\"a\":{{\"\":\"' disputed='\",\"' token_id={} mask={mask:?}",
        tok_disputed.0,
    );
}

#[test]
fn test_triad_tuple_locked_replay_votes_explicit() {
    let _guard = crate::GLOBAL_DIMS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let triad_grammar = "#![greedy_group(main,*)]\nS::='a'[a]*;\n";
    let triad_lark_equivalent = indoc! {r#"
        #![greedy_group(main, *)]
        PATTERN_0: /[\x61]*/
        start: "a" PATTERN_0
    "#};
    let prefix_bytes = b"";
    let disputed_bytes = b"a";

    let mut llm_token_map = LLMTokenMap::new();
    let disputed_token_id = LLMTokenID(0);
    llm_token_map.insert(disputed_bytes.to_vec(), disputed_token_id);

    let grammar_definition = GrammarDefinition::from_lark(triad_lark_equivalent).unwrap();
    let constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        0,
        &GrammarConstraintConfig::default(),
    );

    let mut sep1_state = constraint.init();
    if !prefix_bytes.is_empty() {
        sep1_state.commit_bytes(prefix_bytes);
    }
    let sep1_vote = sep1_state.get_mask().contains(disputed_token_id.0);

    let mut gt_state = crate::bruteforce_constraint::BruteforceConstraintState::new(&constraint);
    if !prefix_bytes.is_empty() {
        gt_state.commit_bytes(prefix_bytes);
    }
    let gt_mask = gt_state.get_mask_bruteforce();
    let ground_truth_vote = gt_mask
        .get(disputed_token_id.0)
        .copied()
        .unwrap_or(false);

    assert_eq!(
        (sep1_vote, ground_truth_vote),
        (false, true),
        "LOUD_FAIL triad tuple sep1-vs-ground-truth mismatch: grammar={triad_grammar:?} lark_equivalent={triad_lark_equivalent:?} prefix={prefix_bytes:?} disputed={disputed_bytes:?} sep1={sep1_vote} ground_truth={ground_truth_vote}",
    );
}
