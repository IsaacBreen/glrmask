use rand::rngs::StdRng;
use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::eat_u8;
use crate::{choice, choice_fast, groups, seq, seq_fast};
use crate::glr::grammar::{nt, prod, t, NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{assign_non_terminal_ids, assign_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map};
use crate::datastructures::hybrid_bitset::HybridBitset; // Explicitly import HybridBitset
use std::hash::{Hash, Hasher};
use crate::interface::{eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast, eat_any_fast, eat_string_fast, choice_fast, eat_bytestring_fast, repeat1_fast, CompiledGrammar, GrammarDefinition, display_productions}; // Added eat_any_fast, CompiledGrammar
use crate::glr::analyze; // Import the analyze module

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use bimap::BiBTreeMap;
use reqwest::blocking;
use serde_json;
use crate::constraint::GrammarConstraint;
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
use std::collections::HashMap; // For the symbol removal helper


// Use concrete types for merge tests
type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
type TestNodeMerge = Arc<Mutex<TestTrieMerge>>;
// Use simpler types for basic tests
type TestTrieBasic = Trie<&'static str, &'static str, i32>;
type TestNodeBasic = Arc<Mutex<TestTrieBasic>>;

// Use concrete types for EdgeInserter tests
type TestTrieEI = Trie<&'static str, HybridBitset, String>; // Use HybridBitset here
type TestNodeEI = Arc<Mutex<TestTrieEI>>;

// Helper to get Arc pointer for tests
fn arc_ptr<N>(arc: &Arc<Mutex<N>>) -> *const Mutex<N> {
    Arc::as_ptr(arc)
}

// Helper function to load or download GPT-2 vocab
fn load_or_download_gpt2_vocab(
    cache_dir: &Path,
    file_name: &str,
    url: &str,
) -> Result<BTreeMap<String, u32>, Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    let cache_path = cache_dir.join(file_name);

    if cache_path.exists() {
        println!("Loading GPT-2 vocab from cache: {:?}", cache_path);
        let file = File::open(cache_path)?;
        let reader = BufReader::new(file);
        let vocab: BTreeMap<String, u32> = serde_json::from_reader(reader)?;
        Ok(vocab)
    } else {
        println!("Downloading GPT-2 vocab from: {}", url);
        let response = blocking::get(url)?.error_for_status()?;
        let content = response.text()?;

        let mut file = File::create(&cache_path)?;
        file.write_all(content.as_bytes())?;
        println!("Saved GPT-2 vocab to cache: {:?}", cache_path);

        let vocab: BTreeMap<String, u32> = serde_json::from_str(&content)?;
        Ok(vocab)
    }
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
    grammar_token_map.insert(Terminal("A".to_string()), TerminalID(0)); // Corresponds to eat_u8(b'a')
    grammar_token_map.insert(Terminal("AB".to_string()), TerminalID(1)); // Corresponds to seq![eat_u8(b'a'), eat_u8(b'b')]
    grammar_token_map.insert(Terminal("B_OR_C".to_string()), TerminalID(2)); // Corresponds to choice![eat_u8(b'b'), eat_u8(b'c')]
    grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(3)); // Corresponds to eat_u8(b'$')

    let productions = vec![
        prod("S", vec![nt("X"), t("EOF")]), // S -> X $
        prod("X", vec![t("A"), t("B_OR_C")]), // X -> a (b|c)
        prod("X", vec![t("AB")]),             // X -> ab
    ];

    let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map.clone());
    dbg!(&parser);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.0.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map,
        token_name_map,
        3, // max_llm_token_id should be 3 for 0, 1, 2
    );
    constraint.dump_precomputed(); // Commented out dump for cleaner test output

    let mut constraint_state = constraint.init();

    constraint_state.step_with_all_llm_tokens();

    // Initially, we can match "a" (part of "ab" or "ac") or "ab".
    // "a" leads to expecting "b" or "c".
    // "ab" leads to expecting "$".
    let mask = constraint_state.get_mask();
    println!("Initial mask: {:?}", mask);
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1])); // Expect "ab" or "ac"

    // Commit "ab" (LLMTokenID 0)
    constraint_state.commit(LLMTokenID(0));
    assert!(constraint_state.is_active());
    constraint_state.step_with_all_llm_tokens();
    let mask = constraint_state.get_mask();
    assert_eq!(mask, HybridBitset::from_iter(vec![2])); // Expect "$" (EOF)

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method
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
    grammar_token_map.insert(Terminal("PLUS".to_string()), TerminalID(0));
    grammar_token_map.insert(Terminal("TIMES".to_string()), TerminalID(1));
    grammar_token_map.insert(Terminal("LPAREN".to_string()), TerminalID(2));
    grammar_token_map.insert(Terminal("RPAREN".to_string()), TerminalID(3));
    grammar_token_map.insert(Terminal("I".to_string()), TerminalID(4));
    grammar_token_map.insert(Terminal("EOF".to_string()), TerminalID(5));

    let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map.clone()); // Start production is index 6
    println!("Parser: {}", parser);

    let mut token_name_map = BiBTreeMap::new();
     for (term, id) in &grammar_token_map {
        token_name_map.insert(term.0.clone(), id.0);
    }

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map,
        token_name_map,
        7, // max_llm_token_id should be 7 for IDs 0-6
    );
    constraint.dump_precomputed(); // Commented out dump for cleaner test output

    // Initial state and step
    let mut state = constraint.init();
    state.step_with_all_llm_tokens();
    let mask = state.get_mask();
    // Expect LLM tokens that can start an expression: i (0), '(' (3), "(i" (5)
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 3, 5]));

    // Commit "(i"
    state.commit(LLMTokenID(5));
    state.step_with_all_llm_tokens();
    let mask = state.get_mask();
    // Now expect '+', '*', ')', '+i' => IDs 1,2,4,6
    assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 4, 6]));

    // // Commit "(i"
    // state.commit(LLMTokenID(5));
    // state.step_with_all_llm_tokens();
    // state.commit(LLMTokenID(4)); // Assuming ")"
    // state.step_with_all_llm_tokens();
    // let mask = state.get_mask();
    // assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 5, 6, 3])); // Expect '+', '*', '(', '(i', '+i'

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the new assert_eq method
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
    let llm_tokens_slices: Vec<&[u8]> = llm_tokens.iter().map(|token| &token[..]).collect();
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
        &internal_llm_token_map_for_precompute, // Use the manually created internal map
        &BiBTreeMap::new(), // empty name‐map
        internal_llm_token_map_for_precompute.iter().map(|(_, id)| id.0).max().unwrap_or(0),
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
        &internal_llm_token_map_for_precompute, // Use the manually created internal map
        &BiBTreeMap::new(), // empty name‐map
        internal_llm_token_map_for_precompute.iter().map(|(_, id)| id.0).max().unwrap_or(0),
    );
    // print_precomputed(&_precomputed);
    println!("Done precomputing");
}

#[test]
fn test_precompute_with_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define tokenizer: matches anything
    // The tokenizer will have one group (ID 0)
    let tokenizer_expr = groups![
        // tokens["FSTRING_MIDDLE"] = rep(choice([
        //     eat_u8_negation(ord("{")),
        //     eat("{{"),
        // ])),
        repeat0_fast(choice_fast!(eat_u8_negation_fast(b'{'), eat_bytestring_fast(b"{{".to_vec()))),
        eat_string_fast("def"),
    ];
    let tokenizer = tokenizer_expr.build();

    // 2. Load LLM tokens from GPT-2 vocab.json
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let cache_dir = Path::new(".cache/test_vocabs");
    let vocab_file_name = "gpt2_vocab.json";

    let gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_llm_token_id_val: u32 = 0;

    // Sample GPT-2 tokens to speed up this test
    let prop = 1.0;
    // let prop = 0.15;
    let total_tokens = gpt2_raw_vocab.len();
    let sample_size = (total_tokens as f64 * prop) as usize; // Changed 64 to 66 to introduce a compile error
    println!("Sampling {} out of {} GPT-2 tokens for precompute", sample_size, total_tokens);
    for (token_str, id_val) in gpt2_raw_vocab.into_iter().take(sample_size) {
        llm_token_map.insert(token_str.into_bytes(), LLMTokenID(id_val as usize));
        if id_val > max_llm_token_id_val {
            max_llm_token_id_val = id_val;
        }
    }

    // 2. Map the LLM tokens
    let mut internal_llm_token_map = GrammarConstraint::setup_llm_token_mappings(&llm_token_map);
    let internal_token_name_map: BiBTreeMap<Vec<u8>, LLMTokenID> = llm_token_map.iter().map(|(bytes, id)| (bytes.clone(), *id)).collect();

    // 3. Create token_name_map for grammar tokens
    // Our tokenizer has one grammar token (GroupID 0)
    let mut token_name_map: BiBTreeMap<String, usize> = BiBTreeMap::new();
    token_name_map.insert("FSTRING_MIDDLE".to_string(), 0 as usize);
    token_name_map.insert("DEF".to_string(), 1 as usize);

    // 4. Call precompute
    println!(
        "Starting precompute with GPT-2 vocab ({} tokens, max_original_id_val: {})...",
        llm_token_map.len(),
        max_llm_token_id_val, // Max original ID value encountered
    );

    // // This is the main part of the test: ensure it runs without error.
    // let _precomputed = GrammarConstraint::precompute(
    //     &tokenizer,
    //     &internal_token_name_map,
    //     &token_name_map,
    //     internal_llm_token_map.iter().map(|(_, id)| *id).max().unwrap_or(0),
    // );
    //
    // println!("Successfully precomputed with GPT-2 vocab.");

    // 2. Create a parser
    let productions = vec![
        prod("S'", vec![nt("S")]), // Start
        prod("S", vec![nt("A"), t("DEF")]),
        // prod("S", vec![t("FSTRING_MIDDLE"), t("FSTRING_MIDDLE")]),
        // prod("S", vec![t("FSTRING_MIDDLE")]),
        // prod("S", vec![t("FSTRING_MIDDLE"), t("DEF")]),
        prod("A", vec![]),
    ];
    let terminal_map: BiBTreeMap<Terminal, TerminalID> = token_name_map.iter().map(|(name, id)| (Terminal(name.clone()), TerminalID(*id))).collect();
    let parser = generate_glr_parser(&productions, 0);

    // Ensure that "def" is a valid initial LLM token
    let max_llm_token_id = token_name_map.iter().map(|(_, id)| *id).max().unwrap_or(0);
    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map,
        max_llm_token_id,
    );
    let mut constraint_state = constraint.init();
    constraint_state.step_with_all_llm_tokens();

    let mask = constraint_state.get_mask();

    let d_id = llm_token_map.get_by_left(&b"def"[..]).unwrap().0;
    assert!(mask.contains(d_id), "Expected LLM token ID {} to be in mask", d_id);

    // Step and commit the LLM token "a" repeatedly.
    println!("{}", constraint.parser);
    let mut constraint_state = constraint.init();
    let a_id = llm_token_map.get_by_left(&b"a"[..]).unwrap().0;
    for i in 0..10 {
        println!("{}. Stepping with LLM token ID {}", i, a_id);
        constraint_state.step_with_all_llm_tokens();
        constraint_state.commit(LLMTokenID(a_id));
        assert!(constraint_state.is_active(), "Constraint state should be active after committing token {} (ID {})", a_id, a_id);
    }

    Ok(())
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
    token_name_map.insert("FSTRING_MIDDLE".to_string(), 0);

    // 5. Create the Parser
    let parser = generate_glr_parser(&productions, 0);
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
    for i in 0..10000 {
        println!("{}. Stepping with LLM token ID {}", i, a_id);
        constraint_state.step_with_all_llm_tokens();
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
    grammar_token_map.insert(Terminal("DEF_T".to_string()), TerminalID(0));
    // Note: For this minimal test focusing on the initial mask for "def",
    // we don't strictly need an EOF terminal in the grammar or tokenizer if
    // the goal is just to see "def" allowed initially.
    // If the grammar was S -> DEF_T EOF_T, then EOF_T would need a tokenizer group.

    let parser = generate_glr_parser_with_terminal_map(
        &productions,
        0, // start_production_id
        grammar_token_map.clone()
    );

    // 5. Token name map for stats/debugging (maps grammar terminal name to tokenizer group ID)
    let mut token_name_map_for_stats = BiBTreeMap::new();
    token_name_map_for_stats.insert("DEF_T".to_string(), 0);

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
    constraint_state.step_with_all_llm_tokens();

    // 8. Get the initial mask.
    //    In the Python script, get_mask is called *before* any step or commit.
    //    The initial mask should reflect what's possible from the start.
    let mask = constraint_state.get_mask();

    // 9. Define the expected mask.
    //    It should contain the original LLMTokenID for "def".
    let mut expected_mask = HybridBitset::new();
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

#[test]
fn test_constraint_from_serialized_compiled_grammar_and_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define file path for the serialized CompiledGrammar
    let serialized_grammar_path = "src/serialized_compiled_grammar.json";

    println!("Loading CompiledGrammar from: {}", serialized_grammar_path);
    let json_string = match fs::read_to_string(serialized_grammar_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read serialized grammar file '{}': {}", serialized_grammar_path, e);
            eprintln!("Please ensure the file exists and is readable. Skipping this test.");
            return Ok(());
        }
    };

    let json_node = JSONNode::from_json_string(&json_string)?;
    let compiled_grammar = CompiledGrammar::from_json(json_node)?;
    println!("Successfully loaded CompiledGrammar from JSON.");
    println!("{}", compiled_grammar);

    // --- New test section for grammar terminal sequences ---
    println!("\nTesting GLR parser with specific grammar terminal sequences...");

    // Define the sequences of terminal names to test
    let mut test_sequences_str = vec![
        // Sequence 0
        vec!["IGNORE[0][0]", "\"->\""],
        // Sequence 1
        vec!["STRING[0]", "\"->\""],
        // Sequence 2
        vec!["FSTRING_START[0]", "\"->\""],
        // Sequence 3
        vec!["IGNORE[0][0]", "\"...\"", "\"->\""],
        // Sequence 4
        vec!["STRING[0]", "\"...\"", "\"->\""],
        // Sequence 5
        vec!["FSTRING_START[0]", "\"...\"", "\"->\""],
        // Sequence 6
        vec!["IGNORE[0][0]", "\"==\"", "\"->\""],
        // Sequence 7
        vec!["STRING[0]", "\"==\"", "\"->\""],
        // Sequence 8
        vec!["FSTRING_START[0]", "\"==\"", "\"->\""],
        // Sequence 9
        vec!["IGNORE[0][0]", "\"!=\"", "\"->\""],
        // Sequence 10
        vec!["STRING[0]", "\"!=\"", "\"->\""],
        // Sequence 11
        vec!["FSTRING_START[0]", "\"!=\"", "\"->\""],
        // Sequence 12
        vec!["IGNORE[0][0]", "\"<=\"", "\"->\""],
        // Sequence 13
        vec!["STRING[0]", "\"<=\"", "\"->\""],
        // Sequence 14
        vec!["FSTRING_START[0]", "\"<=\"", "\"->\""],
        // Sequence 15
        vec!["IGNORE[0][0]", "\">=\"", "\"->\""],
        // Sequence 16
        vec!["STRING[0]", "\">=\"", "\"->\""],
        // Sequence 17
        vec!["FSTRING_START[0]", "\">=\"", "\"->\""],
        // Sequence 18
        vec!["IGNORE[0][0]", "\"<<\"", "\"->\""],
        // Sequence 19
        vec!["STRING[0]", "\"<<\"", "\"->\""],
        // Sequence 20
        vec!["FSTRING_START[0]", "\"<<\"", "\"->\""],
        // Sequence 21
        vec!["IGNORE[0][0]", "STRING[0]", "\"->\""],
        // Sequence 22
        vec!["STRING[0]", "STRING[0]", "\"->\""],
        // Sequence 23
        vec!["FSTRING_START[0]", "STRING[0]", "\"->\""],
        // Sequence 24
        vec!["IGNORE[0][0]", "FSTRING_START[0]", "\"->\""],
        // Sequence 25
        vec!["STRING[0]", "FSTRING_START[0]", "\"->\""],
        // Sequence 26
        vec!["FSTRING_START[0]", "FSTRING_START[0]", "\"->\""],
        // Sequence 27
        vec!["IGNORE[0][0]", "FSTRING_MIDDLE[0]", "\"->\""],
        // Sequence 28
        vec!["STRING[0]", "FSTRING_MIDDLE[0]", "\"->\""],
        // Sequence 29
        vec!["FSTRING_START[0]", "FSTRING_MIDDLE[0]", "\"->\""],

        vec!["IGNORE[0][0]", "\"->\""],
        vec!["STRING[0]", "\"->\""],
        vec!["IGNORE[0][0]", "\"!=\"", "\"->\""],
        vec!["STRING[0]", "\"!=\"", "\"->\""],
        vec!["IGNORE[0][0]", "\">=\"", "\"->\""],
        vec!["STRING[0]", "\">=\"", "\"->\""],
        vec!["IGNORE[0][0]", "STRING[0]", "\"->\""],
        vec!["STRING[0]", "STRING[0]", "\"->\""],

        // THIS one is important. Actual failure case. Causes goto not found panic.
        vec!["\"...\"", "\";\"", "\"elif\""],
    ];
    test_sequences_str.reverse(); // Reverse the order of the test sequences

    let mut all_sequences_passed = true;

    for i in 0..test_sequences_str.len() {
        // Add a reversed version of the sequence to test
        let mut seq_terminal_names = test_sequences_str[i].clone();
        let last = seq_terminal_names.pop().unwrap();
        seq_terminal_names.reverse();
        seq_terminal_names.push(last);
        test_sequences_str.push(seq_terminal_names);
    }

    for (seq_idx, seq_terminal_names) in test_sequences_str.iter().enumerate() {
        println!("\nTesting sequence {} ({} tokens): '{}'... ", seq_idx, seq_terminal_names.len(), seq_terminal_names.join(" → "));

        let mut terminal_id_sequence = Vec::new();
        let mut current_sequence_token_names_valid = true;

        for token_name in seq_terminal_names {
            if let Some(terminal_id_val) = compiled_grammar.glr_parser.terminal_map.get_by_left(&Terminal(token_name.to_string())) {
                terminal_id_sequence.push(terminal_id_val);
            } else {
                println!(
                    "  Warning: Terminal name '{}' not found in compiled_grammar.token_name_map for sequence {}. This sequence will be skipped.",
                    token_name, seq_idx
                );
                current_sequence_token_names_valid = false;
                all_sequences_passed = false; // Mark as failed due to unknown token
                break;
            }
        }

        if !current_sequence_token_names_valid {
            continue; // Move to the next sequence
        }

        if terminal_id_sequence.is_empty() {
            if seq_terminal_names.is_empty() {
                println!("  Sequence {} is empty by definition, skipping actual parsing test.", seq_idx);
            } else {
                // This case should ideally be caught by the token name check,
                // but as a safeguard if a sequence of valid names somehow results in empty IDs.
                println!(
                    "  Sequence {} ('{}') resulted in an empty TerminalID sequence despite non-empty names. Skipping.",
                    seq_idx, seq_terminal_names.join(" → ")
                );
                all_sequences_passed = false;
            }
            continue;
        }

        // Initialize GLRParserState with a dummy accumulator.
        // For this test, we are focused on the GLR parser's grammar rule processing,
        // not LLM token constraints, so the accumulator's content is not critical.
        let dummy_llm_token_info = crate::constraint::LLMTokenInfo {
            active: HybridBitset::new(), // Empty bitset
            intersection: HybridBitset::new(), // Empty bitset; for PathAccumulator, this might differ from default
        };

        let mut glr_state = compiled_grammar.glr_parser.init_glr_parser_with_acc(dummy_llm_token_info);

        let seq_names_display = seq_terminal_names.join(" → ");
        print!(
            "  Testing sequence {} ({} tokens): '{}'... ",
            seq_idx,
            terminal_id_sequence.len(),
            seq_names_display
        );

        let mut sequence_parse_ok = true;
        for (token_in_seq_idx, &grammar_token_id) in terminal_id_sequence.iter().enumerate() {
            glr_state.step(*grammar_token_id);
            if !glr_state.is_ok() {
                println!(
                    "failed at token #{} ('{}'). Parser no longer in OK state.",
                    token_in_seq_idx + 1,
                    seq_terminal_names[token_in_seq_idx]
                );
                sequence_parse_ok = false;
                all_sequences_passed = false;
                break;
            }
        }

        if sequence_parse_ok {
            // If the loop completed, check the final state.
            // is_ok() should still be true if the sequence is a valid partial parse.
            if glr_state.is_ok() {
                println!("succeeded. Parser remains in OK state.");
            } else {
                // This should ideally not be reached if the loop's check is comprehensive,
                // but included for robustness.
                println!("failed. Parser not in OK state after processing the full sequence.");
                all_sequences_passed = false;
            }
        }
    }

    // assert!(all_sequences_passed, "One or more grammar terminal sequence tests failed. See warnings/errors above.");
    println!("GLR parser testing with specific grammar terminal sequences finished.");

    // --- GLR Parser Fuzz Test ---
    println!("\nStarting GLR parser fuzz test...");
    let num_fuzz_iterations = 1000;
    let max_tokens_per_fuzz_attempt = 50;

    // Re-use dummy_llm_token_info defined earlier for initializing GLRParserState
    let dummy_llm_token_info = crate::constraint::LLMTokenInfo {
        active: HybridBitset::new(),
        intersection: HybridBitset::new(),
    };

    let all_grammar_terminal_ids: Vec<_> = compiled_grammar.glr_parser.terminal_map.right_values().cloned().collect();

    if all_grammar_terminal_ids.is_empty() {
        println!("  Warning: No grammar terminal IDs found in compiled_grammar.glr_parser.terminal_map. Fuzz test will be trivial or skipped.");
    } else {
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..num_fuzz_iterations {
            if i % 100 == 0 { // Log progress
                println!("  Fuzz test iteration {}/{}", i, num_fuzz_iterations);
            }
            let mut glr_state = compiled_grammar.glr_parser.init_glr_parser_with_acc(dummy_llm_token_info.clone());
            
            let num_tokens_this_attempt = rng.gen_range(0..=max_tokens_per_fuzz_attempt);
            let mut current_fuzz_sequence_names: Vec<String> = Vec::new();
            let mut current_fuzz_sequence_ids: Vec<TerminalID> = Vec::new();

            for _ in 0..num_tokens_this_attempt {
                let random_terminal_id = all_grammar_terminal_ids.choose(&mut rng).unwrap();
                // For debugging, you could find the name:
                let token_name = compiled_grammar.glr_parser.terminal_map.get_by_right(random_terminal_id).map(|t| t.0.clone()).unwrap_or_else(|| "UNKNOWN_TOKEN".to_string());
                current_fuzz_sequence_names.push(token_name);
                current_fuzz_sequence_ids.push(*random_terminal_id);
            }
            println!("  Fuzz sequence: {}", current_fuzz_sequence_names.join(" → "));
            for (i, terminal_id) in current_fuzz_sequence_ids.iter().enumerate() {
                // The core of the fuzz test: step and see if it panics.
                // We don't care about glr_state.is_ok() here.
                let seen_so_far: Vec<_> = current_fuzz_sequence_names[..=i].iter().cloned().collect();
                println!("    Stepping with token {}/{}: '{}' (Terminal {}). Seen so far: {:?}", i + 1, num_tokens_this_attempt, current_fuzz_sequence_names[i], terminal_id.0, seen_so_far);
                glr_state.step(*terminal_id);
            }
            // If a panic occurs, the test will fail here.
            // If we wanted to log the sequence that caused a panic, it would require more setup
            // (e.g. std::panic::catch_unwind), but the goal here is just to detect panics.
        }
    }
    println!("GLR parser fuzz test completed ({} iterations).", num_fuzz_iterations);
    // --- End of GLR Parser Fuzz Test ---

    println!("\nLoading GPT-2 vocabulary...");
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let cache_dir = Path::new(".cache/test_vocabs");
    let vocab_file_name = "gpt2_vocab.json";
    let gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;
    // let gpt2_raw_vocab = BTreeMap::from([("________________________________________________________________", 0)]);

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;

    let prop = 1.0; // Use full vocab for this test to ensure token presence
    let total_tokens = gpt2_raw_vocab.len();
    let sample_size = ((total_tokens as f64 * prop) as usize).max(1);
    println!("Sampling {} out of {} GPT-2 tokens for the test.", sample_size, total_tokens);

    for (token_str, id_val_u32) in gpt2_raw_vocab.into_iter().take(sample_size) {
        let id_val = id_val_u32 as usize;
        // Replace 'Ġ' with ' '
        let token_str = token_str.replace("Ġ", " ");
        let token_bytes = token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(id_val));
        if id_val > max_original_llm_token_id_val {
            max_original_llm_token_id_val = id_val;
        }
    }

    if llm_token_map.is_empty() {
        println!("Warning: LLM token map is empty after sampling. Max original ID will be 0.");
    }
    println!("GPT-2 vocab loaded and processed into LLMTokenMap ({} tokens, max_original_id: {}).", llm_token_map.len(), max_original_llm_token_id_val);

    // 4. Construct GrammarConstraint
    let dummy_eof_placeholder = 0;
    println!("Constructing GrammarConstraint...");
    let grammar_constraint = GrammarConstraint::from_compiled_grammar(
        compiled_grammar,
        llm_token_map.clone(),
        dummy_eof_placeholder,
        max_original_llm_token_id_val
    );
    println!("GrammarConstraint constructed successfully.");
    grammar_constraint.dump_precomputed();

    // --- TOKENIZATION AND SEQUENCE TESTING ---

    // Build a VocabPrefixTree from the LLM token map for tokenization
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = grammar_constraint.llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    // The full text to tokenize.
    let full_text_to_tokenize = "from typing import Any";
    // let full_text_to_tokenize = "((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((";
    // let full_text_to_tokenize = "a";

    // Tokenize the full_text_to_tokenize using the VocabPrefixTree
    let mut test_token_sequence_ids = Vec::new();
    // This list will store the actual string content of tokens as produced by the vocab tree, primarily for logging.
    let mut tokenized_strs_for_logging = Vec::new();
    let mut text_to_process = full_text_to_tokenize.as_bytes();

    println!("\nTokenizing '{}' using VocabPrefixTree:", full_text_to_tokenize);
    while !text_to_process.is_empty() {
        match tokenizer_vocab_tree.find_longest_prefix_token(text_to_process) {
            Some((token_id, matched_bytes)) => {
                let matched_str = String::from_utf8_lossy(matched_bytes).to_string();
                println!("  Matched: '{}' (ID {})", matched_str, token_id);

                test_token_sequence_ids.push(LLMTokenID(token_id));
                tokenized_strs_for_logging.push(matched_str); // Store for logging

                text_to_process = &text_to_process[matched_bytes.len()..];
            }
            None => {
                // If the vocab tree cannot tokenize a part of a known-good string,
                // it implies an an issue with the vocab tree or the test string itself
                // relative to the loaded vocabulary.
                panic!(
                    "Failed to tokenize with VocabPrefixTree. No prefix token found for remaining text: {:?}. This might indicate the test string '{}' contains segments not representable by single tokens in the loaded vocabulary, or the vocabulary is missing expected tokens.",
                    String::from_utf8_lossy(text_to_process),
                    full_text_to_tokenize
                );
            }
        }
    }

    if test_token_sequence_ids.is_empty() && !full_text_to_tokenize.is_empty() {
        panic!("VocabPrefixTree failed to produce any tokens for the non-empty input string: '{}'. Check vocabulary content.", full_text_to_tokenize);
    }
    if test_token_sequence_ids.is_empty() && full_text_to_tokenize.is_empty() {
        println!("Input string was empty, and no tokens were produced, which is expected.");
    } else {
        println!("Successfully tokenized input string into {} tokens using VocabPrefixTree.", test_token_sequence_ids.len());
    }

    // 5. Basic Interaction with the GrammarConstraintState
    let mut constraint_state = grammar_constraint.init();
    // Initial step to populate possibilities
    constraint_state.step_with_all_llm_tokens();
    let initial_mask = constraint_state.get_mask();
    println!("\nInitial mask obtained ({} allowed LLM tokens).", initial_mask.iter_bits().count());

    println!("\nStepping through the token sequence with GrammarConstraint:");
    for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
        // Use tokenized_strs_for_logging for logging, as it corresponds to the llm_token_id
        let current_token_str = &tokenized_strs_for_logging[i];
        println!(
            "Processing token {}/{}: '{}' (LLMTokenID({}))",
            i + 1,
            test_token_sequence_ids.len(),
            current_token_str,
            llm_token_id.0
        );

        assert!(
            constraint_state.is_active(),
            "Constraint state should be active before processing token {} ('{}')",
            i + 1, current_token_str
        );

        let step_start = Instant::now();
        constraint_state.step_with_all_llm_tokens();
        let step_duration = step_start.elapsed();
        println!("  step_with_all_llm_tokens took: {:?}", step_duration);
        let current_mask = constraint_state.get_mask();
        println!(
            "  Mask (after step_with_all_llm_tokens) allows {} tokens. Checking for current token LLMTokenID({})...",
            current_mask.iter_bits().count(),
            llm_token_id.0
        );

        assert!(
            current_mask.contains(llm_token_id.0),
            "Expected LLMTokenID({}) for '{}' to be in the mask. Mask (first 100 if many): {:?}",
            llm_token_id.0, current_token_str, current_mask.iter_bits().take(100).collect::<Vec<_>>()
        );
        println!("  LLMTokenID({}) for '{}' is in the mask.", llm_token_id.0, current_token_str);

        let commit_start = Instant::now();
        constraint_state.commit(llm_token_id);
        let commit_duration = commit_start.elapsed();
        println!("  commit LLMTokenID({}) took: {:?}", llm_token_id.0, commit_duration);
        println!("  Committed LLMTokenID({}) for '{}'.", llm_token_id.0, current_token_str);

        assert!(
            constraint_state.is_active(),
            "Constraint state should be active after committing token {} ('{}')",
            i + 1, current_token_str
        );
        println!("  Constraint state is active after commit.");
    }

    println!("\nFinished processing token sequence with GrammarConstraint.");
    if !test_token_sequence_ids.is_empty() { // Only assert if there were tokens to process
        assert!(
            constraint_state.is_active(),
            "Constraint state should still be active after processing the entire sequence."
        );
        println!("Constraint state is active after the full sequence.");
    } else if full_text_to_tokenize.is_empty() {
         println!("Constraint state was not stepped as the input string was empty and produced no tokens.");
    }


    Ok(())
}

// TODO: This test needs to be uncommented and passed once the fix for the panic is in place.
// #[test]
// #[ignore] // Ignore this test until the root cause of the panic is fixed.
// fn test_filtered_grammar_with_specific_sequence() -> Result<(), Box<dyn std::error::Error>> {
//     // 1. Load the serialized CompiledGrammar
//     let serialized_grammar_path = "src/serialized_compiled_grammar.json";
//     println!("[Test] Loading CompiledGrammar from: {}", serialized_grammar_path);
//     let json_string = match fs::read_to_string(serialized_grammar_path) {
//         Ok(s) => s,
//         Err(e) => {
//             eprintln!("[Test] Failed to read serialized grammar file '{}': {}", serialized_grammar_path, e);
//             eprintln!("[Test] Ensure the file exists (e.g., by running the test that generates it or placing it manually).");
//             return Err(Box::new(e)); // Fail the test if the prerequisite file is not found
//         }
//     };
//     let json_node = JSONNode::from_json_string(&json_string)?;
//     let compiled_grammar = CompiledGrammar::from_json(json_node)?;
//     println!("[Test] Successfully loaded CompiledGrammar from JSON.");
//     println!("[Test] Original grammar structure: {}", compiled_grammar.definition);

//     // 2. Define "interesting" symbols for filtering based on the sequence
//     let sequence_to_test_names = vec!["\"...\"", "\";\"", "\"elif\""];
//     let mut interesting_symbols = BTreeSet::new();
//     for name_str in &sequence_to_test_names {
//         // All elements in this specific sequence are terminals
//         interesting_symbols.insert(Symbol::Terminal(Terminal(name_str.to_string())));
//     }
//     println!("[Test] Interesting symbols for filtering: {:?}", sequence_to_test_names);

//     // 3. Apply the filter_productions_by_reachability function
//     let initially_filtered_productions = filter_productions_by_reachability(
//         &compiled_grammar.definition.productions,
//         &interesting_symbols,
//     );
//     println!("[Test] Productions after initial filter_productions_by_reachability: {}", initially_filtered_productions.len());

//     // 4. Apply remove_productions_with_undefined_nonterminals
//     let final_filtered_productions = remove_productions_with_undefined_nonterminals(&initially_filtered_productions);
//     println!("[Test] Productions after remove_productions_with_undefined_nonterminals: {}", final_filtered_productions.len());


//     println!("[Test] Original number of productions: {}", compiled_grammar.definition.productions.len());
//     println!("[Test] Filtered number of productions (final): {}", final_filtered_productions.len());

//     if final_filtered_productions.is_empty() {
//         if compiled_grammar.definition.productions.is_empty() {
//              println!("[Test] Original grammar was empty, so filtered grammar is also empty. This is expected.");
//         } else {
//             panic!("[Test] All productions were filtered out after cleanup. This indicates the interesting symbols are not reachable or productive in the original grammar, or the filter is too aggressive for this scenario, or the cleanup removed everything.");
//         }
//     }


//     // 5. Determine the start_production_id for the filtered set.
//     // It must be the same augmented start production as in the original grammar.
//     let original_start_production = &compiled_grammar.definition.productions[compiled_grammar.definition.start_production_id];
//     let new_start_production_id = match final_filtered_productions.iter().position(|p| p == original_start_production) {
//         Some(id) => id,
//         None => {
//              if final_filtered_productions.is_empty() {
//                 println!("[Test] Filtered productions list is empty, so original start production cannot be found. Skipping parser rebuild and sequence test.");
//                 return Ok(()); // Or handle as a test failure if an empty grammar is not expected
//             }
//             panic!("[Test] Original start production ('{}') was filtered out. This is unexpected if the sequence is meant to be parsable by the grammar. Cannot proceed with parser rebuild.", original_start_production);
//         }
//     };
//     println!("[Test] Original start production found in filtered set at new index: {}.", new_start_production_id);

//     // 6. Rebuild the GLR parser using the filtered productions.
//     // Use the original terminal_map and non_terminal_map from the loaded compiled_grammar.
//     // `generate_glr_parser_with_maps` includes validation, which might panic if the filtered grammar is invalid.
//     println!("[Test] Rebuilding parser with filtered productions...");
//     let filtered_definition = GrammarDefinition {
//         productions: final_filtered_productions.clone(), // Use the cleaned list
//         start_production_id: new_start_production_id,
//         terminal_name_to_group_id: compiled_grammar.definition.terminal_name_to_group_id.clone(),
//         terminal_expr_to_group_id: compiled_grammar.definition.terminal_expr_to_group_id.clone(),
//     };
//     // For debugging the structure of the filtered parser:
//     println!("[Test] Filtered grammar structure: {}", filtered_definition);
//     let filtered_parser = generate_glr_parser_with_maps(
//         &final_filtered_productions, // Use the cleaned list
//         new_start_production_id,
//         compiled_grammar.glr_parser.terminal_map.clone(),
//         compiled_grammar.glr_parser.non_terminal_map.clone(),
//     );
//     println!("[Test] Rebuilt parser with filtered productions. New parser has {} states.", filtered_parser.stage_7_table.len());

//     // 7. Convert the test sequence names to TerminalIDs using the *filtered_parser's* terminal_map.
//     let mut sequence_to_test_ids = Vec::new();
//     let mut all_terminals_mapped = true;
//     for name_str in &sequence_to_test_names {
//         if let Some(term_id) = filtered_parser.terminal_map.get_by_left(&Terminal(name_str.to_string())) {
//             sequence_to_test_ids.push(*term_id);
//         } else {
//             eprintln!("[Test] Error: Terminal '{}' from test sequence not found in filtered_parser's terminal_map.", name_str);
//             all_terminals_mapped = false;
//             break;
//         }
//     }

//     if !all_terminals_mapped {
//         panic!("[Test] Cannot run sequence test on filtered parser: one or more terminals from the sequence ('{:?}') were not found in its terminal_map. The filter might have removed necessary terminal definitions, or they were not part of the original grammar's terminal mapping in a way that survived filtering.", sequence_to_test_names);
//     }
//     assert_eq!(sequence_to_test_ids.len(), sequence_to_test_names.len(), "[Test] Mismatch in length between terminal names and resolved IDs for the test sequence.");

//     // 8. Initialize GLRParserState for the filtered parser.
//     // We use a unit accumulator `()` as this test focuses on grammar rule processing, not LLM token constraints.
//     let mut glr_state_filtered = filtered_parser.init_glr_parser::<()>();
//     println!("[Test] Initialized GLR state for filtered parser.");

//     // 9. Step the GLRParserState with the sequence of TerminalIDs.
//     println!("[Test] Stepping filtered parser with sequence: {:?} (IDs: {:?})", sequence_to_test_names, sequence_to_test_ids.iter().map(|id| id.0).collect::<Vec<_>>());
//     let mut step_by_step_ok = true;
//     for (idx, &terminal_id) in sequence_to_test_ids.iter().enumerate() {
//         glr_state_filtered.step(terminal_id);
//         println!("[Test]   Stepped with '{}' (ID {}). Parser OK: {}", sequence_to_test_names[idx], terminal_id.0, glr_state_filtered.is_ok());
//         if !glr_state_filtered.is_ok() {
//             step_by_step_ok = false;
//             eprintln!("[Test]   Parser became NOT OK after stepping with '{}'. This is the failure point for the sequence with the filtered grammar.", sequence_to_test_names[idx]);
//             // For detailed debugging, you might want to print the GSS of the failed state:
//             // glr_state_filtered.log_gss("GSS at failure point", terminal_id);
//             break;
//         }
//     }

//     // 10. Assert the outcome.
//     if step_by_step_ok {
//         println!("[Test] Filtered parser successfully processed the sequence token by token: {:?}. Final state OK: {}", sequence_to_test_names, glr_state_filtered.is_ok());
//         // This sequence should be a valid prefix or complete parse if the grammar logic for it is correct.
//         // assert!(glr_state_filtered.is_ok(), "[Test] Filtered parser should be in OK state after processing the sequence: {:?}", sequence_to_test_names);
//     } else {
//         // This assertion will cause the test to fail if any step resulted in a non-OK state.
//         // assert!(step_by_step_ok, "[Test] Filtered parser FAILED to process the sequence: {:?}. Check logs above for the failing step.", sequence_to_test_names);
//     }

//     println!("[Test] Test 'test_filtered_grammar_with_specific_sequence' completed successfully.");
//     Ok(())
// }

const PANIC_SUBSTRING_TO_FIND: &str = "not found in gotos for";

fn causes_specific_panic(
    productions_to_test: &[Production],
    original_augmented_start_lhs: &NonTerminal,
    sequence_to_test_names: &[&str],
    panic_substring: &str,
) -> bool {
    if productions_to_test.is_empty() {
        return false;
    }

    let current_start_prod_index = match productions_to_test
        .iter()
        .position(|p| p.lhs == *original_augmented_start_lhs)
    {
        Some(idx) => idx,
        None => {
            return false;
        }
    };

    let current_terminal_map = assign_terminal_ids(productions_to_test);
    let current_non_terminal_map = assign_non_terminal_ids(productions_to_test);

    let mut sequence_terminal_ids = Vec::new();
    for name_str in sequence_to_test_names {
        let terminal_to_find = Terminal(name_str.to_string());
        if let Some(term_id) = current_terminal_map.get_by_left(&terminal_to_find) {
            sequence_terminal_ids.push(*term_id);
        } else {
            return false;
        }
    }

    println!("[Test MRE] Productions to test: {}", display_productions(&productions_to_test));

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let parser = generate_glr_parser_with_maps(
            productions_to_test,
            current_start_prod_index,
            current_terminal_map,
            current_non_terminal_map,
        );

        let mut glr_state = parser.init_glr_parser();

        for &terminal_id in &sequence_terminal_ids {
            glr_state.step(terminal_id);
        }
    }));

    match result {
        Ok(_) => false,
        Err(panic_payload) => {
            let panic_message = if let Some(s) = panic_payload.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "Unknown panic payload type".to_string()
            };
            panic_message.contains(panic_substring)
        }
    }
}

/// Helper to remove symbols from productions.
/// `productions`: The vector of productions to modify.
/// `locations_to_remove`: A slice of `(prod_idx, symbol_idx_in_rhs)` tuples.
/// `prod_idx` refers to the index in the `productions` vector.
fn remove_symbols_at_locations_destructive(
    productions: &mut Vec<Production>, // Mutably borrow to modify in place
    locations_to_remove: &[(usize, usize)], // (prod_idx, symbol_idx_in_rhs)
) {
    // Group symbol indices to remove by their production index
    let mut symbols_to_remove_by_prod: HashMap<usize, BTreeSet<usize>> = HashMap::new();
    for &(prod_idx, symbol_idx) in locations_to_remove {
        symbols_to_remove_by_prod
            .entry(prod_idx)
            .or_default()
            .insert(symbol_idx);
    }

    // Iterate over the productions. If a production is targeted, rebuild its RHS.
    for (prod_idx, prod_ref_mut) in productions.iter_mut().enumerate() {
        if let Some(indices_in_rhs_to_remove) = symbols_to_remove_by_prod.get(&prod_idx) {
            let original_rhs = std::mem::take(&mut prod_ref_mut.rhs); // Take ownership of old RHS
            let mut new_rhs = Vec::with_capacity(original_rhs.len());
            for (symbol_idx, symbol) in original_rhs.into_iter().enumerate() {
                if !indices_in_rhs_to_remove.contains(&symbol_idx) {
                    new_rhs.push(symbol);
                }
            }
            prod_ref_mut.rhs = new_rhs; // Assign the new RHS
        }
    }
}

fn get_all_terminals_in_grammar(productions: &[Production]) -> BTreeSet<Terminal> {
    let mut terminals = BTreeSet::new();
    for prod in productions {
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                terminals.insert(t.clone());
            }
        }
    }
    terminals
}

fn get_all_nonterminals_in_grammar(productions: &[Production]) -> BTreeSet<NonTerminal> {
    let mut nonterminals = BTreeSet::new();
    for prod in productions {
        nonterminals.insert(prod.lhs.clone());
        for symbol in &prod.rhs {
            if let Symbol::NonTerminal(nt) = symbol {
                nonterminals.insert(nt.clone());
            }
        }
    }
    nonterminals
}

/// Final simplification pass: Inlines rules of the form A -> B (single non-terminal RHS)
/// if the inlining preserves the target panic.
fn simplify_and_inline_unit_nonterminal_rules(
    mut productions: Vec<Production>, // Takes ownership, returns new Vec
    augmented_start_rule_lhs: &NonTerminal,
    sequence_to_test_names: &[&str],
    panic_substring: &str,
) -> Vec<Production> {
    println!("\n[Simplifier] Starting final unit non-terminal inlining pass...");
    let mut iteration = 0;
    loop {
        iteration += 1;
        let initial_prod_count = productions.len();
        println!("[Simplifier] Iteration {}. Current productions: {}", iteration, initial_prod_count);
        
        let mut made_change_in_this_iteration = false;
        let mut best_candidate_to_inline: Option<(NonTerminal, NonTerminal, usize)> = None;

        // Find the first eligible candidate A -> B
        for (idx, p) in productions.iter().enumerate() {
            // Rule: A -> B (where B is a single NonTerminal)
            // Condition 1: LHS (A) is not the augmented start symbol.
            // Condition 2: RHS has exactly one symbol.
            // Condition 3: That one symbol is a NonTerminal (B).
            // Condition 4: A != B (avoid trivial A -> A inlining here).
            if p.lhs != *augmented_start_rule_lhs &&
               p.rhs.len() == 1 {
                if let Symbol::NonTerminal(ref b_nt) = p.rhs[0] {
                    if p.lhs != *b_nt {
                        best_candidate_to_inline = Some((p.lhs.clone(), b_nt.clone(), idx));
                        break; // Found a candidate, try to process it
                    }
                }
            }
        }

        if let Some((a_nt_to_inline, b_nt_replacement, p_candidate_idx)) = best_candidate_to_inline {
            println!("[Simplifier] Attempting to inline {} -> {} (from P{})", 
                     a_nt_to_inline.0, b_nt_replacement.0, p_candidate_idx);

            let mut temp_productions_after_inlining = Vec::new();
            // 1. Copy all productions except the one being inlined (A -> B)
            for (idx, p) in productions.iter().enumerate() {
                if idx != p_candidate_idx {
                    temp_productions_after_inlining.push(p.clone());
                }
            }

            // 2. Perform the inlining: replace `a_nt_to_inline` with `b_nt_replacement` in RHS of other rules
            for prod_being_modified in temp_productions_after_inlining.iter_mut() {
                let mut new_rhs = Vec::with_capacity(prod_being_modified.rhs.len());
                let mut rhs_was_changed = false;
                for symbol_in_rhs in prod_being_modified.rhs.iter() {
                    if let Symbol::NonTerminal(nt_in_rhs) = symbol_in_rhs {
                        if *nt_in_rhs == a_nt_to_inline {
                            new_rhs.push(Symbol::NonTerminal(b_nt_replacement.clone()));
                            rhs_was_changed = true;
                        } else {
                            new_rhs.push(symbol_in_rhs.clone());
                        }
                    } else {
                        new_rhs.push(symbol_in_rhs.clone());
                    }
                }
                if rhs_was_changed {
                    prod_being_modified.rhs = new_rhs;
                }
            }
            
            if temp_productions_after_inlining.is_empty() || 
               !temp_productions_after_inlining.iter().any(|p| p.lhs == *augmented_start_rule_lhs) {
                println!("[Simplifier] Inlining {} would remove the augmented start rule or empty the grammar. Reverting.", 
                         a_nt_to_inline.0);
            } else if causes_specific_panic(
                &temp_productions_after_inlining,
                augmented_start_rule_lhs,
                sequence_to_test_names,
                panic_substring,
            ) {
                println!("[Simplifier] Successfully inlined {} -> {}. Productions count: {} -> {}", 
                         a_nt_to_inline.0, b_nt_replacement.0, 
                         productions.len(), temp_productions_after_inlining.len());
                productions = temp_productions_after_inlining; // Commit the change
                made_change_in_this_iteration = true;
            } else {
                println!("[Simplifier] Inlining {} removed the target panic. Reverting this step.", 
                         a_nt_to_inline.0);
            }
        }


        if !made_change_in_this_iteration {
            if initial_prod_count == productions.len() { // No change in production count means no successful inlining
                 println!("[Simplifier] No more valid unit non-terminal inlinings found in this iteration.");
                break; // Exit the simplification loop
            }
        }
    }
    println!("[Simplifier] Final unit non-terminal inlining pass finished.");
    productions
}

fn inline_sole_productions_pass(
    mut productions: Vec<Production>, // Takes ownership
    augmented_start_rule_lhs: &NonTerminal,
    sequence_to_test_names: &[&str],
    panic_substring: &str,
) -> (Vec<Production>, bool) { // Returns new productions and bool indicating if a change was made
    let mut made_change_this_outer_iteration = false;
    loop { // Keep iterating until a full pass makes no changes
        let mut changed_in_current_scan = false;
        
        let mut nts_to_productions_info: BTreeMap<NonTerminal, Vec<(usize, Production)>> = BTreeMap::new();
        for (idx, p) in productions.iter().enumerate() {
            nts_to_productions_info.entry(p.lhs.clone()).or_default().push((idx, p.clone()));
        }

        let mut candidate_to_inline_info: Option<(NonTerminal, Vec<Symbol>, usize)> = None;

        // Find a candidate NT to inline
        for (nt_candidate, defining_rules_with_indices) in &nts_to_productions_info {
            if nt_candidate == augmented_start_rule_lhs { continue; }

            if defining_rules_with_indices.len() == 1 {
                let (original_prod_idx, single_prod_rule) = &defining_rules_with_indices[0];
                
                let is_used_in_rhs_of_other_rules = productions.iter().any(|p| {
                    // Only consider it "used" if it's in the RHS of a *different* NT's rule,
                    // or in a different rule for the same NT (if it had multiple rules, which it doesn't here)
                    // OR if it's used in the RHS of the augmented start rule.
                    (p.lhs != *nt_candidate || p.lhs == *augmented_start_rule_lhs) && 
                    p.rhs.iter().any(|s| match s {
                        Symbol::NonTerminal(nt_in_rhs) => nt_in_rhs == nt_candidate,
                        _ => false,
                    })
                });

                let is_recursive_in_its_own_sole_rule = single_prod_rule.rhs.iter().any(|s| match s {
                    Symbol::NonTerminal(nt_in_rhs) => nt_in_rhs == nt_candidate,
                    _ => false,
                });

                // Only inline if it's used and not directly recursive in its sole definition
                // (e.g. A -> alpha A beta)
                if is_used_in_rhs_of_other_rules && !is_recursive_in_its_own_sole_rule {
                    candidate_to_inline_info = Some((nt_candidate.clone(), single_prod_rule.rhs.clone(), *original_prod_idx));
                    break; 
                }
            }
        }

        if let Some((nt_to_inline, alpha_rhs_to_substitute, original_prod_idx_to_remove)) = candidate_to_inline_info {
            println!("[Simplifier-Sole] Attempting to inline {} -> {:?} (from P{})", 
                     nt_to_inline.0, alpha_rhs_to_substitute.iter().map(|s| match s { Symbol::Terminal(t) => t.0.clone(), Symbol::NonTerminal(n) => n.0.clone() }).collect::<Vec<_>>(), original_prod_idx_to_remove);

            let mut temp_productions_after_inlining = Vec::new();
            for (idx, p) in productions.iter().enumerate() {
                if idx != original_prod_idx_to_remove {
                    temp_productions_after_inlining.push(p.clone());
                }
            }

            for prod_being_modified in temp_productions_after_inlining.iter_mut() {
                let mut new_rhs_for_this_prod = Vec::new();
                let mut current_prod_rhs_changed = false;
                for symbol_in_rhs in prod_being_modified.rhs.iter() {
                    if let Symbol::NonTerminal(nt_in_rhs_val) = symbol_in_rhs {
                        if *nt_in_rhs_val == nt_to_inline {
                            new_rhs_for_this_prod.extend_from_slice(&alpha_rhs_to_substitute);
                            current_prod_rhs_changed = true;
                        } else {
                            new_rhs_for_this_prod.push(symbol_in_rhs.clone());
                        }
                    } else {
                        new_rhs_for_this_prod.push(symbol_in_rhs.clone());
                    }
                }
                if current_prod_rhs_changed {
                    prod_being_modified.rhs = new_rhs_for_this_prod;
                }
            }
            
            if temp_productions_after_inlining.is_empty() || 
               !temp_productions_after_inlining.iter().any(|p| p.lhs == *augmented_start_rule_lhs) {
                println!("[Simplifier-Sole] Inlining {} would remove augmented start or empty grammar. Skipping this specific inlining.", nt_to_inline.0);
            } else if causes_specific_panic(
                &temp_productions_after_inlining,
                augmented_start_rule_lhs,
                sequence_to_test_names,
                panic_substring,
            ) {
                println!("[Simplifier-Sole] Successfully inlined {}. Productions count: {} -> {}", 
                         nt_to_inline.0, productions.len(), temp_productions_after_inlining.len());
                productions = temp_productions_after_inlining; 
                changed_in_current_scan = true;
                made_change_this_outer_iteration = true; 
                break; // Restart scan from the beginning with the new production set
            } else {
                println!("[Simplifier-Sole] Inlining {} removed target panic. Reverting this attempt.", nt_to_inline.0);
            }
        } // end if let Some(candidate_to_inline_info)
        
        if !changed_in_current_scan {
            break; // No change made in a full scan over NTs, exit the outer `loop {}`
        }
    } // end outer loop for iterative application

    (productions, made_change_this_outer_iteration)
}

#[test]
fn test_minimize_grammar_for_goto_panic() -> Result<(), Box<dyn std::error::Error>> {
    // --- Initial Setup (same as before) ---
    let serialized_grammar_path = "src/serialized_compiled_grammar.json";
    println!("[Minimizer] Loading base CompiledGrammar from: {}", serialized_grammar_path);
    let json_string = match fs::read_to_string(serialized_grammar_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[Minimizer] Failed to read serialized grammar file '{}': {}", serialized_grammar_path, e);
            return Err(Box::new(e));
        }
    };
    let json_node = JSONNode::from_json_string(&json_string)?;
    let compiled_grammar = CompiledGrammar::from_json(json_node)?;
    println!("[Minimizer] Successfully loaded CompiledGrammar.");

    let initial_productions = compiled_grammar.definition.productions.clone();
    let augmented_start_rule_lhs = compiled_grammar.definition.productions
        [compiled_grammar.definition.start_production_id].lhs.clone();
    // let sequence_to_test_names = ["\"...\"", "\";\"", "\"elif\""];
    // let sequence_to_test_names = ["\"yield\"", "IGNORE[0][0]", "NEWLINE[0]", "\"-\""];
    let sequence_to_test_names = ["\"return\"", "\";\"", "IGNORE[0][0]", "\"[\"[0]"];

    println!("[Minimizer] Starting stochastic minimization for panic substring: '{}'", PANIC_SUBSTRING_TO_FIND);
    println!("[Minimizer] Initial number of productions: {}", initial_productions.len());
    println!("[Minimizer] Test sequence: {:?}", sequence_to_test_names);
    println!("[Minimizer] Augmented start LHS: {}", augmented_start_rule_lhs.0);

    let mut current_productions = initial_productions;

    if !causes_specific_panic(
        &current_productions,
        &augmented_start_rule_lhs,
        &sequence_to_test_names,
        PANIC_SUBSTRING_TO_FIND,
    ) {
        eprintln!("[Minimizer] Initial grammar does not cause the specific panic. Cannot proceed.");
        assert!(false, "Initial grammar does not cause the specific panic.");
        return Ok(());
    }
    println!("[Minimizer] Confirmed: Initial grammar causes the specific panic.");

    let mut rng = StdRng::seed_from_u64(42);
    let mut pass_num = 0;

    const ATTEMPTS_PER_STRATEGY_PASS: usize = 100;
    const MAX_PRODUCTIONS_TO_REMOVE_IN_CHUNK: usize = 3;
    const MAX_SYMBOLS_TO_REMOVE_IN_CHUNK: usize = 5; // Increased slightly

    'pass_loop: loop {
        pass_num += 1;
        println!(
            "\n[Minimizer] Starting Pass {}. Current productions: {}",
            pass_num,
            current_productions.len()
        );
        std::io::stdout().flush().unwrap();
        let mut made_change_in_this_pass = false;

        // --- Strategy 1: Try removing chunks of productions ---
        if current_productions.len() > 1 {
            print!("[Minimizer] Pass {}: Trying production chunk removal ({} attempts)...", pass_num, ATTEMPTS_PER_STRATEGY_PASS);
            std::io::stdout().flush().unwrap();
            for attempt in 0..ATTEMPTS_PER_STRATEGY_PASS {
                let eligible_prod_indices: Vec<usize> = (0..current_productions.len())
                    .filter(|&idx| current_productions[idx].lhs != augmented_start_rule_lhs)
                    .collect();

                if eligible_prod_indices.is_empty() { break; }

                let chunk_size = rng.gen_range(1..=std::cmp::min(MAX_PRODUCTIONS_TO_REMOVE_IN_CHUNK, eligible_prod_indices.len()));
                
                let chosen_indices_to_remove: Vec<usize> = eligible_prod_indices
                    .choose_multiple(&mut rng, chunk_size)
                    .cloned()
                    .collect();

                if chosen_indices_to_remove.is_empty() { continue; }

                let mut temp_productions = current_productions.clone();
                let mut sorted_indices = chosen_indices_to_remove;
                sorted_indices.sort_unstable_by(|a, b| b.cmp(a));
                for &idx_to_remove in &sorted_indices {
                    temp_productions.remove(idx_to_remove);
                }
                
                if temp_productions.is_empty() || !temp_productions.iter().any(|p| p.lhs == augmented_start_rule_lhs) {
                    continue;
                }

                if causes_specific_panic(
                    &temp_productions,
                    &augmented_start_rule_lhs,
                    &sequence_to_test_names,
                    PANIC_SUBSTRING_TO_FIND,
                ) {
                    if attempt % (ATTEMPTS_PER_STRATEGY_PASS / 10 + 1) == 0 { print!("#"); std::io::stdout().flush().unwrap(); }
                    current_productions = temp_productions;
                    made_change_in_this_pass = true;
                    println!("\n[Minimizer] Reduced by production chunk ({} removed). New count: {}", chunk_size, current_productions.len());
                    continue 'pass_loop; 
                }
            }
            println!();
        }

        // --- Strategy 2: Try removing chunks of symbols ---
        let mut all_symbol_locations: Vec<(usize, usize)> = Vec::new();
        for (prod_idx, prod) in current_productions.iter().enumerate() {
            for symbol_idx in 0..prod.rhs.len() {
                all_symbol_locations.push((prod_idx, symbol_idx));
            }
        }

        if !all_symbol_locations.is_empty() {
            print!("[Minimizer] Pass {}: Trying symbol chunk removal ({} attempts)...", pass_num, ATTEMPTS_PER_STRATEGY_PASS);
            std::io::stdout().flush().unwrap();
            for attempt in 0..ATTEMPTS_PER_STRATEGY_PASS {
                let chunk_size = rng.gen_range(1..=std::cmp::min(MAX_SYMBOLS_TO_REMOVE_IN_CHUNK, all_symbol_locations.len()));
                let locations_to_remove: Vec<(usize, usize)> = all_symbol_locations
                    .choose_multiple(&mut rng, chunk_size)
                    .cloned()
                    .collect();

                if locations_to_remove.is_empty() { continue; }

                let mut temp_productions = current_productions.clone();
                remove_symbols_at_locations_destructive(&mut temp_productions, &locations_to_remove);
                
                if temp_productions.is_empty() || !temp_productions.iter().any(|p| p.lhs == augmented_start_rule_lhs) {
                     continue;
                }

                if causes_specific_panic(
                    &temp_productions,
                    &augmented_start_rule_lhs,
                    &sequence_to_test_names,
                    PANIC_SUBSTRING_TO_FIND,
                ) {
                    if attempt % (ATTEMPTS_PER_STRATEGY_PASS / 10 + 1) == 0 { print!("*"); std::io::stdout().flush().unwrap(); }
                    current_productions = temp_productions;
                    made_change_in_this_pass = true;
                    println!("\n[Minimizer] Reduced by symbol chunk ({} removed). New count: {}", chunk_size, current_productions.len());
                    continue 'pass_loop;
                }
            }
            println!();
        }

        if !made_change_in_this_pass {
            println!("[Minimizer] Pass {}: No further reductions found in this stochastic pass.", pass_num);
            break 'pass_loop;
        }
    }

    println!("\n[Minimizer] Stochastic minimization phase complete.");
    println!("[Minimizer] Productions after stochastic phase: {}", current_productions.len());

    // --- Final Simplification Pass: Inline A -> B rules ---
    println!("\n[Minimizer] Deterministic simplification phase starting.");
    loop {
        let productions_at_start_of_deterministic_iter = current_productions.clone();

        // Apply A -> B inlining (iterates to fixed point internally)
        let prev_len_unit_inline = current_productions.len();
        current_productions = simplify_and_inline_unit_nonterminal_rules(
            current_productions, // Takes ownership
            &augmented_start_rule_lhs,
            &sequence_to_test_names,
            PANIC_SUBSTRING_TO_FIND,
        );
        if current_productions.len() != prev_len_unit_inline {
             println!("[Minimizer-Determ] simplify_and_inline_unit_nonterminal_rules changed productions: {} -> {}", prev_len_unit_inline, current_productions.len());
        }
        
        // Apply A -> alpha (sole production) inlining (iterates to fixed point internally)
        let prev_len_sole_inline = current_productions.len();
        let (next_prods_after_sole_inline, _sole_inlined_overall) = inline_sole_productions_pass( // We don't strictly need sole_inlined_overall for this new check
            current_productions, // Takes ownership
            &augmented_start_rule_lhs,
            &sequence_to_test_names,
            PANIC_SUBSTRING_TO_FIND,
        );
        current_productions = next_prods_after_sole_inline;
        if current_productions.len() != prev_len_sole_inline {
            println!("[Minimizer-Determ] inline_sole_productions_pass changed productions: {} -> {}", prev_len_sole_inline, current_productions.len());
        }

        // Check for convergence by comparing the entire set of productions
        if current_productions == productions_at_start_of_deterministic_iter {
            println!("[Minimizer] Deterministic simplification phase converged. Final production count: {}", current_productions.len());
            break;
        }
        println!("[Minimizer] Deterministic iteration made changes. Productions count: {} -> {}. Repeating deterministic loop.", productions_at_start_of_deterministic_iter.len(), current_productions.len());
    }

    // --- Output Final Minimized Grammar ---
    println!("\n[Minimizer] Final number of productions after all simplifications: {}", current_productions.len());
    println!("[Minimizer] Minimized Grammar Productions (Augmented Start LHS: {}):", augmented_start_rule_lhs.0);
    for (idx, prod) in current_productions.iter().enumerate() {
        println!("  P{}: {}", idx, prod);
    }

    assert!(false, "[Minimizer] Minimization finished. Review the MRE printed above and address the underlying panic.");
    Ok(())
}

#[test]
fn test_minimized_grammar_causes_panic() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[Test MRE] Testing the manually defined minimized grammar that causes the panic.");

    // 1. Manually define the minimized grammar
    // P0: start' -> IGNORE[0] "return" simple_stmts[0] ";"
    // P1: simple_stmts[0] -> ";" IGNORE[0] "return"
    // P2: simple_stmts[0] ->
    // P3: t_primary -> IGNORE[0] "["[0]
    // P4: IGNORE[0] -> IGNORE[0][0]
    // P5: IGNORE[0] ->
    let minimized_productions = vec![
        prod("start'", vec![nt("IGNORE[0]"), t("\"return\""), nt("simple_stmts[0]"), t("\";\"")]), // P0
        prod("simple_stmts[0]", vec![t("\";\""), nt("IGNORE[0]"), t("\"return\"")]), // P1
        prod("simple_stmts[0]", vec![]), // P2
        prod("t_primary", vec![nt("IGNORE[0]"), t("\"[\"[0]")]), // P3
        prod("IGNORE[0]", vec![t("IGNORE[0][0]")]), // P4
        prod("IGNORE[0]", vec![]), // P5
    ];
    let start_production_id_for_minimized = 0; // P0 is the start rule

    // 2. Define the input sequence that triggers the panic
    // let input_sequence_names = ["...", ";", "elif"];
    // let input_sequence_names = ["\"yield\"", "IGNORE[0][0]", "NEWLINE[0]", "\"-\""];
    let input_sequence_names = ["\"return\"", "\";\"", "IGNORE[0][0]", "\"[\"[0]"];
    println!("[Test MRE] Input sequence: {:?}", input_sequence_names);

    // 3. Create terminal and non-terminal maps specifically for this minimized grammar
    //    This ensures IDs are compact and consistent for the small grammar.
    let terminal_map_for_minimized = assign_terminal_ids(&minimized_productions);
    let non_terminal_map_for_minimized = assign_non_terminal_ids(&minimized_productions);

    println!("[Test MRE] Terminal Map for Minimized Grammar: {:?}", terminal_map_for_minimized);
    println!("[Test MRE] Non-Terminal Map for Minimized Grammar: {:?}", non_terminal_map_for_minimized);


    // 4. Convert input sequence names to TerminalIDs using the new map
    let mut input_sequence_ids = Vec::new();
    for name_str in &input_sequence_names {
        let terminal_to_find = Terminal(name_str.to_string());
        if let Some(term_id) = terminal_map_for_minimized.get_by_left(&terminal_to_find) {
            input_sequence_ids.push(*term_id);
        } else {
            panic!("[Test MRE] Critical error: Terminal '{}' from input sequence not found in minimized grammar's terminal map. Map: {:?}", 
                   name_str, terminal_map_for_minimized);
        }
    }
    println!("[Test MRE] Input sequence TerminalIDs: {:?}", input_sequence_ids.iter().map(|id| id.0).collect::<Vec<_>>());


    // 5. Attempt to generate parser and step
    println!("[Test MRE] Attempting to generate parser and run sequence...");
    
    // Generate GLRParser for the minimized grammar
    let parser = generate_glr_parser_with_maps(
        &minimized_productions,
        start_production_id_for_minimized,
        terminal_map_for_minimized.clone(), // Pass the maps specific to this grammar
        non_terminal_map_for_minimized.clone(),
    );
    println!("Parser: {}", parser);

    // Initialize GLRParserState (accumulator type `()` is fine for this test)
    let mut glr_state = parser.init_glr_parser();

    // Step through the input sequence
    for (idx, &terminal_id) in input_sequence_ids.iter().enumerate() {
        println!("[Test MRE] Stepping with token {}/{} ('{}', ID {})", 
                 idx + 1, input_sequence_ids.len(), input_sequence_names[idx], terminal_id.0);
        glr_state.step(terminal_id);
        // If a panic occurs during step, the test will fail here naturally.
    }
    
    // If the code reaches this point, no panic occurred.
    println!("[Test MRE] Sequence processed without panic. If a panic was expected, this MRE does not reproduce it.");

    Ok(())
}
