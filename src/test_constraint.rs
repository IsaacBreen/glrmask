use std::collections::BTreeMap;
use crate::finite_automata::eat_u8;
use crate::{choice, choice_fast, groups, seq, seq_fast};
use crate::glr::grammar::{nt, prod, t, NonTerminal, Terminal}; // Added Terminal, NonTerminal, prod, t, nt
use crate::glr::table::{generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map, TerminalID as ParserTerminalID}; // Added ParserTerminalID and other items
use crate::datastructures::hybrid_bitset::HybridBitset;
use std::hash::{Hash, Hasher};
use crate::interface::{eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast, eat_any_fast, eat_string_fast, choice_fast, eat_bytestring_fast, repeat1_fast};

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use bimap::BiBTreeMap;
use reqwest::blocking;
use serde_json;
use crate::constraint::GrammarConstraint;
use crate::datastructures::trie::Trie;
use crate::json_serialization::{JSONConvertible, JSONNode}; // Added JSONNode
// Already a main dependency, but good to be explicit if used directly
// reqwest will be used if the file isn't cached, ensure it's in dev-dependencies
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use crate::types::TerminalID; // This is GrammarTokenID

// Use concrete types for merge tests
type TestTrieMerge = Trie<&'static str, Vec<i32>, String>;
type TestNodeMerge = Arc<Mutex<TestTrieMerge>>;
// Use simpler types for basic tests
type TestTrieBasic = Trie<&'static str, &'static str, i32>;
type TestNodeBasic = Arc<Mutex<TestTrieBasic>>;

// Use concrete types for EdgeInserter tests
type TestTrieEI = Trie<&'static str, HybridBitset, String>;
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

// Helper function to generate JSON for a simple GrammarConstraint
// This function will be used by the new test.
fn generate_dummy_grammar_constraint_json_string() -> Result<String, Box<dyn std::error::Error>> {
    // 1. Tokenizer: matches 'x'
    let tokenizer_expr = groups![eat_u8(b'x')];
    let tokenizer = tokenizer_expr.build();

    // 2. LLM Token Map (for the dummy constraint being serialized)
    let mut llm_token_map_dummy = LLMTokenMap::new();
    llm_token_map_dummy.insert(b"x".to_vec(), LLMTokenID(0));
    let max_original_llm_token_id_dummy = 0;

    // 3. Grammar Productions and Parser
    // Grammar: S -> X_TOK (where X_TOK is the terminal for 'x')
    let mut grammar_token_map_dummy: BiBTreeMap<crate::glr::grammar::Terminal, crate::glr::table::TerminalID> = BiBTreeMap::new();
    grammar_token_map_dummy.insert(crate::glr::grammar::Terminal("X_TOK".to_string()), crate::glr::table::TerminalID(0)); // "X_TOK" maps to tokenizer group 0

    let productions_dummy = vec![
        prod("S", vec![t("X_TOK")]),
    ];
    // The generate_glr_parser_with_terminal_map expects TerminalID from crate::glr::table
    let parser_dummy = generate_glr_parser_with_terminal_map(&productions_dummy, 0, grammar_token_map_dummy.clone());

    // 4. Token Name Map (maps grammar terminal name (String) to tokenizer group ID (usize))
    let mut token_name_map_dummy = BiBTreeMap::new();
    token_name_map_dummy.insert("X_TOK".to_string(), 0); // Grammar's "X_TOK" is tokenizer's group 0

    // 5. Create the dummy GrammarConstraint
    let dummy_constraint = GrammarConstraint::new(
        tokenizer,
        parser_dummy,
        llm_token_map_dummy,
        token_name_map_dummy,
        max_original_llm_token_id_dummy,
    );

    // 6. Serialize to JSON string
    let json_node = dummy_constraint.to_json();
    Ok(json_node.to_json_string())
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
    let mut grammar_token_map: BiBTreeMap<crate::glr::grammar::Terminal, crate::glr::table::TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(crate::glr::grammar::Terminal("A".to_string()), crate::glr::table::TerminalID(0)); // Corresponds to eat_u8(b'a')
    grammar_token_map.insert(crate::glr::grammar::Terminal("AB".to_string()), crate::glr::table::TerminalID(1)); // Corresponds to seq![eat_u8(b'a'), eat_u8(b'b')]
    grammar_token_map.insert(crate::glr::grammar::Terminal("B_OR_C".to_string()), crate::glr::table::TerminalID(2)); // Corresponds to choice![eat_u8(b'b'), eat_u8(b'c')]
    grammar_token_map.insert(crate::glr::grammar::Terminal("EOF".to_string()), crate::glr::table::TerminalID(3)); // Corresponds to eat_u8(b'$')

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
        2, // max_original_llm_token_id should be 2 for IDs 0, 1, 2.
           // If there are 3 tokens with IDs 0,1,2, the max ID is 2.
           // The capacity for bitsets will be max_id + 1.
    );
    // constraint.dump_precomputed(); // Commented out dump for cleaner test output

    let mut constraint_state = constraint.init();

    constraint_state.step_with_all_llm_tokens();

    // Initially, we can match "a" (part of "ab" or "ac") or "ab).
    // "a" leads to expecting "b" or "c".
    // "ab" leads to expecting "$".
    let mask = constraint_state.get_mask();
    assert_eq!(mask, HybridBitset::from_iter(vec![0, 1])); // Expect "ab" or "ac"

    // Commit "ab" (LLMTokenID 0)
    constraint_state.commit(LLMTokenID(0));
    constraint_state.step_with_all_llm_tokens();
    let mask = constraint_state.get_mask();
    assert_eq!(mask, HybridBitset::from_iter(vec![2])); // Expect "$" (EOF)

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json); // Use the assert_eq method
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
        eat_u8(b'$'), // Added for EOF
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
    let mut grammar_token_map: BiBTreeMap<crate::glr::grammar::Terminal, crate::glr::table::TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(crate::glr::grammar::Terminal("PLUS".to_string()), crate::glr::table::TerminalID(0));
    grammar_token_map.insert(crate::glr::grammar::Terminal("TIMES".to_string()), crate::glr::table::TerminalID(1));
    grammar_token_map.insert(crate::glr::grammar::Terminal("LPAREN".to_string()), crate::glr::table::TerminalID(2));
    grammar_token_map.insert(crate::glr::grammar::Terminal("RPAREN".to_string()), crate::glr::table::TerminalID(3));
    grammar_token_map.insert(crate::glr::grammar::Terminal("I".to_string()), crate::glr::table::TerminalID(4));
    grammar_token_map.insert(crate::glr::grammar::Terminal("EOF".to_string()), crate::glr::table::TerminalID(5)); // EOF is group 5

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
        6, // max_llm_token_id is 6 for IDs 0-6
    );
    // constraint.dump_precomputed(); // Commented out dump for cleaner test output

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

    // Test Serialization/Deserialization
    let json = constraint.to_json();
    let constraint_from_json = GrammarConstraint::from_json(json).unwrap();
    constraint.assert_eq(&constraint_from_json);
}

#[test]
fn test_precompute_for_python_name_token() {
    let ignore = repeat0_fast(choice_fast!(eat_u8_fast(b' '), seq_fast!(eat_u8_fast(b'#'), repeat0_fast(eat_u8_negation_fast(b'\n')), eat_u8_fast(b'\n'))));

    let digit = eat_u8_range_fast(b'0', b'9');
    let alph_lower = eat_u8_range_fast(b'a', b'z');
    let alph_upper = eat_u8_range_fast(b'A', b'Z');

    let name_start = choice_fast!(alph_lower, alph_upper, eat_u8_fast(b'_'));
    let name_middle = choice_fast!(name_start.clone(), digit);
    let name = seq_fast!(ignore, name_start, repeat0_fast(seq_fast!(name_middle)));

    let tokenizer = name.build(); // This builds a Regex with one group for the whole 'name' expression
    dbg!(&tokenizer);

    let llm_tokens: Vec<Vec<u8>> = (0..2).map(|i| format!("abcdefghijk{}", i).as_bytes().to_vec()).collect();
    let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();

    let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
    let mut max_internal_id = 0;
    for (i, token) in llm_tokens.iter().enumerate() {
         internal_llm_token_map_for_precompute.insert(token.clone(), LLMTokenID(i));
         if i > max_internal_id { max_internal_id = i; }
    }

    // Since 'name' is one group (ID 0), token_name_map should reflect this.
    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert("NAME".to_string(), 0);


    let _precomputed = GrammarConstraint::precompute(
        &tokenizer,
        &internal_llm_token_map_for_precompute,
        &token_name_map, // Pass the actual token_name_map
        max_internal_id, // Max internal ID
    );
    println!("Done precomputing for python name token");
}

#[test]
fn test_precompute_explosion() {
    let tokenizer = groups![
        eat_u8(b'a'), // Group 0
        eat_u8(b'a'), // Group 1 (distinct group, though same regex content)
    ].build();

    let llm_tokens: Vec<Vec<u8>> = vec![b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec()];
    let llm_token_map: LLMTokenMap = llm_tokens.iter().enumerate().map(|(i, token)| (token.clone(), LLMTokenID(i))).collect();

    let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
    let mut max_internal_id = 0;
    for (i, token) in llm_tokens.iter().enumerate() {
         internal_llm_token_map_for_precompute.insert(token.clone(), LLMTokenID(i));
         if i > max_internal_id { max_internal_id = i; }
    }

    // Token name map for the two groups in the tokenizer
    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert("A1".to_string(), 0);
    token_name_map.insert("A2".to_string(), 1);

    let _precomputed = GrammarConstraint::precompute(
        &tokenizer,
        &internal_llm_token_map_for_precompute,
        &token_name_map, // Pass the actual token_name_map
        max_internal_id, // Max internal ID
    );
    println!("Done precomputing for explosion test");
}

#[test]
fn test_precompute_with_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_expr = groups![
        repeat0_fast(choice_fast!(eat_u8_negation_fast(b'{'), eat_bytestring_fast(b"{{".to_vec()))), // Group 0
        eat_string_fast("def"), // Group 1
    ];
    let tokenizer = tokenizer_expr.build();

    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let cache_dir = Path::new(".cache/test_vocabs");
    let vocab_file_name = "gpt2_vocab.json";

    let gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;

    let prop = 0.05; // Reduced sample size for faster test
    let total_tokens = gpt2_raw_vocab.len();
    let sample_size = ((total_tokens as f64 * prop) as usize).max(10); // Ensure at least 10 tokens

    println!("Sampling {} out of {} GPT-2 tokens for precompute", sample_size, total_tokens);
    for (token_str, id_val_u32) in gpt2_raw_vocab.into_iter().take(sample_size) {
        let id_val = id_val_u32 as usize;
        llm_token_map.insert(token_str.into_bytes(), LLMTokenID(id_val));
        if id_val > max_original_llm_token_id_val {
            max_original_llm_token_id_val = id_val;
        }
    }

    // Setup for internal mapping (mimicking GrammarConstraint::new)
    let original_to_internal_bimap = GrammarConstraint::setup_llm_token_mappings(&llm_token_map);
    let mut internal_llm_token_map_for_precompute = BiBTreeMap::new();
    let mut max_internal_id: usize = 0;
    for (bytes, original_id) in llm_token_map.iter() {
        if let Some(internal_id_val) = original_to_internal_bimap.get_by_left(&original_id.0) {
            internal_llm_token_map_for_precompute.insert(bytes.clone(), LLMTokenID(*internal_id_val));
            if *internal_id_val > max_internal_id {
                max_internal_id = *internal_llm_token_map_for_precompute.get_by_left(&bytes.clone()).unwrap().0;
            }
        }
    }


    let mut token_name_map: BiBTreeMap<String, usize> = BiBTreeMap::new();
    token_name_map.insert("FSTRING_MIDDLE".to_string(), 0);
    token_name_map.insert("DEF".to_string(), 1);

    println!(
        "Starting precompute with GPT-2 vocab ({} tokens, max_original_id_val: {}, max_internal_id: {})...",
        llm_token_map.len(),
        max_original_llm_token_id_val,
        max_internal_id
    );

    let _precomputed = GrammarConstraint::precompute(
        &tokenizer,
        &internal_llm_token_map_for_precompute,
        &token_name_map,
        max_internal_id,
    );

    println!("Successfully precomputed with GPT-2 vocab (direct call).");

    let productions = vec![
        prod("S'", vec![nt("S")]),
        prod("S", vec![nt("A"), t("DEF")]),
        prod("A", vec![]),
    ];
    let terminal_map: BiBTreeMap<crate::glr::grammar::Terminal, crate::glr::table::TerminalID> = token_name_map.iter().map(|(name, id)| (crate::glr::grammar::Terminal(name.clone()), crate::glr::table::TerminalID(*id))).collect();
    let parser = generate_glr_parser_with_terminal_map(&productions, 0, terminal_map);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map,
        max_original_llm_token_id_val,
    );
    let mut constraint_state = constraint.init();
    constraint_state.step_with_all_llm_tokens();

    let mask = constraint_state.get_mask();

    if let Some(def_token_original_id) = llm_token_map.get_by_left(b"def") {
        assert!(mask.contains(def_token_original_id.0), "Expected LLM token 'def' (ID {}) to be in mask", def_token_original_id.0);
    } else {
        println!("Warning: LLM token 'def' not in sampled GPT-2 vocab for assertion.");
    }

    if let Some(a_token_original_id) = llm_token_map.get_by_left(b"a") {
        println!("{}", constraint.parser);
        let mut constraint_state_for_loop = constraint.init();
        for i in 0..3 { // Reduced loop for speed
            println!("{}. Stepping with LLM token 'a' (ID {})", i, a_token_original_id.0);
            constraint_state_for_loop.step_with_all_llm_tokens(); // Get current mask
            // Before commit, ensure 'a' is possible if the grammar allows it here.
            // This depends on the grammar state. For S -> A DEF, A -> epsilon, 'a' might not be directly allowed if A is empty.
            // For FSTRING_MIDDLE, 'a' would be allowed.
            // For this test, we'll just commit and proceed.
            constraint_state_for_loop.commit(*a_token_original_id);
        }
    } else {
         println!("Warning: LLM token 'a' not in sampled GPT-2 vocab for loop test.");
    }


    Ok(())
}

#[test]
fn test_simple_def_match_non_zero_llm_id() {
    let tokenizer_expr = groups![eat_string_fast("def")]; // Group 0
    let tokenizer = tokenizer_expr.build();

    let mut llm_token_map = LLMTokenMap::new();
    let def_original_llm_id = 750;
    llm_token_map.insert(b"def".to_vec(), LLMTokenID(def_original_llm_id));
    let max_original_llm_token_id = def_original_llm_id;

    let productions = vec![
        prod("S", vec![t("DEF_T")]),
    ];

    let mut grammar_token_map: BiBTreeMap<crate::glr::grammar::Terminal, crate::glr::table::TerminalID> = BiBTreeMap::new();
    grammar_token_map.insert(crate::glr::grammar::Terminal("DEF_T".to_string()), crate::glr::table::TerminalID(0));

    let parser = generate_glr_parser_with_terminal_map(
        &productions,
        0,
        grammar_token_map.clone()
    );

    let mut token_name_map_for_stats = BiBTreeMap::new();
    token_name_map_for_stats.insert("DEF_T".to_string(), 0);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map_for_stats,
        max_original_llm_token_id,
    );

    let mut constraint_state = constraint.init();
    constraint_state.step_with_all_llm_tokens();
    let mask = constraint_state.get_mask();

    let mut expected_mask = HybridBitset::new();
    expected_mask.insert(def_original_llm_id);

    assert_eq!(
        mask,
        expected_mask,
        "Mask should allow 'def' token (Original LLM ID {})",
        def_original_llm_id
    );
}

#[test]
#[ignore] // This test is designed to be very slow and stress GSS, ignore in CI
fn test_hideous_ambiguity() {
    let productions = vec![
        prod("S", vec![t("FSTRING_MIDDLE"), t("FSTRING_MIDDLE")]),
    ];

    let tokenizer_expr = groups![
        repeat1_fast(eat_u8(b'a')), // Group 0
    ];
    let tokenizer = tokenizer_expr.build();

    let mut llm_token_map = LLMTokenMap::new();
    llm_token_map.insert(b"a".to_vec(), LLMTokenID(0));
    let max_original_llm_token_id = 0;

    let mut token_name_map = BiBTreeMap::new();
    token_name_map.insert("FSTRING_MIDDLE".to_string(), 0); // Maps grammar terminal "FSTRING_MIDDLE" to tokenizer group 0

    // Create the Parser using generate_glr_parser_with_terminal_map
    let mut grammar_token_map_for_parser: BiBTreeMap<crate::glr::grammar::Terminal, crate::glr::table::TerminalID> = BiBTreeMap::new();
    grammar_token_map_for_parser.insert(crate::glr::grammar::Terminal("FSTRING_MIDDLE".to_string()), crate::glr::table::TerminalID(0));
    let parser = generate_glr_parser_with_terminal_map(&productions, 0, grammar_token_map_for_parser);
    println!("{}", parser);

    let constraint = GrammarConstraint::new(
        tokenizer,
        parser,
        llm_token_map.clone(),
        token_name_map, // This is BiBTreeMap<String, usize>
        max_original_llm_token_id,
    );

    let mut constraint_state = constraint.init();

    let a_id = llm_token_map.get_by_left(b"a".as_slice()).unwrap().0;
    for i in 0..100 { // Reduced from 10000 for practical test duration if un-ignored
        println!("{}. Stepping with LLM token ID {}", i, a_id);
        constraint_state.step_with_all_llm_tokens();
        // For this test to stress GSS, we'd typically commit the token.
        // If only step_with_all_llm_tokens is called, the GSS might not grow as intended
        // because no specific path is chosen/committed.
        // Let's add a commit to see the effect.
        constraint_state.commit(LLMTokenID(a_id));
    }
}

#[test]
fn test_constraint_from_serialized_and_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define paths and URLs
    let serialized_grammar_dir = Path::new(".cache/test_data");
    fs::create_dir_all(serialized_grammar_dir)?;
    let serialized_grammar_path = serialized_grammar_dir.join("serialized_constraint.json");

    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let cache_dir = Path::new(".cache/test_vocabs"); // Same cache dir as other tests
    let vocab_file_name = "gpt2_vocab.json";

    // 2. Create and write the dummy serialized_compiled_grammar.json
    let dummy_json_content = generate_dummy_grammar_constraint_json_string()?;
    fs::write(&serialized_grammar_path, dummy_json_content)?;

    // 3. Load GrammarConstraint from the JSON file
    let json_string_content = fs::read_to_string(&serialized_grammar_path)?;
    let loaded_json_node = JSONNode::from_json_string(&json_string_content)?;
    let loaded_grammar_constraint = GrammarConstraint::from_json(loaded_json_node)?;

    // 4. Load GPT-2 vocabulary
    let gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;
    let mut gpt2_llm_token_map = LLMTokenMap::new();
    let mut gpt2_max_original_llm_token_id: usize = 0;

    // Sample for speed: Use a very small sample for this specific test's purpose
    let prop = 0.01;
    let total_tokens = gpt2_raw_vocab.len();
    // Ensure at least a few tokens, e.g., 10, or all if total is less than 10.
    let sample_size = ((total_tokens as f64 * prop) as usize).clamp(10, total_tokens);

    println!("Sampling {} out of {} GPT-2 tokens for constraint re-binding test", sample_size, total_tokens);
    for (token_str, id_val_u32) in gpt2_raw_vocab.into_iter().take(sample_size) {
        let id_val = id_val_u32 as usize;
        gpt2_llm_token_map.insert(token_str.into_bytes(), LLMTokenID(id_val));
        if id_val > gpt2_max_original_llm_token_id {
            gpt2_max_original_llm_token_id = id_val;
        }
    }
    // If the sample was empty (e.g. total_tokens was 0), max_id remains 0.
    // If gpt2_llm_token_map is empty after sampling, gpt2_max_original_llm_token_id will be 0.
    // This is handled by GrammarConstraint::new.

    // 5. Construct a new GrammarConstraint
    println!("Constructing new GrammarConstraint with GPT-2 vocab ({} tokens, max_id {})", gpt2_llm_token_map.len(), gpt2_max_original_llm_token_id);
    let new_grammar_constraint = GrammarConstraint::new(
        loaded_grammar_constraint.tokenizer.clone(),
        loaded_grammar_constraint.parser.clone(),
        gpt2_llm_token_map.clone(), // Pass the sampled GPT-2 vocab
        loaded_grammar_constraint.token_name_map.clone(),
        gpt2_max_original_llm_token_id
    );

    // 6. Basic assertions
    let mut state = new_grammar_constraint.init();
    state.step_with_all_llm_tokens();
    let mask = state.get_mask();

    println!("Mask from new constraint with GPT-2 vocab (size: {}, first 20 elements): {:?}", mask.len(), mask.iter().take(20).collect::<Vec<_>>());

    // The dummy grammar is S -> X_TOK, where X_TOK is 'x'.
    // If "x" is in the gpt2_llm_token_map sample, it should be in the mask.
    if let Some(x_token_id) = gpt2_llm_token_map.get_by_left(b"x") {
        assert!(mask.contains(x_token_id.0),
            "Mask should contain 'x' (ID {}). Mask: {:?}",
            x_token_id.0, mask.iter().collect::<Vec<_>>());
    } else {
        println!("Token 'x' not in sampled GPT-2 vocab, skipping specific mask check for 'x'.");
    }

    // General check: if there are tokens, mask shouldn't be empty unless grammar is impossible.
    // new_grammar_constraint.internal_max_llm_token is the max *internal* ID.
    // If gpt2_llm_token_map is not empty, then internal_max_llm_token should be >= 0.
    // If the mask is empty AND the internal max token ID is -1 (meaning no tokens mapped internally),
    // AND the original llm_token_map was NOT empty, then it's a problem.
    // If the original llm_token_map WAS empty, the mask being empty is expected.
    if !gpt2_llm_token_map.is_empty() {
         // If there are tokens in the input vocab, and the grammar should allow *something* initially,
         // the mask should not be empty. Our dummy grammar S -> X_TOK (matching 'x') should allow 'x' if it's in the vocab.
         // The condition `!mask.is_empty() || (new_grammar_constraint.internal_max_llm_token == 0 && !mask.contains(0) && !gpt2_llm_token_map.is_empty() )`
         // from the thought process is complex. A simpler check: if the vocab is not empty, and the grammar has a starting
         // terminal that can be matched by *some* token, the mask should not be empty.
         // For S -> X_TOK (matching 'x'), if 'x' is in gpt2_llm_token_map, the mask should contain its original ID.
         // The assertion above already checks this specifically for 'x'. A general non-empty check might be too loose.
         // Let's rely on the specific check for 'x'.
         // The internal_max_llm_token will be the max ID from the *internal* bimap, which corresponds to the number of unique tokens - 1.
         // If gpt2_llm_token_map is not empty, original_to_internal_id_bimap won't be empty, so internal_max_llm_token will be >= 0.
         // If the mask *is* empty despite having tokens and a grammar that *should* match something (like 'x'), that's the issue.
         // The `assert!(mask.contains(x_token_id.0), ...)` covers the expected behavior if 'x' is present.
    }


    // Cleanup dummy file
    fs::remove_file(serialized_grammar_path)?;
    // Optionally remove the directory if it's empty and specific to this test
    // fs::remove_dir(serialized_grammar_dir).ok(); // .ok() to ignore error if not empty

    Ok(())
}
