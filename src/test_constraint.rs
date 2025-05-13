use std::collections::BTreeMap;
use crate::finite_automata::eat_u8;
use crate::{choice, choice_fast, groups, seq, seq_fast};
use crate::glr::grammar::{nt, prod, t, NonTerminal, Terminal};
use crate::glr::table::{generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map};
use crate::datastructures::hybrid_bitset::HybridBitset; // Explicitly import HybridBitset
use std::hash::{Hash, Hasher};
use crate::interface::{eat_u8_fast, eat_u8_negation_fast, eat_u8_range_fast, repeat0_fast, eat_any_fast, eat_string_fast}; // Added eat_any_fast

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use bimap::BiBTreeMap;
use reqwest::blocking;
use serde_json;
use crate::constraint::GrammarConstraint;
use crate::datastructures::trie::Trie;
// Already a main dependency, but good to be explicit if used directly
// reqwest will be used if the file isn't cached, ensure it's in dev-dependencies
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use crate::types::TerminalID;

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
        7, // max_llm_token_id should be 7 for IDs 0-6
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

    // // Commit "(i"
    // state.commit(LLMTokenID(5));
    // state.step_with_all_llm_tokens();
    // state.commit(LLMTokenID(4)); // Assuming ")"
    // state.step_with_all_llm_tokens();
    // let mask = state.get_mask();
    // assert_eq!(mask, HybridBitset::from_iter(vec![1, 2, 5, 6, 3])); // Expect '+', '*', '(', '(i', '+i'

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
        internal_num_llm_tokens, // Pass the number of tokens
    );
    // print_precomputed(&precomputed);
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
        internal_num_llm_tokens, // Pass the number of tokens
    );
    // print_precomputed(&precomputed);
    println!("Done precomputing");
}

#[test]
fn test_precompute_with_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define tokenizer: matches anything
    // The tokenizer will have one group (ID 0)
    let tokenizer_expr = groups![
        repeat0_fast(eat_any_fast()),
        repeat0_fast(eat_any_fast()),
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

    // Manually perform mapping for the test, similar to setup_llm_token_mappings
    // We need a map from bytes to internal IDs (0..N-1 sequence based on sorted bytes)
    let mut sorted_tokens_for_test: Vec<(Vec<u8>, LLMTokenID)> = llm_token_map
        .iter()
        .map(|(bytes, original_id)| (bytes.clone(), *original_id))
        .collect();
    sorted_tokens_for_test.sort_by(|(bytes_a, _), (bytes_b, _)| bytes_a.cmp(bytes_b));

    let mut test_internal_llm_token_map = BiBTreeMap::new(); // bytes -> internal LLMTokenID
    let mut internal_id_counter_for_test = 0;

    for (bytes, _original_llm_id) in sorted_tokens_for_test { // original_llm_id not directly used to make internal map
        let internal_llm_id = LLMTokenID(internal_id_counter_for_test);
        test_internal_llm_token_map.insert(bytes.clone(), internal_llm_id);
        internal_id_counter_for_test += 1;
    }
    let test_internal_num_llm_tokens = internal_id_counter_for_test;


    // 3. Create token_name_map for grammar tokens
    // Our tokenizer has one grammar token (GroupID 0)
    let mut token_name_map: BiBTreeMap<String, usize> = BiBTreeMap::new();
    token_name_map.insert("ANYTHING_GRAMMAR_TOKEN".to_string(), 0 as usize); // GrammarTokenID 0
    token_name_map.insert("ANYTHING_GRAMMAR_TOKEN2".to_string(), 1 as usize); // GrammarTokenID 0
    token_name_map.insert("DEF".to_string(), 2 as usize); // GrammarTokenID 0

    // 4. Call precompute
    println!(
        "Starting precompute with GPT-2 vocab ({} tokens, max_original_id_val: {}, internal_num_tokens: {})...",
        llm_token_map.len(),
        max_llm_token_id_val, // Max original ID value encountered
        test_internal_num_llm_tokens // Number of unique internal tokens for precompute
    );

    // This is the main part of the test: ensure it runs without error.
    let _precomputed = GrammarConstraint::precompute(
        &tokenizer,
        &test_internal_llm_token_map,
        &token_name_map,
        test_internal_num_llm_tokens,
    );

    println!("Successfully precomputed with GPT-2 vocab.");

    // 2. Create a parser
    let productions = vec![
        prod("S", vec![t("DEF")]),
    ];
    let terminal_map: BiBTreeMap<Terminal, TerminalID> = token_name_map.iter().map(|(name, id)| (Terminal(name.clone()), TerminalID(*id))).collect();
    let parser = generate_glr_parser_with_terminal_map(&productions, 0, terminal_map);

    // Ensure that the letter "d" is a valid initial LLM token
    let max_llm_token_id = token_name_map.iter().map(|(_, id)| *id).max().unwrap();
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

    let d_id = llm_token_map.get_by_left(&b"d"[..]).unwrap().0;
    assert!(mask[d_id], "Mask should contain ID for 'd'");

    Ok(())
}
