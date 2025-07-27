// This file contains integration tests for the GrammarConstraint using a full JavaScript grammar.
// It is structured as follows:
// - Helper functions for loading vocabularies and debugging.
// - Integration tests for the JavaScript grammar:
//   - `test_js_glr_parser_sanity_checks`: Verifies the GLR parser with specific, hand-crafted sequences of grammar terminals.
//   - `test_js_glr_parser_fuzzing`: A fuzz test that feeds random sequences of terminals to the GLR parser to check for panics.
//   - `test_js_constraint_integration`: A full-stack test that loads a real vocabulary (GPT-2), tokenizes a JS file, and steps through it with the GrammarConstraint, checking the allowed token mask at each step.
// - A `minimizer` module containing a tool to find minimal reproducing examples for bugs in the grammar or parser.
// - Tests that use the minimizer, which are ignored by default as they are for debugging specific issues.

use crate::constraint::GrammarConstraint;
use crate::glr::grammar::{nt, prod, t, regex, NonTerminal, Production, Symbol, Terminal, literal};
use crate::glr::parser::GLRParserState;
use crate::glr::stats::get_stats;
use crate::glr::table::{assign_non_terminal_ids, assign_terminal_ids, generate_glr_parser_with_maps, StateID};
use crate::interface::{CompiledGrammar, GrammarDefinition};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::tokenizer::{LLMTokenID, LLMTokenMap, TokenizerStateID};
use crate::types::TerminalID;
use crate::datastructures::gss::{GSSNode, sample_path};
use crate::datastructures::vocab_prefix_tree::VocabPrefixTree;
use bimap::BiBTreeMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use reqwest::blocking;
use similar::TextDiff;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use rand::prelude::IndexedRandom;
use crate::profiler::{print_summary, print_summary_flat, reset};
// --- Helper Functions ---

/// Loads a vocabulary from a JSON file, downloading it if not present in the cache.
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

/// Helper function to print context around a token in a larger text.
fn print_token_context(
    full_text: &str,
    all_lines: &[&str],
    token_start_global_byte: usize,
    token_end_global_byte: usize, // Exclusive end
    context_lines_count: usize,
) {
    if all_lines.is_empty() {
        println!("    Context: (empty or malformed input lines)");
        println!("    ----");
        return;
    }

    let mut current_scan_byte_offset = 0;
    let mut token_start_line_idx = 0;
    let mut token_start_col_byte_in_line = 0;
    let mut token_end_line_idx = 0;
    let mut token_end_col_byte_in_line = 0;

    // Determine start line and column
    let mut found_start_line = false;
    for (idx, line_content) in all_lines.iter().enumerate() {
        let line_start_byte_offset = current_scan_byte_offset;
        let line_content_end_byte_offset = line_start_byte_offset + line_content.len();

        if !found_start_line && token_start_global_byte >= line_start_byte_offset && token_start_global_byte <= line_content_end_byte_offset {
            token_start_line_idx = idx;
            token_start_col_byte_in_line = token_start_global_byte - line_start_byte_offset;
            found_start_line = true;
        }

        current_scan_byte_offset += line_content.len();
        if idx < all_lines.len() - 1 {
            current_scan_byte_offset += 1; // Account for '\n'
        }
    }

    // Determine end line and column
    current_scan_byte_offset = 0;
    for (idx, line_content) in all_lines.iter().enumerate() {
        let line_start_byte_offset = current_scan_byte_offset;
        let line_content_end_byte_offset = line_start_byte_offset + line_content.len();

        if token_end_global_byte > line_start_byte_offset && token_end_global_byte <= line_content_end_byte_offset {
            token_end_line_idx = idx;
            token_end_col_byte_in_line = token_end_global_byte - line_start_byte_offset;
            break;
        }
        if idx < all_lines.len() - 1 && token_end_global_byte == line_content_end_byte_offset + 1 {
            token_end_line_idx = idx;
            token_end_col_byte_in_line = line_content.len();
            break;
        }

        current_scan_byte_offset += line_content.len();
        if idx < all_lines.len() - 1 {
            current_scan_byte_offset += 1; // Account for '\n'
        }
    }
     if token_end_global_byte == full_text.len() && !full_text.ends_with('\n') && !full_text.is_empty() {
        token_end_line_idx = all_lines.len() - 1;
        token_end_col_byte_in_line = all_lines.last().map_or(0, |s| s.len());
    }

    let display_start_line = token_start_line_idx.saturating_sub(context_lines_count);
    let display_end_line = (token_end_line_idx + context_lines_count).min(all_lines.len().saturating_sub(1));

    println!("    Context Highlight (Token bytes [{}, {})):", token_start_global_byte, token_end_global_byte);
    for i in display_start_line..=display_end_line {
        let line_content = all_lines[i];
        println!("{:5} | {}", i + 1, line_content);

        if i >= token_start_line_idx && i <= token_end_line_idx {
            let start_col = if i == token_start_line_idx { token_start_col_byte_in_line } else { 0 };
            let end_col = if i == token_end_line_idx { token_end_col_byte_in_line } else { line_content.len() };
            
            let effective_start_col = start_col.min(line_content.len());
            let effective_end_col = end_col.min(line_content.len()).max(effective_start_col);

            if effective_end_col > effective_start_col {
                let prefix = " ".repeat(effective_start_col);
                let carets = "^".repeat(effective_end_col - effective_start_col);
                println!("      | {}{}", prefix, carets);
            } else if token_start_global_byte == token_end_global_byte && i == token_start_line_idx && effective_start_col <= line_content.len() {
                let prefix = " ".repeat(effective_start_col);
                println!("      | {}{}", prefix, "^");
            }
        }
    }
    println!("    ------------------------------------------");
}

// --- Main Integration Tests ---

#[test]
fn test_js_constraint_integration() -> Result<(), Box<dyn std::error::Error>> {
    // --- Setup Phase ---
    println!("--- Setting up for JS Constraint Integration Test ---");

    // 1. Load and compile the JavaScript grammar from EBNF.
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    println!("Compiling GrammarDefinition into CompiledGrammar...");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Successfully compiled GrammarDefinition.");
    println!("{}", compiled_grammar);

    // 2. Load the GPT-2 vocabulary.
    println!("\nLoading GPT-2 vocabulary...");
    let cache_dir = Path::new(".cache/test_vocabs");
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let vocab_file_name = "gpt2_vocab.json";
    let gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;
    
    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;
    for (token_str, id_val_u32) in gpt2_raw_vocab {
        let id_val = id_val_u32 as usize;
        let token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        let token_bytes = token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(id_val));
        max_original_llm_token_id_val = max_original_llm_token_id_val.max(id_val);
    }
    println!("GPT-2 vocab loaded ({} tokens, max_original_id: {}).", llm_token_map.len(), max_original_llm_token_id_val);

    if true { // Manual vocabulary modifications for debugging
        println!("\n--- Applying manual vocabulary modifications ---");

        // Filter 1: Keep only tokens with length <= x
        let x = 4;
        llm_token_map.retain(|bytes, _| bytes.len() <= x);
        println!("  - After length filter (<= {x}): {} tokens remaining.", llm_token_map.len());

        // Filter 2: Keep only tokens where all alphabetic chars are 'a'
        // llm_token_map.retain(|bytes, _| {
        //     bytes.iter().all(|&b| {
        //         if b.is_ascii_alphabetic() {
        //             b.to_ascii_lowercase() == b'a'
        //         } else {
        //             true
        //         }
        //     })
        // });
        // println!("  - After 'a'-only alphabetic filter: {} tokens remaining.", llm_token_map.len());

        // Option 3: Set to a few specific tokens
        let mut specific_tokens = LLMTokenMap::new();
        specific_tokens.insert(b"[".to_vec(), LLMTokenID(324));
        specific_tokens.insert(b"x".to_vec(), LLMTokenID(325));
        specific_tokens.insert(b"]:".to_vec(), LLMTokenID(327));
        specific_tokens.insert(b"]".to_vec(), LLMTokenID(328));
        // // specific_tokens.insert(b":".to_vec(), LLMTokenID(329));
        specific_tokens.insert(b"&".to_vec(), LLMTokenID(323));
        // // specific_tokens.insert(b" =".to_vec(), LLMTokenID(323));
        // // specific_tokens.insert(b" 1".to_vec(), LLMTokenID(290));
        // // specific_tokens.insert(b";".to_vec(), LLMTokenID(26));
        // llm_token_map = specific_tokens;
        println!("  - Set to a specific small set of tokens: {} tokens.", llm_token_map.len());

        // llm_token_map.retain(|v, _| [b"x".as_ref(), b"[", b"]", b"]:", b" &"].contains(&v.as_ref()));

        // After filtering, we must recalculate max_original_llm_token_id_val
        max_original_llm_token_id_val = llm_token_map.right_values().map(|id| id.0).max().unwrap_or(0);
        println!("  - Recalculated max original token ID: {}", max_original_llm_token_id_val);
        println!("--- Finished manual vocabulary modifications ---\n");
    }

    // 3. Construct the GrammarConstraint.
    let dummy_eof_placeholder = 0;
    println!("\nConstructing GrammarConstraint...");
    let grammar_constraint = GrammarConstraint::from_compiled_grammar(
        compiled_grammar.clone(),
        llm_token_map.clone(),
        LLMTokenID(dummy_eof_placeholder),
        max_original_llm_token_id_val
    );
    grammar_constraint.dump_precomputed();
    println!("GrammarConstraint constructed successfully.");

    // --- Tokenization Phase ---
    
    // 4. Tokenize a sample JS file using a VocabPrefixTree built from the LLM vocab.
    let example_code_path = "src/example_code.js";
    let full_text_to_tokenize = fs::read_to_string(example_code_path)?;
    
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = grammar_constraint.llm_vocab.llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    let mut test_token_sequence_ids = Vec::new();
    let mut tokenized_strs_for_logging = Vec::new();
    let mut text_to_process = full_text_to_tokenize.as_bytes();

    println!("\nTokenizing '{}' using VocabPrefixTree:", example_code_path);
    while !text_to_process.is_empty() {
        match tokenizer_vocab_tree.find_longest_prefix_token(text_to_process) {
            Some((token_id, matched_bytes)) => {
                let matched_str = String::from_utf8_lossy(matched_bytes).to_string();
                test_token_sequence_ids.push(LLMTokenID(token_id));
                tokenized_strs_for_logging.push(matched_str);
                text_to_process = &text_to_process[matched_bytes.len()..];
            }
            None => {
                panic!("Failed to tokenize. No prefix token found for remaining text: {:?}", String::from_utf8_lossy(text_to_process));
            }
        }
    }
    println!("Successfully tokenized into {} tokens.", test_token_sequence_ids.len());

    // --- Execution Phase ---
    
    // 5. Step through the tokenized file, checking the mask and committing tokens.
    let mut constraint_state = grammar_constraint.init();
    let all_code_lines: Vec<&str> = full_text_to_tokenize.lines().collect();
    let mut current_text_byte_offset = 0;

    println!("\nStepping through the token sequence with GrammarConstraint:");
    for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
        if true {
            // Reinitialize the constraint state fresh
            constraint_state = grammar_constraint.init();
            let prefix_token_ids = test_token_sequence_ids[..i].to_vec();
            let prefix_bytes: Vec<Vec<u8>> = prefix_token_ids.iter()
                .map(|&id| grammar_constraint.llm_vocab.llm_token_map.get_by_right(&id).unwrap().clone())
                .collect();
            let prefix_bytes: Vec<u8> = prefix_bytes.iter().flat_map(|b| b.clone()).collect();
            println!("Reinitializing constraint state for token {}: {:?}", i + 1, llm_token_id);
            println!("  Committing prefix bytes: {:?}", String::from_utf8_lossy(&prefix_bytes));
            constraint_state.commit_bytes(&prefix_bytes);
        }

        let current_token_str = &tokenized_strs_for_logging[i];
        println!("Processing token {}/{}: {:?} (LLMTokenID({}))", i + 1, test_token_sequence_ids.len(), current_token_str, llm_token_id.0);

        let token_start_byte = current_text_byte_offset;
        let token_end_byte = token_start_byte + current_token_str.as_bytes().len();
        print_token_context(&full_text_to_tokenize, &all_code_lines, token_start_byte, token_end_byte, 2);

        assert!(constraint_state.is_active(), "State became inactive before token {}", i + 1);

        let mask_start = Instant::now();
        let current_mask = constraint_state.get_mask();
        println!("  get_mask took: {:?}", mask_start.elapsed());

        assert!(current_mask.contains(llm_token_id.0), "Token {:?} (ID {}) not in mask at step {}", current_token_str, llm_token_id.0, i + 1);
        println!("  Token is in the mask.");

        let commit_start = Instant::now();
        constraint_state.commit(llm_token_id);
        println!("  commit took: {:?}", commit_start.elapsed());

        current_text_byte_offset = token_end_byte;
    }

    println!("\nFinished processing token sequence.");
    assert!(constraint_state.is_active(), "Final state should be active.");

    // This is a useful, but very verbose, debugging tool.
    // It checks if committing tokens one-by-one is equivalent to committing the whole prefix.
    if false {
        let mut constraint_state1 = grammar_constraint.init();
        let mut constraint_state2 = grammar_constraint.init();
        pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string());
        assert_eq!(constraint_state1.state, constraint_state2.state);
        for (i, byte) in full_text_to_tokenize.as_bytes().iter().enumerate() {
            println!("Committing byte {}: '{}'", i + 1, *byte as char);
            constraint_state1.commit_bytes(&[*byte]);
            constraint_state2.commit_bytes(&[*byte]);
            pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string());
            assert_eq!(constraint_state1.state, constraint_state2.state);
        }
    }

    Ok(())
}

#[test]
fn test_js_glr_parser_sanity_checks() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Setting up for JS GLR Parser Sanity Checks ---");
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Grammar compiled successfully.");

    let test_sequences_str = vec![
        // Valid JS sequences
        vec!["\"var\"", "IDENTIFIER", "\";\""],
        vec!["\"var\"", "IDENTIFIER", "\"=\"", "NUMERIC_LITERAL", "\";\""],
        vec!["\"if\"", "\"(\"", "IDENTIFIER", "\")\"", "\"{\""],
        vec!["IDENTIFIER", "\"||\"", "IDENTIFIER"],
        vec!["\"function\"", "IDENTIFIER", "\"(\"", "\")\"", "\"{\"","\"}\""],
        vec!["\"return\"", "STRING_LITERAL", "\";\""],
        // Sequences that should be valid prefixes
        vec!["\"var\""],
        vec!["\"var\"", "IDENTIFIER"],
        vec!["\"var\"", "IDENTIFIER", "\"=\""],
        vec!["\"if\"", "\"(\""],
        // Potentially problematic sequences
        vec!["\"=\"", "\"=\""], // "=="
        vec!["\"<\"", "\"=\""], // "<="
        vec!["\"+\"", "\"+\""], // "++"
        vec!["\"|\"", "\"|\""], // "||"
        vec!["STRING_LITERAL", "IDENTIFIER"],
        vec!["NUMERIC_LITERAL", "\"+\"", "NUMERIC_LITERAL"],
        vec!["\"...\"", "\";\"", "\"elif\""], // Known failing case from previous bug
    ];

    let mut all_sequences_passed = true;
    for (seq_idx, seq_terminal_names) in test_sequences_str.iter().enumerate() {
        let mut terminal_id_sequence = Vec::new();
        let mut sequence_is_valid = true;
        for token_name in seq_terminal_names {
            if let Some(terminal_id) = compiled_grammar.glr_parser.terminal_map.get_by_left(&regex(token_name)) {
                terminal_id_sequence.push(*terminal_id);
            } else {
                println!("  Warning: Terminal name '{}' not found in grammar for sequence {}. Skipping.", token_name, seq_idx);
                sequence_is_valid = false;
                break;
            }
        }

        if !sequence_is_valid {
            all_sequences_passed = false;
            continue;
        }

        let mut glr_state = compiled_grammar.glr_parser.init_glr_parser(None);
        let mut sequence_parse_ok = true;
        for (token_idx, &grammar_token_id) in terminal_id_sequence.iter().enumerate() {
            glr_state.step(grammar_token_id);
            if !glr_state.is_ok() {
                println!("  FAILED at token #{} ('{}') in sequence: '{}'", token_idx + 1, seq_terminal_names[token_idx], seq_terminal_names.join(" → "));
                sequence_parse_ok = false;
                all_sequences_passed = false;
                break;
            }
        }

        if sequence_parse_ok {
            println!("  PASSED sequence: '{}'", seq_terminal_names.join(" → "));
        }
    }

    assert!(all_sequences_passed, "One or more grammar terminal sequence tests failed.");
    Ok(())
}

#[test]
#[ignore] // Fuzzing can be slow and is not for regular CI.
fn test_js_glr_parser_fuzzing() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Setting up for JS GLR Parser Fuzzing ---");
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Grammar compiled successfully.");

    let num_fuzz_iterations = 1000;
    let max_tokens_per_fuzz_attempt = 50;
    let all_grammar_terminal_ids: Vec<_> = compiled_grammar.glr_parser.terminal_map.right_values().cloned().collect();

    if all_grammar_terminal_ids.is_empty() {
        println!("  Warning: No grammar terminals found. Fuzz test is trivial.");
        return Ok(());
    }

    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..num_fuzz_iterations {
        if i % 100 == 0 {
            println!("  Fuzz test iteration {}/{}", i, num_fuzz_iterations);
        }
        let mut glr_state = compiled_grammar.glr_parser.init_glr_parser(None);
        let num_tokens = rng.gen_range(1..=max_tokens_per_fuzz_attempt);
        for _ in 0..num_tokens {
            let random_terminal_id = all_grammar_terminal_ids.choose(&mut rng).unwrap();
            glr_state.step(*random_terminal_id);
            // The test passes if it doesn't panic. We don't care if the parse is valid.
        }
    }
    println!("GLR parser fuzz test completed ({} iterations) without panics.", num_fuzz_iterations);
    Ok(())
}

#[test]
fn test_js_parser_direct_feed_for_phase3_debug() -> Result<(), Box<dyn std::error::Error>> {
    // This test is designed to investigate performance issues with a large number of
    // phase 3 reductions, by bypassing the GrammarConstraint and tokenization layers
    // and feeding a sequence of grammar terminals directly to the GLR parser.

    // 1. Load and compile the JavaScript grammar.
    println!("--- Setting up for JS GLR Parser Direct Terminal Feed Test ---");
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Grammar compiled successfully.");

    let parser = &compiled_grammar.glr_parser;

    // 2. Define the terminal sequence for "let x = 1111111111;".
    // The tokenizer would produce: "let", IDENTIFIER, "=", NUMERIC_LITERAL, ";"
    // The IGNORE rule handles whitespace.
    let terminals = vec![
        // "\"let\"",
        // "IDENTIFIER",
        // "\"=\"",
        // "NUMERIC_LITERAL",
        // "\";\"",
        literal(b"let"),
        regex("IDENTIFIER"),
        literal(b"="),
        regex("NUMERIC_LITERAL"),
        literal(b";"),
    ];

    // 3. Convert terminal names to TerminalIDs.
    let mut terminal_ids = Vec::new();
    for terminal_obj in &terminals {
        let terminal_id = parser.terminal_map.get_by_left(&terminal_obj)
            .unwrap_or_else(|| {
                eprintln!("Terminals in parser's terminal map:");
                for (name, id) in &parser.terminal_map {
                    eprintln!("  - '{}' ({:?}, ID {})", name, name, id.0);
                }
                panic!("Terminal '{}' ({:?}) not found in parser's terminal map.", terminal_obj, terminal_obj);
            });
        terminal_ids.push(*terminal_id);
    }

    println!("Terminal sequence to parse: {:?}", terminals);
    println!("Corresponding Terminal IDs: {:?}", terminal_ids.iter().map(|id| id.0).collect::<Vec<_>>());

    // Reset the profiler
    reset();

    // 4. Initialize the parser state and step through the terminals.
    let mut glr_state = parser.init_glr_parser(None);
    for (i, &terminal_id) in terminal_ids.iter().enumerate() {
        println!("\n--- Stepping with terminal {} ({:?}) ---", i, parser.terminal_map.get_by_right(&terminal_id).unwrap());
        glr_state.step(terminal_id);
        glr_state.do_phase3();
        // Print profiler stats after each step
        print_summary_flat();
        print_summary();
        reset();
        assert!(glr_state.is_ok(), "Parser state became invalid after terminal {} ({:?})", i, terminals[i]);
    }

    println!("\n--- Finished parsing terminal sequence ---");
    assert!(glr_state.is_ok(), "Final parser state is not OK.");

    Ok(())
}