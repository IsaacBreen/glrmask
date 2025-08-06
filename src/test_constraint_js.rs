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
use crate::glr::grammar::{literal, nt, prod, t, regex_name, NonTerminal, Production, Symbol, Terminal};
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
use crate::constraint_extra::dump_precompute_trie_recursive;
use crate::profiler::{self, print_summary, print_summary_flat, reset};
// --- Helper Functions ---

/// Loads a vocabulary from a JSON file, downloading it if not present in the cache.
fn load_or_download_gpt2_vocab(
    cache_dir: &Path,
    file_name: &str,
    url: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    let cache_path = cache_dir.join(file_name);

    let vocab_map: BTreeMap<String, u32> = if cache_path.exists() {
        println!("Loading GPT-2 vocab from cache: {:?}", cache_path);
        let file = File::open(cache_path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    } else {
        println!("Downloading GPT-2 vocab from: {}", url);
        let response = blocking::get(url)?.error_for_status()?;
        let content = response.text()?;

        let mut file = File::create(&cache_path)?;
        file.write_all(content.as_bytes())?;
        println!("Saved GPT-2 vocab to cache: {:?}", cache_path);

        serde_json::from_str(&content)?
    };
    Ok(vocab_map.into_keys().collect())
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

#[test]
fn test_js_parser_reduction_explosion_isolated() -> Result<(), Box<dyn std::error::Error>> {
    // This test isolates the GLR parser to investigate situations where an *ambiguous first
    // token* is followed by a *second token* that triggers a large amount of work (many
    // reductions). It now iterates through all configurations of enabling/disabling
    // `process_default_reductions` at key points.
    
    // ------------------------------------------------------------------------
    // Helper utilities
    // ------------------------------------------------------------------------
    /// Convenience wrapper that:
    ///   1. Resets the profiler,
    ///   2. Executes `action`,
    ///   3. Prints the number of `GLRParserState::reduce_and_goto` hits,
    ///   4. Asserts that the hit-count is not excessive, logging the state on failure.
    ///   5. Returns that hit-count.
    fn measure<'a, F: FnOnce(&mut GLRParserState<'a>)>(label: &str, state: &mut GLRParserState<'a>, action: F) -> u64 {
        let state_before = state.clone();
        profiler::reset();
        action(state);
        let hits = profiler::get_all_hits();
        let reduce_hits = hits
            .get("GLRParserState::reduce_and_goto")
            .copied()
            .unwrap_or(0);
        println!("  - {:<50}: reduce hits = {}", label, reduce_hits);
        if reduce_hits > 200 {
            state_before.log_gss(&format!("State BEFORE action '{}' that caused assertion failure", label), TerminalID(0), false, false);
            profiler::print_summary();
            panic!("Too many reductions ({}) for action '{}'.", reduce_hits, label);
        }
        reduce_hits
    }

    /// Returns the list of candidate tokens we want to try as “first” or “second” tokens.
    fn candidate_tokens() -> Vec<Terminal> {
        vec![
            regex_name("BIGINT_LITERAL"),
            regex_name("BOOLEAN_LITERAL"),
            regex_name("ELISION"),
            regex_name("IDENTIFIER"),
            regex_name("IGNORE"),
            regex_name("NUMERIC_LITERAL"),
            regex_name("STRING_LITERAL"),
            regex_name("TEMPLATE_CHAR"),
            literal(b"`"),
            regex_name("REGULAR_EXPRESSION_LITERAL"),
            literal(b"{"),
            literal(b"("),
            literal(b"["),
            literal(b"+"),
            literal(b"-"),
            literal(b"++"),
            literal(b"--"),
            literal(b"~"),
            literal(b"!"),
        ]
    }

    // ------------------------------------------------------------------------
    // 1. Load and compile the JavaScript grammar.
    // ------------------------------------------------------------------------
    println!("--- Setting up for JS GLR Parser Reduction Explosion Test ---");
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let parser = &compiled_grammar.glr_parser;
    println!("Grammar compiled successfully.");

    // 2. Assemble the candidate token lists.
    let first_tokens_to_test = candidate_tokens();
    let second_tokens_to_test = candidate_tokens();

    // Iterate through all 16 configurations of process_default_reductions calls.
    for i in 0..16 {
        let config_pdr_initial = (i & 1) != 0;
        let config_pdr_after_first_token = (i & 2) != 0;
        let config_pdr_after_merge = (i & 4) != 0;
        let config_pdr_after_second_token = (i & 8) != 0;

        println!("\n\n======================================================================");
        println!(
            "Testing config {}/16: initial={}, after_first={}, after_merge={}, after_second={}",
            i + 1,
            config_pdr_initial,
            config_pdr_after_first_token,
            config_pdr_after_merge,
            config_pdr_after_second_token
        );
        println!("======================================================================\n");

        // 3. Initialize the parser and collect states after feeding each of the first tokens.
        let mut initial_state = parser.init_glr_parser(None);
        let mut successful_first_step_states = Vec::new();

        if config_pdr_initial {
            measure("Processed default reductions (initial)", &mut initial_state, |s| {
                s.process_default_reductions()
            });
        }

        println!("\n--- Phase 1: Feeding initial tokens individually ---");
        for terminal in &first_tokens_to_test {
            println!("\n--- Testing terminal '{}' ---", terminal);
            let terminal_id = *parser.terminal_map.get_by_left(terminal).unwrap();

            let mut state_clone = initial_state.clone();

            // measure(&format!("Fed token '{terminal:?}'"), &mut state_clone, |s| {
                state_clone.step(terminal_id);
            // });

            if config_pdr_after_first_token {
                measure("Processed default reductions (after first token)", &mut state_clone, |s| {
                    s.process_default_reductions()
                });
            }

            if state_clone.is_ok() {
                successful_first_step_states.push(state_clone);
            } else {
                println!("    -> State is not OK after this token.");
            }
        }

        // 4. Merge all successful states into one.
        println!("\n--- Phase 2: Merging {} successful states ---", successful_first_step_states.len());
        assert!(!successful_first_step_states.is_empty(), "No first tokens resulted in a valid parser state for config {}", i);

        // Include the initial state
        let mut merged_state = initial_state.clone();
        // let mut merged_state = successful_first_step_states.pop().unwrap();
        for state in successful_first_step_states {
            merged_state.merge_with(state);
        }
        merged_state.log_gss("Merged states", TerminalID(0), false, false);
        println!("States merged successfully.");

        if config_pdr_after_merge {
            measure("Processed default reductions (after merge)", &mut merged_state, |s| {
                s.process_default_reductions()
            });
        }

        // 5. Feed various second tokens to the merged state and check reductions.
        println!("\n--- Phase 3: Feeding second tokens to merged state ---");

        for second_token_terminal in &second_tokens_to_test {
            let second_token_id = *parser.terminal_map.get_by_left(second_token_terminal).unwrap();

            println!("\n--- Testing second token: '{}' ---", second_token_terminal);
            let mut state_for_second_token = merged_state.clone();

            // 5a. Feed the second token to the cloned merged state and measure.
            // measure(&format!("Fed token '{second_token_terminal:?}'"), &mut state_for_second_token, |s| {
                state_for_second_token.step(second_token_id);
            // });
            // assert!(
            //     state_for_second_token.is_ok(),
            //     "Merged state became invalid after second token '{}' for config {}.",
            //     second_token_terminal,
            //     i
            // );

            // 5b. Process default reductions.
            if config_pdr_after_second_token {
                measure("Processed default reductions (after second token)", &mut state_for_second_token, |s| {
                    s.process_default_reductions()
                });
            }
        }
    }

    println!("\nTest passed: All configurations ran without assertion failures.");
    Ok(())
}

#[test]
fn test_js_parser_reduction_explosion_simplified() -> Result<(), Box<dyn std::error::Error>> {
    // This test is a simplified version of `test_js_parser_reduction_explosion_isolated`.
    // It focuses on a single ambiguous token (`BOOLEAN_LITERAL`) to investigate reduction
    // behavior when states are merged and processed further. It iterates through all 16
    // configurations of enabling/disabling `process_default_reductions` at key points.

    // Helper to measure reduction hits
    fn measure<'a, F: FnOnce(&mut GLRParserState<'a>)>(label: &str, state: &mut GLRParserState<'a>, action: F) -> u64 {
        let state_before = state.clone();
        profiler::reset();
        action(state);
        let hits = profiler::get_all_hits();
        let reduce_hits = hits
            .get("GLRParserState::reduce_and_goto")
            .copied()
            .unwrap_or(0);
        println!("  - {:<50}: reduce hits = {}", label, reduce_hits);
        if reduce_hits > 200 {
            state_before.log_gss(&format!("State BEFORE action '{}' that caused assertion failure", label), TerminalID(0), false, false);
            profiler::print_summary();
            panic!("Too many reductions ({}) for action '{}'.", reduce_hits, label);
        }
        reduce_hits
    }

    // 1. Load and compile the JavaScript grammar.
    println!("--- Setting up for JS GLR Parser Simplified Reduction Test ---");
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let parser = &compiled_grammar.glr_parser;
    println!("Grammar compiled successfully.");

    // 2. Define the single token to test.
    let test_terminal = regex_name("BIGINT_LITERAL");
    let test_terminal_id = *parser.terminal_map.get_by_left(&test_terminal).unwrap();

    // 3. Iterate through all 16 configurations of process_default_reductions calls.
    for i in 0..16 {
        let config_pdr_initial = (i & 1) != 0;
        let config_pdr_after_first_token = (i & 2) != 0;
        let config_pdr_after_merge = (i & 4) != 0;
        let config_pdr_after_second_token = (i & 8) != 0;

        println!("\n\n======================================================================");
        println!(
            "Testing config {}/16: initial={}, after_first={}, after_merge={}, after_second={}",
            i + 1,
            config_pdr_initial,
            config_pdr_after_first_token,
            config_pdr_after_merge,
            config_pdr_after_second_token
        );
        println!("======================================================================\n");

        // --- Step 1: Initial state ---
        let mut initial_state = parser.init_glr_parser(None);
        if config_pdr_initial {
            measure("Processed default reductions (initial)", &mut initial_state, |s| {
                s.process_default_reductions()
            });
        }

        // --- Step 2: Feed the first token ---
        println!("\n--- Phase 1: Feeding first token: '{}' ---", test_terminal);
        let mut state_after_first_token = initial_state.clone();
        state_after_first_token.step(test_terminal_id);

        if config_pdr_after_first_token {
            measure("Processed default reductions (after first token)", &mut state_after_first_token, |s| {
                s.process_default_reductions()
            });
        }
        assert!(state_after_first_token.is_ok(), "State is not OK after first token for config {}", i);

        // --- Step 3: Merge the initial state with the state after the first token ---
        println!("\n--- Phase 2: Merging initial state with the post-token state ---");
        let mut merged_state = initial_state.clone();
        merged_state.merge_with(state_after_first_token);
        merged_state.log_gss("Merged states", TerminalID(0), false, false);
        println!("States merged successfully.");

        if config_pdr_after_merge {
            measure("Processed default reductions (after merge)", &mut merged_state, |s| {
                s.process_default_reductions()
            });
        }

        // --- Step 4: Feed the second token to the merged state ---
        println!("\n--- Phase 3: Feeding second token to merged state ---");
        println!("\n--- Testing second token: '{}' ---", test_terminal);
        let mut state_for_second_token = merged_state.clone();

        state_for_second_token.step(test_terminal_id);
        // assert!(
        //     state_for_second_token.is_ok(),
        //     "Merged state became invalid after second token '{}' for config {}.",
        //     test_terminal,
        //     i
        // );

        if config_pdr_after_second_token {
            measure("Processed default reductions (after second token)", &mut state_for_second_token, |s| {
                s.process_default_reductions()
            });
        }
    }

    println!("\nTest passed: All configurations ran without assertion failures.");
    Ok(())
}

#[test]
fn test_js_parser_reduction_explosion_minimal_repro() -> Result<(), Box<dyn std::error::Error>> {
    // This test is a minimal reproduction of a reduction explosion observed in
    // `test_js_parser_reduction_explosion_simplified`. The issue occurs when
    // processing default reductions after feeding a single ambiguous token like
    // `BIGINT_LITERAL`. The original DFS-like processing order (`pop_last`) of
    // the worklist in `process_default_reductions` could lead to a large number
    // of reductions. This test verifies that the behavior is controlled.

    // Helper to measure reduction hits
    fn measure<'a, F: FnOnce(&mut GLRParserState<'a>)>(label: &str, state: &mut GLRParserState<'a>, action: F) -> u64 {
        profiler::reset();
        action(state);
        let hits = profiler::get_all_hits();
        let reduce_hits = hits
            .get("GLRParserState::reduce_and_goto")
            .copied()
            .unwrap_or(0);
        println!("  - {:<50}: reduce hits = {}", label, reduce_hits);
        if reduce_hits > 200 {
            state.log_gss(&format!("State BEFORE action '{}' that caused assertion failure", label), TerminalID(0), false, false);
            profiler::print_summary();
            panic!("Too many reductions ({}) for action '{}'.", reduce_hits, label);
        }
        reduce_hits
    }

    // 1. Load and compile the JavaScript grammar.
    println!("--- Setting up for JS GLR Parser Minimal Reduction Test ---");
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    let parser = &compiled_grammar.glr_parser;
    println!("Grammar compiled successfully.");

    // 2. Define the single token to test.
    let test_terminal = regex_name("BIGINT_LITERAL");
    let test_terminal_id = *parser.terminal_map.get_by_left(&test_terminal).unwrap();

    // --- Step 1: Initial state and feed token ---
    let mut state = parser.init_glr_parser(None);
    println!("\n--- Feeding token: '{}' ---", test_terminal);
    state.step(test_terminal_id);
    assert!(state.is_ok(), "State is not OK after token");

    // --- Step 2: Process default reductions ---
    println!("\n--- Processing default reductions ---");
    measure("Processed default reductions", &mut state, |s| {
        s.process_default_reductions()
    });

    println!("\nTest passed: Did not panic on excessive reductions.");
    Ok(())
}

// --- Main Integration Tests ---

#[test]
fn test_js_constraint_integration() -> Result<(), Box<dyn std::error::Error>> {
    // --- Setup Phase ---
    println!("--- Setting up for JS Constraint Integration Test ---");

    // 1. Load and compile the JavaScript grammar.
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
    let mut gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;

    if true { // Manual vocabulary modifications for debugging
        println!("\n--- Applying manual vocabulary modifications ---");

        // Filter 1: Keep only tokens with length <= x
        let x = 2;
        gpt2_raw_vocab.retain(|s| {
            let processed = s.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
            processed.as_bytes().len() <= x
        });
        println!("  - After length filter (<= {x}): {} tokens remaining.", gpt2_raw_vocab.len());

        gpt2_raw_vocab.push("-----".to_string()); // Add a specific token for testing

        // Filter 2: Keep only tokens where all alphabetic chars are 'a'
        gpt2_raw_vocab.retain(|s| {
            s.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n").as_bytes().iter().all(|&b| {
                if b.is_ascii_alphabetic() {
                    b.to_ascii_lowercase() == b'a'
                } else {
                    // true
                    // b"- &=a*;#[(:".contains(&b)
                    !b" &=a*;#[(:{}()[]?|<>+%~!.`,".contains(&b)
                }
            })
        });
        println!("  - After 'a'-only alphabetic filter: {} tokens remaining.", gpt2_raw_vocab.len());

        // Filter 3: Keep only tokens that contain at least one '-'
        // gpt2_raw_vocab.retain(|s| { s.len() == 1 || s.contains('-') });

        // Option 3: Set to a few specific tokens
        // gpt2_raw_vocab = vec![
        //     "[x]:",
        //     " &",
        // ].iter().map(|s| s.to_string()).collect();


        println!("  - Set to a specific small set of tokens: {} tokens.", gpt2_raw_vocab.len());

        println!("--- Finished manual vocabulary modifications ---\n");
    }

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;
    for (i, token_str) in gpt2_raw_vocab.iter().enumerate() {
        let id_val = i;
        let processed_token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        let token_bytes = processed_token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(id_val));
        max_original_llm_token_id_val = max_original_llm_token_id_val.max(id_val);
    }
    println!("GPT-2 vocab loaded ({} tokens, max_original_id: {}).", llm_token_map.len(), max_original_llm_token_id_val);

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

    // Log the source code in a structured way so the analysis script can be self-contained.
    println!("---BEGIN-SOURCE-JSON---");
    // Use serde_json to ensure proper escaping.
    println!("{}", serde_json::to_string(&full_text_to_tokenize)?);
    println!("---END-SOURCE-JSON---");
    
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
        if false {
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
        println!("  Current mask: {:?}", current_mask);

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
fn test_js_constraint_isolated_and_minimized() -> Result<(), Box<dyn std::error::Error>> {
    // This test serves as a tool for debugging the GrammarConstraint. It does the following:
    // 1. Defines a test case (an input string and a small vocabulary).
    // 2. Loads the full JS grammar.
    // 3. Manually defines a set of interesting terminals to focus the minimization.
    // 4. Minimizes the grammar to only what's necessary for those terminals.
    // 5. Compiles the minimized grammar and creates a GrammarConstraint.
    // 6. Feeds the input string to the new constraint to check for correctness.
    // 7. Saves the minimized grammar to an EBNF file for use in template tests.
    // 1. Define the test case.
    // let input_string = include_str!("../src/example_code.js");
    let input_string = "[x]:";
    let llm_token_strs = &["[x]:", " &"];
    let mut llm_token_map = LLMTokenMap::new();
    for (i, &token_str) in llm_token_strs.iter().enumerate() {
        llm_token_map.insert(token_str.as_bytes().to_vec(), LLMTokenID(i));
    }
    let max_token_id = if llm_token_strs.is_empty() { 0 } else { llm_token_strs.len() - 1 };

    // 2. Load the full JS grammar.
    println!("--- Loading and Minimizing JS Grammar for Constraint Test ---");
    let grammar_path = "src/js.ebnf";
    let mut full_grammar_def = GrammarDefinition::from_ebnf_file(grammar_path)?;
    println!("Initial production count: {}", full_grammar_def.productions.len());

    // 3. Manually define the interesting terminals for minimization.
    let interesting_terminals: BTreeSet<Terminal> = vec![
        regex_name("IGNORE"),
        regex_name("IDENTIFIER"),
        // literal(b"x"),
        literal(b"["),
        literal(b":"),
        literal(b"]"),
    ]
    .into_iter()
    .collect();
    println!("Manually specified interesting terminals: {:?}", interesting_terminals);

    // 4. Minimize the grammar.
    // let (minimized_productions, minimized_start_id) = crate::glr::minimizer::simplify_grammar_for_test_case(
    //     &full_grammar_def.productions,
    //     full_grammar_def.start_production_id,
    //     &interesting_terminals,
    // );
    let (minimized_productions, minimized_start_id) = (full_grammar_def.productions.clone(), full_grammar_def.start_production_id);

    // 5. Create and compile the minimized grammar definition.
    let mut minimized_def = full_grammar_def; // Start with a clone
    minimized_def.productions = minimized_productions;
    minimized_def.start_production_id = minimized_start_id;

    println!("\n--- Minimized Grammar ---");
    println!("{}", minimized_def);

    // 7. Save the minimized grammar to an EBNF file.
    let minimized_ebnf = minimized_def.to_ebnf();
    println!("\n--- Minimized Grammar (EBNF) ---");
    println!("{}", minimized_ebnf);
    fs::write("minimized_js_for_constraint.ebnf", minimized_ebnf)?;

    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(minimized_def));
    println!("\n--- Compiled Minimized Grammar ---");
    println!("{}", compiled_grammar);

    // 6. Create GrammarConstraint and run the test.
    let constraint = GrammarConstraint::from_compiled_grammar(
        compiled_grammar,
        llm_token_map.clone(),
        LLMTokenID(max_token_id + 1),
        max_token_id,
    );

    println!("\n--- Created GrammarConstraint ---");
    constraint.dump_precomputed();

    println!("\n--- Running Constraint with Minimized Grammar (Step-by-Step) ---");

    println!("---BEGIN-SOURCE-JSON---");
    println!("{}", serde_json::to_string(&input_string.to_string())?);
    println!("---END-SOURCE-JSON---");
    // Tokenize the input string
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    let mut test_token_sequence_ids = Vec::new();
    let mut tokenized_strs_for_logging = Vec::new();
    let mut text_to_process = input_string.as_bytes();

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

    // Step through the tokenized sequence
    let mut state = constraint.init();
    let all_code_lines: Vec<&str> = input_string.lines().collect();
    let mut current_text_byte_offset = 0;

    for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
        let current_token_str = &tokenized_strs_for_logging[i];
        println!("Processing token {}/{}: {:?} (LLMTokenID({}))", i + 1, test_token_sequence_ids.len(), current_token_str, llm_token_id.0);

        let token_start_byte = current_text_byte_offset;
        let token_end_byte = token_start_byte + current_token_str.as_bytes().len();
        print_token_context(&input_string, &all_code_lines, token_start_byte, token_end_byte, 2);

        assert!(state.is_active(), "State became inactive before token {}", i + 1);

        let mask_start = Instant::now();
        let current_mask = state.get_mask();
        println!("  get_mask took: {:?}", mask_start.elapsed());
        println!("  Current mask: {:?}", current_mask);

        assert!(current_mask.contains(llm_token_id.0), "Token {:?} (ID {}) not in mask at step {}", current_token_str, llm_token_id.0, i + 1);
        println!("  Token is in the mask.");

        let commit_start = Instant::now();
        state.commit(llm_token_id);
        println!("  commit took: {:?}", commit_start.elapsed());

        current_text_byte_offset = token_end_byte;
    }
    assert!(state.is_active(), "Constraint state became inactive after committing '{}'", input_string);
    println!("Successfully processed '{}' with minimized grammar constraint.", input_string);

    Ok(())
}

#[test]
#[ignore] // This is a template for creating new tests from a minimized grammar.
fn test_template_from_minimized_ebnf_for_constraint() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the minimized EBNF grammar.
    //    To use:
    //    a) Run `test_js_constraint_isolated_and_minimized` to generate `minimized_js_for_constraint.ebnf`.
    //    b) Copy this test function and rename it.
    //    c) Update `input_string` and `llm_token_map` with the sequence you want to test.
    //    d) Remove the `#[ignore]` attribute and run the test.
    // let ebnf_source = include_str!("../minimized_js_for_constraint.ebnf");
    let ebnf_source = &fs::read_to_string("minimized_js_for_constraint.ebnf")?;
    let grammar_def = GrammarDefinition::from_ebnf(ebnf_source)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));

    println!("\n--- Loaded Grammar from EBNF ---");
    println!("{}", compiled_grammar);

    // 2. Define the test case (must match the one in the minimizer test).
    // let input_string = include_str!("../src/example_code.js");
    let input_string = "[x]:";
    let llm_token_strs = &["[x]:", " &"];
    let mut llm_token_map = LLMTokenMap::new();
    for (i, &token_str) in llm_token_strs.iter().enumerate() {
        llm_token_map.insert(token_str.as_bytes().to_vec(), LLMTokenID(i));
    }
    let max_token_id = if llm_token_strs.is_empty() { 0 } else { llm_token_strs.len() - 1 };

    // 3. Create GrammarConstraint and run the test.
    let constraint = GrammarConstraint::from_compiled_grammar(
        compiled_grammar,
        llm_token_map.clone(),
        LLMTokenID(max_token_id + 1),
        max_token_id,
    );

    println!("\n--- Created GrammarConstraint ---");
    constraint.dump_precomputed();

    println!("\n--- Running Constraint with Loaded Grammar (Step-by-Step) ---");

    println!("---BEGIN-SOURCE-JSON---");
    println!("{}", serde_json::to_string(&input_string.to_string())?);
    println!("---END-SOURCE-JSON---");
    // Tokenize the input string
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    let mut test_token_sequence_ids = Vec::new();
    let mut tokenized_strs_for_logging = Vec::new();
    let mut text_to_process = input_string.as_bytes();

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

    // Step through the tokenized sequence
    let mut state = constraint.init();
    let all_code_lines: Vec<&str> = input_string.lines().collect();
    let mut current_text_byte_offset = 0;

    for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
        let current_token_str = &tokenized_strs_for_logging[i];
        println!("Processing token {}/{}: {:?} (LLMTokenID({}))", i + 1, test_token_sequence_ids.len(), current_token_str, llm_token_id.0);

        let token_start_byte = current_text_byte_offset;
        let token_end_byte = token_start_byte + current_token_str.as_bytes().len();
        print_token_context(&input_string, &all_code_lines, token_start_byte, token_end_byte, 2);

        assert!(state.is_active(), "State became inactive before token {}", i + 1);

        let mask_start = Instant::now();
        let current_mask = state.get_mask();
        println!("  get_mask took: {:?}", mask_start.elapsed());
        println!("  Current mask: {:?}", current_mask);

        assert!(current_mask.contains(llm_token_id.0), "Token {:?} (ID {}) not in mask at step {}", current_token_str, llm_token_id.0, i + 1);
        println!("  Token is in the mask.");

        let commit_start = Instant::now();
        state.commit(llm_token_id);
        println!("  commit took: {:?}", commit_start.elapsed());

        current_text_byte_offset = token_end_byte;
    }
    assert!(state.is_active(), "Constraint state became inactive after committing '{}'", input_string);
    println!("Successfully processed '{}' with loaded grammar constraint.", input_string);

    Ok(())
}

#[test]
fn test_template_from_minimized_ebnf_for_constraint_28_jul_2025() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the minimized EBNF grammar.
    //    To use:
    //    a) Run `test_js_constraint_isolated_and_minimized` to generate `minimized_js_for_constraint.ebnf`.
    //    b) Copy this test function and rename it.
    //    c) Update `input_string` and `llm_token_map` with the sequence you want to test.
    //    d) Remove the `#[ignore]` attribute and run the test.
    let ebnf_source = "start ::= '[' 'x' ']' ':' ;\nIGNORE ::= ' ';\n";
    let grammar_def = GrammarDefinition::from_ebnf(ebnf_source)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));

    println!("\n--- Loaded Grammar from EBNF ---");
    println!("{}", compiled_grammar);

    // 2. Define the test case (must match the one in the minimizer test).
    // let input_string = include_str!("../src/example_code.js");
    let input_string = "[x]:";
    let llm_token_strs = &["[x]:", " &"];
    let mut llm_token_map = LLMTokenMap::new();
    for (i, &token_str) in llm_token_strs.iter().enumerate() {
        llm_token_map.insert(token_str.as_bytes().to_vec(), LLMTokenID(i));
    }
    let max_token_id = if llm_token_strs.is_empty() { 0 } else { llm_token_strs.len() - 1 };

    // 3. Create GrammarConstraint and run the test.
    let constraint = GrammarConstraint::from_compiled_grammar(
        compiled_grammar,
        llm_token_map.clone(),
        LLMTokenID(max_token_id + 1),
        max_token_id,
    );

    println!("\n--- Created GrammarConstraint ---");
    constraint.dump_precomputed();

    println!("\n--- Running Constraint with Loaded Grammar (Step-by-Step) ---");

    println!("---BEGIN-SOURCE-JSON---");
    println!("{}", serde_json::to_string(&input_string.to_string())?);
    println!("---END-SOURCE-JSON---");
    // Tokenize the input string
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    let mut test_token_sequence_ids = Vec::new();
    let mut tokenized_strs_for_logging = Vec::new();
    let mut text_to_process = input_string.as_bytes();

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

    // Step through the tokenized sequence
    let mut state = constraint.init();
    let all_code_lines: Vec<&str> = input_string.lines().collect();
    let mut current_text_byte_offset = 0;

    for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
        let current_token_str = &tokenized_strs_for_logging[i];
        println!("Processing token {}/{}: {:?} (LLMTokenID({}))", i + 1, test_token_sequence_ids.len(), current_token_str, llm_token_id.0);

        let token_start_byte = current_text_byte_offset;
        let token_end_byte = token_start_byte + current_token_str.as_bytes().len();
        print_token_context(&input_string, &all_code_lines, token_start_byte, token_end_byte, 2);

        assert!(state.is_active(), "State became inactive before token {}", i + 1);

        let mask_start = Instant::now();
        let current_mask = state.get_mask();
        println!("  get_mask took: {:?}", mask_start.elapsed());
        println!("  Current mask: {:?}", current_mask);

        assert!(current_mask.contains(llm_token_id.0), "Token {:?} (ID {}) not in mask at step {}", current_token_str, llm_token_id.0, i + 1);
        println!("  Token is in the mask.");

        let commit_start = Instant::now();
        state.commit(llm_token_id);
        println!("  commit took: {:?}", commit_start.elapsed());

        current_text_byte_offset = token_end_byte;
    }
    assert!(state.is_active(), "Constraint state became inactive after committing '{}'", input_string);
    println!("Successfully processed '{}' with loaded grammar constraint.", input_string);

    Ok(())
}

#[test]
fn test_js_parser_isolated_and_minimized() -> Result<(), Box<dyn std::error::Error>> {
    // This test serves as a tool for debugging the parser. It does the following:
    // 1. Defines a sequence of terminals for a specific JS input string.
    // 2. Loads the full JS grammar.
    // 3. Minimizes the grammar to only what's necessary for that input string.
    // 4. Compiles the minimized grammar.
    // 5. Feeds the terminal sequence to the new parser to check for correctness.

    // 1. Define the terminal sequence for `var a = {[x]: 1};`
    let test_case_terminals: Vec<Terminal> = vec![
        literal(b"["),
        regex_name("IDENTIFIER"),
        literal(b"]"),
        literal(b":"),
    ];
    // Add any other terminals you want to ensure are included in the minimized grammar.
    let additional_interesting_terminals: Vec<Terminal> = vec![
        // For example, if you want to debug interactions with another token.
        // literal(b"some_other_token"),
    ];

    let mut interesting_terminals: BTreeSet<Terminal> = test_case_terminals.iter().cloned().collect();
    interesting_terminals.extend(additional_interesting_terminals.into_iter());

    // 2. Load the full JS grammar.
    println!("--- Loading and Minimizing JS Grammar ---");
    let grammar_path = "src/js.ebnf";
    let full_grammar_def = GrammarDefinition::from_ebnf_file(grammar_path)?;
    println!("Initial production count: {}", full_grammar_def.productions.len());

    // 3. Minimize the grammar.
    // let (minimized_productions, minimized_start_id) = crate::glr::minimizer::simplify_grammar_for_test_case(
    //     &full_grammar_def.productions,
    //     full_grammar_def.start_production_id,
    //     &interesting_terminals,
    // );
    let (minimized_productions, minimized_start_id) = (full_grammar_def.productions.clone(), full_grammar_def.start_production_id);

    // 5. Create and compile the minimized grammar.
    let minimized_def = GrammarDefinition {
        productions: minimized_productions,
        start_production_id: minimized_start_id,
        literal_to_group_id: full_grammar_def.literal_to_group_id.clone(),
        regex_name_to_group_id: full_grammar_def.regex_name_to_group_id.clone(),
        regex_expr_to_group_id: full_grammar_def.regex_expr_to_group_id.clone(),
        ignore_terminal_id: full_grammar_def.ignore_terminal_id,
    };
    println!("\n--- Minimized Grammar ---");
    println!("{}", minimized_def);

    // DUMP TO EBNF
    let minimized_ebnf = minimized_def.to_ebnf();
    println!("\n--- Minimized Grammar (EBNF) ---");
    println!("{}", minimized_ebnf);
    fs::write("minimized_js.ebnf", minimized_ebnf)?;

    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(minimized_def));

    let parser = &compiled_grammar.glr_parser;
    println!("\n--- Parser ----");
    println!("{}", parser);

    // 6. Convert test terminals to IDs using the new parser's map.
    let terminal_ids: Vec<TerminalID> = test_case_terminals
        .iter()
        .map(|t| *parser.terminal_map.get_by_left(t).unwrap())
        .collect();

    // 7. Run the parser and assert success.
    println!("\n--- Parsing with Minimized Grammar ---");
    let mut glr_state = parser.init_glr_parser(None);
    glr_state.parse(&terminal_ids);
    assert!(glr_state.is_ok(), "Parse failed with minimized grammar.");
    println!("Successfully parsed sequence with minimized grammar.");

    Ok(())
}

#[test]
#[ignore] // This is a template for creating new tests from a minimized grammar.
fn test_template_from_minimized_ebnf() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load the minimized EBNF grammar.
    //    To use:
    //    a) Run `test_js_parser_isolated_and_minimized` to generate `minimized_js.ebnf`.
    //    b) Copy this test function and rename it.
    //    c) Use your IDE's "inline macro" feature on `include_str!` below.
    //    d) Update `test_case_terminals` with the sequence you want to test.
    //    e) Remove the `#[ignore]` attribute and run the test.
    // let ebnf_source = include_str!("../minimized_js.ebnf"); // temporarily comment out to avoid compilation errors.
    let ebnf_source = &fs::read_to_string("minimized_js.ebnf")?;
    let grammar_def = GrammarDefinition::from_ebnf(ebnf_source)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));
    let parser = &compiled_grammar.glr_parser;

    println!("\n--- Loaded Grammar from EBNF ---");
    println!("{}", compiled_grammar);

    // 2. Define the sequence of terminals to test.
    //    (Update this with the specific sequence for your new test case)
    let test_case_terminals: Vec<Terminal> = vec![
        // Example:
        // literal(b"["),
        // regex_name("IDENTIFIER"),
        // literal(b"]"),
        // literal(b":"),
    ];

    // 3. Convert terminal objects to TerminalIDs using the parser's map.
    let terminal_ids: Vec<TerminalID> = test_case_terminals
        .iter()
        .map(|t| *parser.terminal_map.get_by_left(t).unwrap_or_else(|| panic!("Terminal '{}' not found in the compiled grammar's terminal map.", t)))
        .collect();

    // 4. Run the parser and assert success.
    println!("\n--- Parsing with Loaded Grammar ---");
    let mut glr_state = parser.init_glr_parser(None);
    glr_state.parse(&terminal_ids);

    assert!(glr_state.is_ok(), "Parse failed with the loaded EBNF grammar.");
    println!("Successfully parsed sequence with loaded grammar.");

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
            if let Some(terminal_id) = compiled_grammar.glr_parser.terminal_map.get_by_left(&regex_name(token_name)) {
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
        regex_name("IDENTIFIER"),
        literal(b"="),
        regex_name("NUMERIC_LITERAL"),
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
        glr_state.process_default_reductions();
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
