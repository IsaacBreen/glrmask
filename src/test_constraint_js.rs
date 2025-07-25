use crate::glr::parser::ParseState;
use rand::rngs::StdRng;
use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::{eat_u8, Match};
use crate::{choice, choice_fast, groups, seq, seq_fast};
use crate::glr::grammar::{nt, prod, t, terminal, NonTerminal, Production, Symbol, Terminal};
use crate::glr::table::{assign_non_terminal_ids, assign_terminal_ids, generate_glr_parser, generate_glr_parser_with_maps, generate_glr_parser_with_terminal_map, StateID};
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
use similar::TextDiff;
use crate::constraint::{GrammarConstraint};
use crate::datastructures::trie::Trie;
use crate::json_serialization::{JSONConvertible, JSONNode};
// Already a main dependency, but good to be explicit if used directly
// reqwest will be used if the file isn't cached, ensure it's in dev-dependencies
use crate::tokenizer::{LLMTokenID, LLMTokenMap, Token, TokenizerStateID};
use crate::types::TerminalID;
use crate::datastructures::vocab_prefix_tree::VocabPrefixTree; // Added for tokenization
use std::time::Instant;
use rand::prelude::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use crate::glr::analyze::{filter_productions_by_reachability, remove_productions_with_undefined_nonterminals};
use std::panic::{self, AssertUnwindSafe}; // Added for panic catching
use std::collections::HashMap;
use serde::__private::ser::constrain;
use crate::datastructures::gss::{gather_gss_stats, GSSNode, reset_llm_tokens, sample_path};
use crate::glr::stats::get_stats;
// For the symbol removal helper


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

#[ignore]
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
    let mut token_name_map: BiBTreeMap<Terminal, usize> = BiBTreeMap::new();
    token_name_map.insert(terminal("FSTRING_MIDDLE"), 0);
    token_name_map.insert(terminal("DEF"), 1); // "def" token

    // 4. Call precompute
    println!(
        "Starting precompute with GPT-2 vocab ({} tokens, max_original_id_val: {})...",
        llm_token_map.len(),
        max_llm_token_id_val, // Max original ID value encountered
    );

    // 2. Create a parser
    let productions = vec![
        prod("S'", vec![nt("S")]), // Start
        prod("S", vec![nt("A"), t("DEF")]),
        // prod("S", vec![t("FSTRING_MIDDLE"), t("FSTRING_MIDDLE")]),
        // prod("S", vec![t("FSTRING_MIDDLE")]),
        // prod("S", vec![t("FSTRING_MIDDLE"), t("DEF")]),
        prod("A", vec![]),
    ];
    let terminal_map: BiBTreeMap<Terminal, TerminalID> = token_name_map.iter().map(|(name, id)| (name.clone(), TerminalID(*id))).collect();
    let parser = generate_glr_parser(&productions, 0, None);

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
    let mask = constraint_state.get_mask();

    let d_id = llm_token_map.get_by_left(&b"def"[..]).unwrap().0;
    assert!(mask.contains(d_id), "Expected LLM token ID {} to be in mask", d_id);

    // Step and commit the LLM token "a" repeatedly.
    println!("{}", constraint.parser);
    let mut constraint_state = constraint.init();
    let a_id = llm_token_map.get_by_left(&b"a"[..]).unwrap().0;
    for i in 0..10 {
        println!("{}. Stepping with LLM token ID {}", i, a_id);
        let mask = constraint_state.get_mask();
        constraint_state.commit(LLMTokenID(a_id));
        assert!(constraint_state.is_active(), "Constraint state should be active after committing token {} (ID {})", a_id, a_id);
    }

    Ok(())
}

/// Helper function to print context around a token in a larger text.
fn print_token_context(
    full_text: &str, // Used for end-of-text checks
    all_lines: &[&str], // Pre-split lines of the full_text
    token_start_global_byte: usize,
    token_end_global_byte: usize, // Exclusive end
    context_lines_count: usize,   // Number of lines before and after
) {
    if all_lines.is_empty() {
        // Should ideally not happen if full_text led to tokenization,
        // but handles empty full_text case.
        // Note: `"".lines().collect::<Vec<_>>()` is `[""]`.
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
            // Do not break, continue to find end line in the same pass if possible, or for next pass's cbo
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

        // Token ends on this line if its end byte falls within this line's content
        if token_end_global_byte > line_start_byte_offset && token_end_global_byte <= line_content_end_byte_offset {
            token_end_line_idx = idx;
            token_end_col_byte_in_line = token_end_global_byte - line_start_byte_offset;
            break;
        }
        // Token ends exactly at the newline character after this line's content
        if idx < all_lines.len() - 1 && token_end_global_byte == line_content_end_byte_offset + 1 {
            token_end_line_idx = idx;
            token_end_col_byte_in_line = line_content.len(); // Covers the entire content part of the line
            break;
        }

        current_scan_byte_offset += line_content.len();
        if idx < all_lines.len() - 1 {
            current_scan_byte_offset += 1; // Account for '\n'
        }
    }
     // If token goes to the very end of the file and file doesn't end with newline
    if token_end_global_byte == full_text.len() && !full_text.ends_with('\n') && !full_text.is_empty() {
        token_end_line_idx = all_lines.len() - 1;
        token_end_col_byte_in_line = all_lines.last().map_or(0, |s| s.len());
    }


    let display_start_line = token_start_line_idx.saturating_sub(context_lines_count);
    let display_end_line = (token_end_line_idx + context_lines_count).min(all_lines.len().saturating_sub(1));

    println!("    Context Highlight (Token bytes [{}, {})):", token_start_global_byte, token_end_global_byte);
    for i in display_start_line..=display_end_line {
        let line_content = all_lines[i];
        println!("{:5} | {}", i + 1, line_content); // 1-indexed line numbers

        if i >= token_start_line_idx && i <= token_end_line_idx {
            let start_col = if i == token_start_line_idx { token_start_col_byte_in_line } else { 0 };
            let end_col = if i == token_end_line_idx { token_end_col_byte_in_line } else { line_content.len() };
            
            let effective_start_col = start_col.min(line_content.len());
            let effective_end_col = end_col.min(line_content.len()).max(effective_start_col);

            if effective_end_col > effective_start_col {
                let prefix = " ".repeat(effective_start_col);
                let carets = "^".repeat(effective_end_col - effective_start_col);
                println!("      | {}{}", prefix, carets);
            } else if token_start_global_byte == token_end_global_byte && i == token_start_line_idx && effective_start_col <= line_content.len() { // Empty token
                let prefix = " ".repeat(effective_start_col);
                println!("      | {}{}", prefix, "^");
            }
        }
    }
    println!("    ------------------------------------------");
}

#[test]
fn test_js_constraint_with_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)
        .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    println!("Compiling GrammarDefinition into CompiledGrammar...");
    // The definition is cloned here because it's moved into CompiledGrammar::from_definition
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));

    let stats = get_stats(&compiled_grammar.glr_parser);
    println!("Parser stats: {}", stats);

    println!("Successfully compiled GrammarDefinition into CompiledGrammar.");
    println!("{}", compiled_grammar);
    // --- New test section for grammar terminal sequences ---
    println!("\nTesting GLR parser with specific grammar terminal sequences...");

    if false {
        // This block can be used for specific tokenizer checks if needed.
    }

    // Define the sequences of terminal names to test
    let mut test_sequences_str = vec![
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
        vec!["STRING_LITERAL", "IDENTIFIER"], // A string followed by an identifier
        vec!["NUMERIC_LITERAL", "\"+\"", "NUMERIC_LITERAL"],
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
            if let Some(terminal_id_val) = compiled_grammar.glr_parser.terminal_map.get_by_left(&terminal(token_name)) {
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
        let mut glr_state = compiled_grammar.glr_parser.init_glr_parser(None);

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
    // let num_fuzz_iterations = 1000;
    let num_fuzz_iterations = 0;
    let max_tokens_per_fuzz_attempt = 50;

    let all_grammar_terminal_ids: Vec<_> = compiled_grammar.glr_parser.terminal_map.right_values().cloned().collect();

    if all_grammar_terminal_ids.is_empty() {
        println!("  Warning: No grammar terminal IDs found in compiled_grammar.glr_parser.terminal_map. Fuzz test will be trivial or skipped.");
    } else {
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..num_fuzz_iterations {
            if i % 100 == 0 { // Log progress
                println!("  Fuzz test iteration {}/{}", i, num_fuzz_iterations);
            }
            let mut glr_state = compiled_grammar.glr_parser.init_glr_parser(None);

            let num_tokens_this_attempt = rng.gen_range(0..=max_tokens_per_fuzz_attempt);
            let mut current_fuzz_sequence_names: Vec<String> = Vec::new();
            let mut current_fuzz_sequence_ids: Vec<TerminalID> = Vec::new();

            for _ in 0..num_tokens_this_attempt {
                let random_terminal_id = all_grammar_terminal_ids.choose(&mut rng).unwrap();
                // For debugging, you could find the name:
                let token_name = compiled_grammar.glr_parser.terminal_map.get_by_right(random_terminal_id).map(|t| t.to_string()).unwrap_or_else(|| "UNKNOWN_TOKEN".to_string());
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
    let cache_dir = Path::new(".cache/test_vocabs");
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let vocab_file_name = "gpt2_vocab.json";
    // let vocab_url = "https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B/raw/main/vocab.json";
    // let vocab_file_name = "qwen_vocab.json";
    let mut gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;
    // let gpt2_raw_vocab = BTreeMap::from([("________________________________________________________________", 0)]);
    // // Just fill with all bytes
    // let mut gpt2_raw_vocab = BTreeMap::new();
    // for i in 0u8..=255u8 {
    //     let c = i as char;
    //     let s = c.to_string();
    //     gpt2_raw_vocab.insert(s, i as usize);
    // }
    // let gpt2_raw_vocab = BTreeMap::from([("from", 0), (" typing", 1)]);
    // gpt2_raw_vocab.insert("import os".to_string(), 0);
    // gpt2_raw_vocab.insert("import ".to_string(), 1);
    // gpt2_raw_vocab.insert(" os".to_string(), 2);
    // gpt2_raw_vocab.insert("import".to_string(), 1);
    // gpt2_raw_vocab.insert(" ".to_string(), 3);
    // gpt2_raw_vocab.insert("os".to_string(), 4);
    // gpt2_raw_vocab.insert("from".to_string(), 100);
    // gpt2_raw_vocab.insert(" typing".to_string(), 101);
    let N = 2000;
    // Add "#" and "*" * N
    // gpt2_raw_vocab.insert("#".to_string(), 0);
    // let mut asterisks = String::new();
    // for _ in 0..1000 {
    //     asterisks.push('*');
    // }
    // gpt2_raw_vocab.insert(asterisks, 1);
    // gpt2_raw_vocab.insert(" ".to_string(), 2);
    // gpt2_raw_vocab.insert("  ".to_string(), 3);
    // gpt2_raw_vocab.insert("    ".to_string(), 4);

    // // Filter the vocabulary. The goal is to keep tokens that are NOT of the form "Ġ" + alphanumeric,
    // // but if they ARE of that form, only keep them if they are "ĠCali" + alphanumeric.
    // let gpt2_raw_vocab: BTreeMap<String, u32> = gpt2_raw_vocab.into_iter().filter(|(token, _id)| {
    //     let is_alphanumeric_continuation =
    //         token.starts_with('Ġ') &&
    //         token.len() > 1 &&
    //         token.chars().skip(1).all(|c| c.is_ascii_alphanumeric());
    //
    //     if is_alphanumeric_continuation {
    //         // This is a token of the form " [a-zA-Z0-9]+". Only keep it if it matches " Cali[...]"
    //         token.starts_with("ĠCali")
    //     } else {
    //         // Keep all other tokens that don't match the initial pattern.
    //         true
    //     }
    // }).collect();

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;

    let prop = 1.0; // Use full vocab for this test to ensure token presence
    let total_tokens = gpt2_raw_vocab.len();

    for (token_str, id_val_u32) in gpt2_raw_vocab.into_iter(){
        let id_val = id_val_u32 as usize;
        // Replace 'Ġ' with ' '
        let token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        let token_bytes = token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(id_val));
        if id_val > max_original_llm_token_id_val {
            max_original_llm_token_id_val = id_val;
        }
    }

    // // Remove tokens longer than length
    // llm_token_map.retain(|v, _| v.len() <= 3);
    //
    // // Keep tokens that are either given length or of the form ' Ax' where x is a letter
    // // llm_token_map.retain(|v, _| v.len() <= 1 || v.starts_with(b" A") && v.len() <= 3
    // // v.len() <= 1 || (v.starts_with(b" A")) && v.len() <= 3
    // llm_token_map.retain(|v, _| v.len() <= 2);
    // // Remove tokens that contain non-space non-alph, non-upper-case characters
    // // llm_token_map.retain(|v, _| v.len() == 1 ||
    // //     v.starts_with(b" A") &&
    // //     v.iter().all(|c| c.is_ascii_alphabetic() || c.is_ascii_uppercase()));
    // // Remove tokens that contain capital letters
    // // llm_token_map.retain(|v, _| v.len() == 1 ||
    // //     v.iter().all(|c| !c.is_ascii_alphabetic() || c.is_ascii_lowercase()));
    // // Remove tokens that contain any of the first x of the capital letters in the alphabet
    // // llm_token_map.retain(|v, _| v.len() == 1 ||
    // //     v.iter().all(|c| !c.is_ascii_alphabetic() || c <= &b'c'));
    // // // Remove tokens that contain letters
    // // llm_token_map.retain(|v, _| v.len() == 1 ||
    // //     v.iter().all(|c| c.is_ascii_alphabetic()));
    //
    // // Remove tokens that contain more than two different digits
    // // llm_token_map.retain(|v, _| v.len() == 1 ||
    // //     v.iter().filter(|c| c.is_ascii_digit()).collect::<BTreeSet<_>>().len() <= 1);
    //
    // // Keep only "1" and "11"
    // // llm_token_map.retain(|v, _| v.len() == 1 || v == b"1" || v == b"11");
    // // llm_token_map.retain(|v, _| v == b"1" || v == b"11");

    // llm_token_map.retain(|v, _| [b"from".as_ref(), b" x".as_ref()].contains(&v.as_ref()) || v.len() == 2 && v.iter().all(|c| !c.is_ascii_alphanumeric()));
    // llm_token_map.retain(|v, _| [b"from".as_ref(), b" x".as_ref(), ].contains(&v.as_ref()) || v.len() == 2 && v.iter().all(|c| !c.is_ascii_alphanumeric()) && *String::from_utf8_lossy(v.as_ref()) <= *")=");
    // llm_token_map.retain(|v, _| [b"from".as_ref(), b" x".as_ref(), ].contains(&v.as_ref()) || v.len() == 2 && v.iter().all(|c| !c.is_ascii_alphanumeric()) && *" `" <= *String::from_utf8_lossy(v.as_ref()) && *String::from_utf8_lossy(v.as_ref()) <= *"\"[");
    // llm_token_map.retain(|v, _| [b"from".as_ref(), b" x", b" ("].contains(&v.as_ref()));
    // let to_keep: Vec<&[u8]> = vec![
    //     b")=",
    //     b" x",
    //     b"from",
    // ];
    // llm_token_map.retain(|v, _| to_keep.contains(&v.as_ref()));
    // assert!(llm_token_map.contains_left(&b"from".to_vec()));
    // assert!(llm_token_map.contains_left(&b" x".to_vec()));

    // fn vec_contains(vec: &[u8], other: &[u8]) -> bool {
    //     vec.windows(other.len()).any(|window| window == other)
    // }
    // llm_token_map.retain(|v, _| v.len() == 1 || ![b"-".as_ref(), b"*", b"...", b"_"].iter().any(|other| vec_contains(v.as_ref(), other)));

    // llm_token_map.retain(|v, _| v.len() <= 2);
    // llm_token_map.retain(|v, _| [b"x".as_ref(), b" =".as_ref()].contains(&v.as_ref()));
    // llm_token_map.retain(|v, _| [b"x".as_ref(), b"=".as_ref(), b" "].contains(&v.as_ref()));
    // llm_token_map.retain(|v, _| [b"'".as_ref()].contains(&v.as_ref()));
    // llm_token_map.retain(|v, _| v.len() == 1);

    llm_token_map.retain(|v, _| [b"x".as_ref(), b"[", b"]", b":"].contains(&v.as_ref()));
    llm_token_map.retain(|v, _| v.len() <= 2);
    // llm_token_map.retain(|v, _| v.len() <= 5);
    // llm_token_map.retain(|v, _| v.len() <= 2 && v.iter().all(|c| c.is_ascii_whitespace() || c == &b'a'));
    // llm_token_map.retain(|v, _| [b"a".as_ref(), b" a"].contains(&v.as_ref()));
    // llm_token_map.retain(|v, _| [b"a".as_ref(), b" a", b"aa"].contains(&v.as_ref()));

    // Print the vocab
    println!("GPT-2 vocab loaded and processed into LLMTokenMap ({} tokens, max_original_id: {}).", llm_token_map.len(), max_original_llm_token_id_val);
    for (token, id) in llm_token_map.iter() {
        println!("  {:?}: {} (ID {})", String::from_utf8_lossy(token), id.0, id.0);
    }

    if llm_token_map.is_empty() {
        println!("Warning: LLM token map is empty after sampling. Max original ID will be 0.");
    }
    println!("GPT-2 vocab loaded and processed into LLMTokenMap ({} tokens, max_original_id: {}).", llm_token_map.len(), max_original_llm_token_id_val);

    // 4. Construct GrammarConstraint
    let dummy_eof_placeholder = 0;
    println!("Constructing GrammarConstraint...");
    let grammar_constraint = GrammarConstraint::from_compiled_grammar(
        compiled_grammar.clone(),
        llm_token_map.clone(),
        LLMTokenID(dummy_eof_placeholder),
        max_original_llm_token_id_val
    );
    grammar_constraint.dump_precomputed();
    println!("GrammarConstraint constructed successfully.");
    println!("GrammarConstraint original to internal ID map:");
    let mut temp = grammar_constraint.llm_vocab.original_to_internal_id_bimap.iter().collect::<Vec<_>>();
    temp.sort_by_key(|(original_id, internal_id)| *internal_id);
    for (original_id, internal_id) in temp {
        let token = llm_token_map.get_by_right(&LLMTokenID(*original_id)).unwrap();
        println!("  original {}, internal {}, token {:?}, raw: {:?}", original_id, internal_id, String::from_utf8_lossy(token), token);
    }

    // Ensure there's an edge in the root precompute node for state 0 that has the terminal for `IGNORE[0][0][1]` on the edge key and which the LLM token for "\n" on the edge value.
    if false {
        // 1. Get the root precompute node for tokenizer state 0.
        let precompute_root_node = grammar_constraint.precomputed.get(&TokenizerStateID(0))
            .expect("Precomputed data for tokenizer state 0 should exist.").lock().unwrap();

        // 2. Get the TerminalID for the terminal we are interested in.
        let newline_terminal_name = "IGNORE[0][0][1]".to_string();
        let newline_terminal_id = grammar_constraint.parser.terminal_map
            .get_by_left(&terminal(&newline_terminal_name))
            .unwrap_or_else(|| panic!("Terminal '{}' not found in parser's terminal map.", newline_terminal_name));

        // 3. Get the LLMTokenID for the newline character.
        let newline_bytes = b"\n";
        let newline_llm_token_id = grammar_constraint.llm_vocab.llm_token_map
            .get_by_left(&newline_bytes.to_vec())
            .unwrap_or_else(|| panic!("LLM token for newline '{:?}' not found in token map.", String::from_utf8_lossy(newline_bytes)));
        let newline_llm_token_id = grammar_constraint.llm_vocab.original_to_internal_id_bimap.get_by_left(&newline_llm_token_id.0).unwrap();

        // 4. Check for the edge in the precompute root node.
        // The edge key is Option<TerminalID>.
        let edge_key = Some(*newline_terminal_id);

        let destinations_map = precompute_root_node.children().get(&edge_key)
            .unwrap_or_else(|| panic!("No edge for terminal '{}' (ID {}) found in precompute root for state 0.", newline_terminal_name, newline_terminal_id.0));

        // 5. Check if any edge for this key contains the newline LLM token.
        let found_edge_with_newline_token = destinations_map.values().any(|edge_value_bv| edge_value_bv.contains(*newline_llm_token_id));

        assert!(found_edge_with_newline_token, "Expected to find an edge for terminal '{}' (ID {}) containing the LLM token for newline (ID {}), but none was found. Got: {:?}", newline_terminal_name, newline_terminal_id.0, newline_llm_token_id, destinations_map);

        // Print the edge value
        println!("Edge value for terminal '{}' (ID {}) containing the LLM token for newline (ID {}): {:?}", newline_terminal_name, newline_terminal_id.0, newline_llm_token_id, destinations_map);

        println!("Successfully verified edge for '{}' with LLM token for '\\n'.", newline_terminal_name);
    }

    // // Ensure grammar constraint creation is deterministic
    // assert_eq!(grammar_constraint, GrammarConstraint::from_compiled_grammar(
    //     compiled_grammar,
    //     llm_token_map.clone(),
    //     dummy_eof_placeholder,
    //     max_original_llm_token_id_val
    // ));

    // grammar_constraint.dump_precomputed(); // Temporarily commented out due to potential verbosity

    // --- TOKENIZATION AND SEQUENCE TESTING ---

    // Build a VocabPrefixTree from the LLM token map for tokenization
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = grammar_constraint.llm_vocab.llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    // The full text to tokenize.
    let example_code_path = "src/example_code.js";
    let full_text_to_tokenize = match fs::read_to_string(example_code_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read example code file '{}': {}", example_code_path, e);
            return Err(Box::new(e)); // Or handle as appropriate for your test
        }
    };
    // let full_text_to_tokenize = "from typing import Any, List";
    // let full_text_to_tokenize = "import os# some comments\nfrom collections";
    // let full_text_to_tokenize = "a";
    // let full_text_to_tokenize = "((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((";
    // let full_text_to_tokenize = "a";
    // let mut full_text_to_tokenize = "#".to_string();
    // // Add * to it
    // for _ in 0..N {
    //     full_text_to_tokenize.push_str("*"); // Causes stack overflow
    //     // full_text_to_tokenize.push_str("+"); // Causes major slowdown
    // }
    // let full_text_to_tokenize = "import os\nimport sys";
    //     let full_text_to_tokenize = "# Top-level comment, challenging parser start\nimport os";
    // let full_text_to_tokenize = "from x";
    // f-strings
    // let full_text_to_tokenize = "x = f'hi!'\n";
    // let full_text_to_tokenize = "x = f'hi{x}'\n";
    // let full_text_to_tokenize = "f'hi{x}'\n";
    // let full_text_to_tokenize = "                        ";
    // let full_text_to_tokenize = "# Top-level comment, challenging parser start\nimport os, sys # Multiple imports on one line\nfrom collections import (defaultdict,\n                         deque) # Multi-line import with parens\n\nGLOBAL_VAR: int = 100";
    // let full_text_to_tokenize = "azazazazazazazazazazazazazazazazazazazazazazazazazazaz";
    // let full_text_to_tokenize = "x =";
    // let full_text_to_tokenize = "message = f\"\nProcessing {value} (type: {\n    'integer' if isinstance(value, int) else\n    ('float' if isinstance(value, float) else 'other')\n}) at index {i}\"\n";
    // let full_text_to_tokenize = "f\"{''}\"";
    // let full_text_to_tokenize = "f\"{'float'}\"";
    // let full_text_to_tokenize = "'float'";
    // let full_text_to_tokenize = "'";
    // let full_text_to_tokenize = "--------------------";
    // let full_text_to_tokenize = "{}";

    // let full_text_to_tokenize = "r\"\"\n{}";

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

    // Feed in the full text
    let mut constraint_state = grammar_constraint.init();
    // constraint_state.commit_bytes(full_text_to_tokenize.as_bytes());
    // assert!(
    //     constraint_state.is_active(),
    //     "Constraint state should be active after committing the full text."
    // );
    // return Ok(());

    if false {
        let mut constraint_state1 = grammar_constraint.init();
        let mut constraint_state2 = grammar_constraint.init();
        pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "Initial constraint states should be equal after initialization.");
        assert_eq!(constraint_state1.state, constraint_state2.state, "Initial constraint states should be equal after initialization.");
        for (i, byte) in full_text_to_tokenize.as_bytes().iter().enumerate() {
            println!("Committing byte {}: '{}'", i + 1, *byte as char);
            constraint_state1.commit_bytes(&[*byte]);
            constraint_state2.commit_bytes(&[*byte]);
            pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "Constraint states should remain equal after committing byte {}.", i + 1);
            assert_eq!(constraint_state1.state, constraint_state2.state, "Constraint states should remain equal after committing byte {}.", i + 1);
        }

        // let mut constraint_state1 = grammar_constraint.init();
        // let mut constraint_state2 = grammar_constraint.init();
        // pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "Initial constraint states should be equal after initialization.");
        // assert_eq!(constraint_state1.state, constraint_state2.state, "Initial constraint states should be equal after initialization.");
        // for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
        //     let current_token_str = &tokenized_strs_for_logging[i];
        //     println!("Committing token {}/{}: '{}' (LLMTokenID({}))", i + 1, test_token_sequence_ids.len(), current_token_str, llm_token_id.0);
        //     constraint_state1.commit(llm_token_id);
        //     constraint_state2.commit(llm_token_id);
        //     pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "Constraint states should remain equal after committing token {}.", i + 1);
        //     assert_eq!(constraint_state1.state, constraint_state2.state, "Constraint states should remain equal after committing token {}.", i + 1);
        // }

        // let mut constraint_state1 = grammar_constraint.init();
        // for (i, byte) in full_text_to_tokenize.as_bytes().iter().enumerate() {
        //     println!("Committing byte {}: '{}'", i + 1, *byte as char);
        //     constraint_state1.commit_bytes(&[*byte]);
        //     println!("Committing prefix up to byte {}", i + 1);
        //     let mut constraint_state2 = grammar_constraint.init();
        //     let prefix_bytes = full_text_to_tokenize.as_bytes()[..i + 1].to_vec();
        //     constraint_state2.commit_bytes(&prefix_bytes);
        //     pretty_assertions::assert_eq!(constraint_state1.to_string(), constraint_state2.to_string(), "Constraint states should remain equal after committing byte {}.", i + 1);
        //     assert_eq!(constraint_state1.state, constraint_state2.state, "Constraint states should remain equal after committing byte {}.", i + 1);
        // }
    }

    // 5. Basic Interaction with the GrammarConstraintState
    let mut constraint_state = grammar_constraint.init();
    // Initial step to populate possibilities
    let step_start = Instant::now();
    let initial_mask = constraint_state.get_mask();
    let step_duration = step_start.elapsed();
    println!("\nInitial get_mask took: {:?}", step_duration);
    println!("\nInitial mask obtained ({} allowed LLM tokens).", initial_mask.iter_bits().count());
    let all_code_lines: Vec<&str> = full_text_to_tokenize.lines().collect();
    let mut current_text_byte_offset = 0;
    // return Ok(());

    if true {
        println!("\nStepping through the token sequence with GrammarConstraint:");
        for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
            // Use tokenized_strs_for_logging for logging, as it corresponds to the llm_token_id
            let current_token_str = &tokenized_strs_for_logging[i];
            println!(
                "Processing token {}/{}: {:?} (LLMTokenID({}))",
                i + 1, // 1-indexed for display
                test_token_sequence_ids.len(),
                current_token_str,
                llm_token_id.0
            );

            if false {
                constraint_state = grammar_constraint.init();
                println!("  Re-initializing constraint state for token {} ('{}').", i + 1, current_token_str);
                // Commit the full text up to this token
                let prefix_bytes = full_text_to_tokenize.as_bytes()[..current_text_byte_offset].to_vec();
                constraint_state.commit_bytes(&prefix_bytes);
                println!("  Committed prefix bytes up to token {} ('{}').", i + 1, current_token_str);
            }

            // Display context
            let token_start_byte_in_full_text = current_text_byte_offset;
            let token_end_byte_in_full_text = current_text_byte_offset + current_token_str.as_bytes().len();
            print_token_context(
                &full_text_to_tokenize,
                &all_code_lines,
                token_start_byte_in_full_text,
                token_end_byte_in_full_text,
                2, // Show 2 lines before and 2 lines after
            );

            assert!(
                constraint_state.is_active(),
                "Constraint state should be active before processing token {} ('{}')",
                i + 1, current_token_str
            );

            let step_start = Instant::now();
            let current_mask = constraint_state.get_mask();
            let step_duration = step_start.elapsed();
            println!("  get_mask took: {:?}", step_duration);

            println!(
                "  Mask (after get_mask) allows {} tokens. Checking for current token LLMTokenID({})...",
                current_mask.iter_bits().count(),
                llm_token_id.0
            );

            assert!(
                current_mask.contains(llm_token_id.0),
                "Expected LLMTokenID({}) for {:?} to be in the mask. Mask: {:?}",
                llm_token_id.0, current_token_str, current_mask
            );
            println!("  LLMTokenID({}) for '{}' is in the mask.", llm_token_id.0, current_token_str);

            let commit_start = Instant::now();
            constraint_state.commit(llm_token_id);
            let commit_duration = commit_start.elapsed();
            println!("  commit LLMTokenID({}) took: {:?}", llm_token_id.0, commit_duration);
            println!("  Committed LLMTokenID({}) for '{:?}'.", llm_token_id.0, current_token_str);

            // if true {
            //     println!("  Checking constraint state integrity after commit:");
            //     let mut new_constraint_state = grammar_constraint.init();
            //     // Commit full bytes for prefix
            //     let prefix_bytes = full_text_to_tokenize.as_bytes()[..token_end_byte_in_full_text].to_vec();
            //     new_constraint_state.commit_bytes(&prefix_bytes);
            //     // use pretty_assertions::{assert_eq, assert_ne};
            //     // println!("new_constraint_state:\n{}", new_constraint_state);
            //     // println!("constraint_state:\n{}", constraint_state);
            //     pretty_assertions::assert_eq!(
            //         new_constraint_state.to_string(), constraint_state.to_string()
            //     );
            //     assert_eq!(
            //         new_constraint_state.state, constraint_state.state,
            //         "New constraint state after committing prefix bytes should match the original constraint state."
            //     );
            // }

            assert!(
                constraint_state.is_active(),
                "Constraint state should be active after committing token {} ('{}')",
                i + 1, current_token_str
            );
            println!("  Constraint state is active after commit.");

            // println!("    Sampling and explaining up to 10 parse stacks:");
            // let gss_roots: Vec<&GSSNode> = constraint_state.state.values()
            //     .map(|glr_state| glr_state.active_state.stack.as_ref())
            //     .collect();
            //
            // if gss_roots.is_empty() {
            //     println!("      No active GSS roots to sample from.");
            // } else {
            //     for sample_idx in 0..10 {
            //         // Create a unique seed for each sample
            //         let seed = (i as u64).wrapping_mul(100) + (sample_idx as u64);
            //         if let Some(sampled_path_edges) = sample_path(&gss_roots, seed) {
            //             let mut sampled_stack: Vec<StateID> = sampled_path_edges.iter()
            //                 .map(|edge| edge.state_id)
            //                 .collect();
            //             // The path is from leaf to root, so we reverse to get stack order (bottom to top)
            //             sampled_stack.reverse();
            //
            //             println!("    --- Sample {} (seed {}) ---", sample_idx + 1, seed);
            //             let explanation = grammar_constraint.parser.explain_stack(&sampled_stack);
            //             // Indent the explanation for readability
            //             for line in explanation.lines() {
            //                 // println!("      {}", line);
            //             }
            //         } else {
            //             println!("    Sample {}: No path found (GSS might be empty or just a root).", sample_idx + 1);
            //             // If one sample fails, others might too. Break to avoid redundant messages.
            //             break;
            //         }
            //     }
            // }

            // Update current_text_byte_offset for the next iteration
            current_text_byte_offset = token_end_byte_in_full_text;
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
    }

    if false {
        let b1 = b"if x: ";
        let b2 = b"print(x)";

        println!("\n--- Testing commit equivalence {:?} {:?} ---", String::from_utf8_lossy(b1), String::from_utf8_lossy(b2));

        // Scenario 1: Commit bytes separately
        let mut state1 = grammar_constraint.init();
        println!("Scenario 1: Committing b1: {:?}", String::from_utf8_lossy(b1));
        state1.commit_bytes(b1);
        println!("Scenario 1: Committing b2: {:?}", String::from_utf8_lossy(b2));
        state1.commit_bytes(b2);

        // Scenario 2: Commit concatenated bytes
        let mut state2 = grammar_constraint.init();
        let combined_bytes = [b1.as_ref(), b2.as_ref()].concat();
        println!("Scenario 2: Committing combined bytes: {:?}", String::from_utf8_lossy(&combined_bytes));
        state2.commit_bytes(&combined_bytes);

        assert_eq!(state1, state2, "States from separate and combined commits should be equivalent.");
        println!("--- Equivalence test passed ---");
    }

    if false {
        // Ensure the parse state after stepping the constraint with all LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
        // In general, this should be true if all LLM tokens cleanly match grammar tokens (or, equivalently, if the only non-empty entry in the precompute tree is under the initial tokenizer state).
        // let grammar_tokenss = vec![vec!["\"from\""], vec!["NAME[0]"]];
        // let llm_tokens_for_comp: Vec<&[u8]> = vec![b"from", b" typing", b" import", b" Any", b",", b" List", b","];
        // let grammar_tokenss_for_comp = vec![vec!["\"from\"", "NAME[0]", "\"import\"", "NAME[0]", "\",\"", "NAME[0]", "\",\""]];
        let llm_tokens_for_comp: Vec<&[u8]> = vec![b"from", b" x"];
        let grammar_tokenss_for_comp = vec![vec!["\"from\"", "NAME[0]"]];
        let llm_token_ids_for_comp = llm_tokens_for_comp.iter().map(|llm_token| llm_token_map.get_by_left(*llm_token).expect(format!("LLM token '{}' not found in llm_token_map", String::from_utf8_lossy(*llm_token)).as_str())).collect::<Vec<_>>();

        let mut parser_state_for_comp = grammar_constraint.parser.init_glr_parser(Some(constraint_state.parent.llm_vocab.clone()));
        for grammar_tokens in grammar_tokenss_for_comp {
            let mut this_parser_state = grammar_constraint.parser.init_glr_parser(Some(constraint_state.parent.llm_vocab.clone()));
            for grammar_token in &grammar_tokens {
                let grammar_token_id = grammar_constraint.parser.terminal_map.get_by_left(&terminal(grammar_token)).unwrap();
                this_parser_state.step(*grammar_token_id);
                assert!(this_parser_state.is_ok(), "Parser failed to step with token {:?} in sequence {:?}", grammar_token, grammar_tokens);
            }
            parser_state_for_comp.merge_with(this_parser_state);
        }

        let mut constraint_state_for_comp = grammar_constraint.init();
        for llm_token_for_comp in llm_tokens_for_comp {
            // Ensure token is allowed before committing
            let llm_token_id_for_comp = llm_token_map.get_by_left(llm_token_for_comp).unwrap();
            let mask_before_commit = constraint_state_for_comp.get_mask();
            assert!(mask_before_commit.contains(llm_token_id_for_comp.0), "Token {:?} (ID {}) not found in mask during comparison setup. Mask: {:?}", String::from_utf8_lossy(llm_token_for_comp), llm_token_id_for_comp.0, mask_before_commit);
            constraint_state_for_comp.commit(*llm_token_id_for_comp);
        }
    }

    if false {
        let mut constraint_state_for_comp = grammar_constraint.init();
        // Ensure the parse state after stepping the constraint with a prefix of LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
        for (i, &llm_token_id) in test_token_sequence_ids.iter().enumerate() {
            let test_token_sequence_ids_prefix = &test_token_sequence_ids[..=i];
            let current_token_str = &tokenized_strs_for_logging[i];
            let current_token_bytes = current_token_str.as_bytes();
            let current_token_length = current_token_bytes.len();
            println!("\nProcessing token {}/{}: {:?} (LLMTokenID({}))",
                i + 1, // 1-indexed for display
                test_token_sequence_ids.len(),
                current_token_str,
                llm_token_id.0
            );
            println!("  Current token string: '{}'", current_token_str);
            constraint_state_for_comp.commit(llm_token_id);
            println!("  Committed LLMTokenID({}) for '{}'.", llm_token_id.0, current_token_str);
            println!("  Committing whole prefix of LLM tokens up to this point to freshconstraint state for comparison.");
            let mut other_constraint_state = grammar_constraint.init();
            let full_prefix: Vec<u8> = test_token_sequence_ids_prefix
                .iter()
                .flat_map(|id| llm_token_map.get_by_right(id).unwrap())
                .cloned()
                .collect();
            let current_text_byte_offset = full_prefix.len() - current_token_bytes.len(); // Adjust to the start of the current token
            println!("  Committing prefix of length {}: {:?}", full_prefix.len(), String::from_utf8_lossy(&full_prefix));
            print_token_context(
                &full_text_to_tokenize,
                &all_code_lines,
                current_text_byte_offset,
                current_text_byte_offset + current_token_length, // End at the current byte offset plus the length of the prefix
                2, // Context lines
            );
            other_constraint_state.commit_bytes(&full_prefix);
            let left_str = format!("{}", constraint_state_for_comp);
            let right_str = format!("{}", other_constraint_state);
            println!("\n--- Left State ---\n{}", left_str);
            println!("\n--- Right State ---\n{}", right_str);
            if constraint_state_for_comp != other_constraint_state {
                println!("  Constraint states differ after committing prefix bytes at step {}.", i);
                // Print text diff between Display representation of states
                let diff = TextDiff::from_lines(&left_str, &right_str);

                println!("\n--- Text Diff ---");
                for change in diff.iter_all_changes() {
                    let sign = match change.tag() {
                        similar::ChangeTag::Delete => "-",
                        similar::ChangeTag::Insert => "+",
                        similar::ChangeTag::Equal => " ",
                    };
                    print!("{}{}", sign, change);
                }
                println!("--- End Diff ---\n");
                assert_eq!(left_str, right_str,
                    "State after committing tokens one-by-one should match state after committing prefix bytes at step {}", i
                );
                assert_eq!(constraint_state_for_comp, other_constraint_state,
                    "Constraint state after committing tokens one-by-one should match state after committing prefix bytes at step {}",
                    i
                );
            }
        }
    }

    // assert_eq!(constraint_state_for_comp.state().len(), 1, "Constraint state for comparison should have one tokenizer state");
    // let initial_tokenizer_state_id = constraint_state_for_comp.parent.tokenizer.initial_state_id();
    // let mut actual_constraint_parser_state_comp = constraint_state_for_comp.state()[&initial_tokenizer_state_id].clone();
    //
    // let mut comparable_parser_gss_comp = (*parser_state_for_comp.active_state.stack).clone();
    // let mut comparable_parser_active_state_comp = ParseState { stack: Arc::new(comparable_parser_gss_comp) };
    //
    //
    // Arc::make_mut(&mut comparable_parser_active_state_comp.stack).reset_tokens();
    // Arc::make_mut(&mut actual_constraint_parser_state_comp.active_state.stack).reset_tokens();
    //
    // assert_eq!(actual_constraint_parser_state_comp.active_state, comparable_parser_active_state_comp, "GSS structures for comparison should match");
    // println!("Number of states: {}", constraint_state_for_comp.state().len());
    // let roots = constraint_state_for_comp.state().values().map(|state| state.active_state.stack.as_ref().clone()).collect::<Vec<_>>();
    // println!("State statistics: {:?}", gather_gss_stats(&roots.iter().collect::<Vec<_>>()));
    // for (tokenizer_state_id, state) in constraint_state_for_comp.state() {
    //     println!("  State {}: {:?}", tokenizer_state_id.0, gather_gss_stats(&vec![&state.active_state.stack.as_ref().clone()]));
    // }

    Ok(())
}

