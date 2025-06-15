use crate::glr::parser::ParseState;
use rand::rngs::StdRng;
use std::collections::{BTreeMap, BTreeSet};
use crate::finite_automata::{eat_u8, Match};
use crate::{choice, choice_fast, groups, seq, seq_fast};
use crate::glr::grammar::{nt, prod, t, NonTerminal, Production, Symbol, Terminal};
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
use crate::datastructures::gss::{gather_gss_stats, reset_llm_tokens};
use crate::datastructures::gss::acc_mod::Acc;
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
    let mut token_name_map: BiBTreeMap<String, usize> = BiBTreeMap::new();
    token_name_map.insert("FSTRING_MIDDLE".to_string(), 0 as usize);
    token_name_map.insert("DEF".to_string(), 1 as usize);

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
fn test_constraint_from_serialized_compiled_grammar_and_gpt2_vocab() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define file path for the serialized CompiledGrammar
    let serialized_definition_path = "src/serialized_grammar_definition.json";

    println!("Loading GrammarDefinition from: {}", serialized_definition_path);
    let json_string = match fs::read_to_string(serialized_definition_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read serialized grammar definition file '{}': {}", serialized_definition_path, e);
            eprintln!("Please ensure the file exists and is readable. Skipping this test.");
            return Ok(());
        }
    };

    let json_node = JSONNode::from_json_string(&json_string)?;
    let grammar_definition = GrammarDefinition::from_json(json_node)?;
    println!("Successfully loaded GrammarDefinition from JSON.");

    // Test serialization/deserialization
    assert_eq!(grammar_definition, GrammarDefinition::from_json(grammar_definition.to_json())?);

    println!("Compiling GrammarDefinition into CompiledGrammar...");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition.clone()));
    println!("Successfully compiled GrammarDefinition into CompiledGrammar.");
    println!("{}", compiled_grammar);
    // --- New test section for grammar terminal sequences ---
    println!("\nTesting GLR parser with specific grammar terminal sequences...");

    // Ensure the test string tokenizes as expected.
    let text = b"f\"";
    let mut expected_matches = Vec::new();
    let expected_terminal_name = "NAME[0]";
    let name_group_id = *grammar_definition.terminal_name_to_group_id.get_by_left(expected_terminal_name).unwrap();
    expected_matches.push(Token {
        id: name_group_id,
        width: 1,
    });
    let expected_terminal_name = "FSTRING_MIDDLE[0]";
    let fstring_middle_group_id = *grammar_definition.terminal_name_to_group_id.get_by_left(expected_terminal_name).unwrap();
    expected_matches.push(Token {
        id: fstring_middle_group_id,
        width: 2,
    });
    let expected_terminal_name = "FSTRING_START[0]";
    let fstring_start_group_id = *grammar_definition.terminal_name_to_group_id.get_by_left(expected_terminal_name).unwrap();
    expected_matches.push(Token {
        id: fstring_start_group_id,
        width: 2,
    });
    let results = compiled_grammar.tokenizer.execute_from_state(text, TokenizerStateID(0));
    // TODO: uncomment this
    // assert_eq!(results.matches.iter().collect::<BTreeSet<_>>(), expected_matches.iter().collect::<BTreeSet<_>>());

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
        let mut glr_state = compiled_grammar.glr_parser.init_glr_parser();

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
            let mut glr_state = compiled_grammar.glr_parser.init_glr_parser();

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

    llm_token_map.retain(|v, _| v.len() <= 1);

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
    let mut temp = grammar_constraint.original_to_internal_id_bimap.iter().collect::<Vec<_>>();
    temp.sort_by_key(|(original_id, internal_id)| *internal_id);
    for (original_id, internal_id) in temp {
        let token = llm_token_map.get_by_right(&LLMTokenID(*original_id)).unwrap();
        println!("  original {}, internal {}, token {:?}, raw: {:?}", original_id, internal_id, String::from_utf8_lossy(token), token);
    }

    // TODO: uncomment this
    // Ensure there's an edge in the root precompute node for state 0 that has the terminal for `IGNORE[0][0][1]` on the edge key and which the LLM token for "\n" on the edge value.
    {
        // 1. Get the root precompute node for tokenizer state 0.
        let precompute_root_node = grammar_constraint.precomputed.get(&TokenizerStateID(0))
            .expect("Precomputed data for tokenizer state 0 should exist.").lock().unwrap();

        // 2. Get the TerminalID for the terminal we are interested in.
        let newline_terminal_name = "IGNORE[0][0][1]".to_string();
        let newline_terminal_id = grammar_constraint.parser.terminal_map
            .get_by_left(&Terminal(newline_terminal_name.clone()))
            .unwrap_or_else(|| panic!("Terminal '{}' not found in parser's terminal map.", newline_terminal_name));

        // 3. Get the LLMTokenID for the newline character.
        let newline_bytes = b"\n";
        let newline_llm_token_id = grammar_constraint.llm_token_map
            .get_by_left(&newline_bytes.to_vec())
            .unwrap_or_else(|| panic!("LLM token for newline '{:?}' not found in token map.", String::from_utf8_lossy(newline_bytes)));
        let newline_llm_token_id = grammar_constraint.original_to_internal_id_bimap.get_by_left(&newline_llm_token_id.0).unwrap();

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
    let vocab_tokens_for_tree: Vec<(usize, Vec<u8>)> = grammar_constraint.llm_token_map
        .iter()
        .map(|(bytes, llm_id)| (llm_id.0, bytes.clone()))
        .collect();
    let tokenizer_vocab_tree = VocabPrefixTree::build(&vocab_tokens_for_tree);

    // The full text to tokenize.
    let example_code_path = "src/example_code.py";
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
    let full_text_to_tokenize = "x=1";

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
    constraint_state.commit_bytes(full_text_to_tokenize.as_bytes());
    constraint_state.is_active();

    // 5. Basic Interaction with the GrammarConstraintState
    let mut constraint_state = grammar_constraint.init();
    // Initial step to populate possibilities
    let initial_mask = constraint_state.get_mask();
    println!("\nInitial mask obtained ({} allowed LLM tokens).", initial_mask.iter_bits().count());
    let all_code_lines: Vec<&str> = full_text_to_tokenize.lines().collect();
    let mut current_text_byte_offset = 0;
    // return Ok(());

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
            "Expected LLMTokenID({}) for '{}' to be in the mask. Mask: {:?}",
            llm_token_id.0, current_token_str, current_mask
        );
        println!("  LLMTokenID({}) for '{}' is in the mask.", llm_token_id.0, current_token_str);

        let commit_start = Instant::now();
        constraint_state.commit(llm_token_id);
        let commit_duration = commit_start.elapsed();
        println!("  commit LLMTokenID({}) took: {:?}", llm_token_id.0, commit_duration);
        println!("  Committed LLMTokenID({}) for '{:?}'.", llm_token_id.0, current_token_str);

        assert!(
            constraint_state.is_active(),
            "Constraint state should be active after committing token {} ('{}')",
            i + 1, current_token_str
        );
        println!("  Constraint state is active after commit.");

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

    if false {
        // Ensure the parse state after stepping the constraint with all LLM tokens and committing an LLM token is the same as the parse state after stepping the parser itself tokens emitted by the tokenizer for that same LLM token.
        // In general, this should be true if all LLM tokens cleanly match grammar tokens (or, equivalently, if the only non-empty entry in the precompute tree is under the initial tokenizer state).
        // let grammar_tokenss = vec![vec!["\"from\""], vec!["NAME[0]"]];
        // let llm_tokens_for_comp: Vec<&[u8]> = vec![b"from", b" typing", b" import", b" Any", b",", b" List", b","];
        // let grammar_tokenss_for_comp = vec![vec!["\"from\"", "NAME[0]", "\"import\"", "NAME[0]", "\",\"", "NAME[0]", "\",\""]];
        let llm_tokens_for_comp: Vec<&[u8]> = vec![b"from", b" x"];
        let grammar_tokenss_for_comp = vec![vec!["\"from\"", "NAME[0]"]];
        let llm_token_ids_for_comp = llm_tokens_for_comp.iter().map(|llm_token| llm_token_map.get_by_left(*llm_token).expect(format!("LLM token '{}' not found in llm_token_map", String::from_utf8_lossy(*llm_token)).as_str())).collect::<Vec<_>>();

        let mut parser_state_for_comp = grammar_constraint.parser.init_glr_parser();
        for grammar_tokens in grammar_tokenss_for_comp {
            let mut this_parser_state = grammar_constraint.parser.init_glr_parser();
            for grammar_token in &grammar_tokens {
                let grammar_token_id = grammar_constraint.parser.terminal_map.get_by_left(&Terminal(grammar_token.to_string())).unwrap();
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

#[test]
fn test_minimize_grammar_for_mask_bug() -> Result<(), Box<dyn std::error::Error>> {
    // --- Initial Setup ---
    let serialized_definition_path = "src/serialized_grammar_definition.json";
    println!("[Minimizer] Loading base GrammarDefinition from: {}", serialized_definition_path);
    let json_string = fs::read_to_string(serialized_definition_path)?;
    let json_node = JSONNode::from_json_string(&json_string)?;
    let grammar_definition = Arc::new(GrammarDefinition::from_json(json_node)?);
    println!("[Minimizer] Successfully loaded GrammarDefinition.");

    // Load a vocabulary
    println!("[Minimizer] Loading GPT-2 vocabulary for mask testing...");
    let cache_dir = Path::new(".cache/test_vocabs");
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let vocab_file_name = "gpt2_vocab.json";
    let mut gpt2_raw_vocab = load_or_download_gpt2_vocab(cache_dir, vocab_file_name, vocab_url)?;
    // Just fill with all bytes
    // let mut gpt2_raw_vocab = BTreeMap::new();
    // for i in 0u8..=255u8 {
    //     let c = i as char;
    //     let s = c.to_string();
    //     gpt2_raw_vocab.insert(s, i as usize);
    // }
    gpt2_raw_vocab.insert("import".to_string(), gpt2_raw_vocab.len() as u32);
    gpt2_raw_vocab.insert(" typing".to_string(), gpt2_raw_vocab.len() as u32);
    // let gpt2_raw_vocab = BTreeMap::from([("from", 0), (" typing", 1)]);
    gpt2_raw_vocab.retain(|k, _| [b"from".as_ref(), b" typing"].contains(&k.as_ref()) || k.len() <= 2);

    let mut llm_token_map = LLMTokenMap::new();
    for (token_str, id_val_u32) in gpt2_raw_vocab {
        let token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        llm_token_map.insert(token_str.into_bytes(), LLMTokenID(id_val_u32 as usize));
    }
    // Ensure the specific tokens we need for the test are present
    if !llm_token_map.contains_left(b"from".as_ref()) {
        panic!("The required token 'from' is not in the loaded vocabulary. Cannot run the mask bug minimizer.");
    }
    if !llm_token_map.contains_left(b" typing".as_ref()) {
        panic!("The required token ' typing' is not in the loaded vocabulary. Cannot run the mask bug minimizer.");
    }
    println!("[Minimizer] Vocabulary loaded.");

    let initial_productions = grammar_definition.productions.clone();
    let augmented_start_rule_lhs = initial_productions[grammar_definition.start_production_id].lhs.clone();

    // --- Define the Predicate for the Mask Bug ---
    let predicate = |prods: &[Production], start_lhs: &NonTerminal| -> bool {
        let mut new_def = (*grammar_definition).clone();
        new_def.productions = prods.to_vec();

        let start_prod_id = match new_def.productions.iter().position(|p| p.lhs == *start_lhs) {
            Some(id) => id,
            None => return false,
        };
        new_def.start_production_id = start_prod_id;

        let compiled_grammar = match panic::catch_unwind(AssertUnwindSafe(|| {
            CompiledGrammar::from_definition(Arc::new(new_def))
        })) {
            Ok(cg) => cg,
            Err(_) => return false,
        };

        let max_id = llm_token_map.right_values().map(|id| id.0).max().unwrap_or(0);
        let constraint = match panic::catch_unwind(AssertUnwindSafe(|| {
            GrammarConstraint::from_compiled_grammar(compiled_grammar, llm_token_map.clone(), LLMTokenID(0), max_id)
        })) {
            Ok(c) => c,
            Err(_) => return false,
        };

        // The actual test logic
        let bug_found = panic::catch_unwind(AssertUnwindSafe(|| {
            let mut constraint_state = constraint.init();
            let import_token_id = llm_token_map.get_by_left(b"from".as_ref()).unwrap();
            let initial_mask = constraint_state.get_mask();
            dbg!(&initial_mask);

            if !initial_mask.contains(import_token_id.0) {
                println!("BUG: \"from\" should be allowed initially.");
                return true; // BUG: "from" should be allowed initially.
            }

            constraint_state.commit(*import_token_id);
            if !constraint_state.is_active() {
                println!("BUG: State should be active after a valid token.");
                return true; // BUG: State should be active after a valid token.
            }

            let typing_token_id = llm_token_map.get_by_left(b" typing".as_ref()).unwrap();
            let next_mask = constraint_state.get_mask();

            if !next_mask.contains(typing_token_id.0) {
                println!("BUG: \" typing\" should be allowed after \"from\".");
                return true; // BUG: " typing" should be allowed after "from".
            }
            
            false // No bug found
        }));

        match bug_found {
            Ok(found) => found,
            Err(_) => false, // A panic during constraint interaction is also a bug, but not a bug we want to zero in on.
        }
    };

    minimize_grammar_and_assert(
        "mask_bug",
        initial_productions,
        augmented_start_rule_lhs,
        predicate,
    )
}

const PANIC_SUBSTRING_TO_FIND: &str = "not found in gotos for";

/// Generic grammar minimization tool.
/// It repeatedly tries to remove productions and symbols from a grammar,
/// keeping any change that still causes a given `predicate` to return true.
fn minimize_grammar_and_assert<F>(
    test_name: &str,
    initial_productions: Vec<Production>,
    augmented_start_rule_lhs: NonTerminal,
    predicate: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: Fn(&[Production], &NonTerminal) -> bool,
{
    println!("[Minimizer] Running for test: {}", test_name);
    println!("[Minimizer] Initial number of productions: {}", initial_productions.len());
    println!("[Minimizer] Augmented start LHS: {}", augmented_start_rule_lhs.0);

    let mut current_productions = initial_productions;

    if !predicate(&current_productions, &augmented_start_rule_lhs) {
        eprintln!("[Minimizer] Initial grammar does not trigger the bug predicate. Cannot proceed.");
        assert!(false, "Initial grammar does not trigger the bug predicate for test '{}'.", test_name);
        return Ok(());
    }
    println!("[Minimizer] Confirmed: Initial grammar triggers the bug predicate.");

    let mut pass_num = 0;
    loop { // This is the main loop that alternates between production and symbol removal until convergence
        pass_num += 1;
        println!(
            "\n[Minimizer] Starting Pass {}. Current productions: {}",
            pass_num,
            current_productions.len()
        );
        let prods_at_pass_start = current_productions.clone();

        // --- Production removal phase ---
        println!("[Minimizer] Pass {}: Trying to remove productions.", pass_num);
        let mut n_prods;
        let mut removable_indices: Vec<usize> = (0..current_productions.len())
            .filter(|&idx| current_productions[idx].lhs != augmented_start_rule_lhs)
            .collect();
        n_prods = removable_indices.len() / 2;

        while n_prods > 0 {
            let mut current_n = n_prods;
            if current_n > removable_indices.len() {
                current_n = removable_indices.len();
            }
            if current_n == 0 { break; }

            println!("[Minimizer]   Trying production chunk size: {}", current_n);
            let mut i = 0;
            let mut changed_this_sub_pass = false;
            while i < removable_indices.len() {
                let indices_to_remove: Vec<usize> = removable_indices.iter().skip(i).take(current_n).cloned().collect();
                
                if indices_to_remove.is_empty() {
                    i += current_n.max(1);
                    continue;
                }

                let mut temp_prods = current_productions.clone();
                let mut sorted_indices = indices_to_remove;
                sorted_indices.sort_unstable_by(|a, b| b.cmp(a));
                for &idx_to_remove in &sorted_indices {
                    temp_prods.remove(idx_to_remove);
                }

                if predicate(&temp_prods, &augmented_start_rule_lhs) {
                    println!("[Minimizer]   Removed a chunk of {} productions. New count: {}", sorted_indices.len(), temp_prods.len());
                    current_productions = temp_prods;
                    changed_this_sub_pass = true;
                    break; // Restart this phase with new `n`
                }
                i += current_n.max(1);
            }

            if changed_this_sub_pass {
                removable_indices = (0..current_productions.len())
                    .filter(|&idx| current_productions[idx].lhs != augmented_start_rule_lhs)
                    .collect();
                n_prods = removable_indices.len() / 2;
            } else {
                n_prods /= 2;
            }
        }

        // --- Symbol removal phase ---
        println!("[Minimizer] Pass {}: Trying to remove symbols.", pass_num);
        let mut all_symbol_locations: Vec<(usize, usize)> = Vec::new();
        for (prod_idx, prod) in current_productions.iter().enumerate() {
            for symbol_idx in 0..prod.rhs.len() {
                all_symbol_locations.push((prod_idx, symbol_idx));
            }
        }
        let mut n_symbols = all_symbol_locations.len() / 2;

        while n_symbols > 0 {
            let mut current_n_symbols = n_symbols;
            if current_n_symbols > all_symbol_locations.len() {
                current_n_symbols = all_symbol_locations.len();
            }
            if current_n_symbols == 0 { break; }

            println!("[Minimizer]   Trying symbol chunk size: {}", current_n_symbols);
            let mut i = 0;
            let mut changed_this_sub_pass = false;
            while i < all_symbol_locations.len() {
                let locations_to_remove: Vec<(usize, usize)> = all_symbol_locations.iter().skip(i).take(current_n_symbols).cloned().collect();
                
                if locations_to_remove.is_empty() {
                    i += current_n_symbols.max(1);
                    continue;
                }

                let mut temp_prods = current_productions.clone();
                remove_symbols_at_locations_destructive(&mut temp_prods, &locations_to_remove);

                if predicate(&temp_prods, &augmented_start_rule_lhs) {
                    println!("[Minimizer]   Removed a chunk of {} symbols.", locations_to_remove.len());
                    current_productions = temp_prods;
                    changed_this_sub_pass = true;
                    break;
                }
                i += current_n_symbols.max(1);
            }

            if changed_this_sub_pass {
                all_symbol_locations.clear();
                for (prod_idx, prod) in current_productions.iter().enumerate() {
                    for symbol_idx in 0..prod.rhs.len() {
                        all_symbol_locations.push((prod_idx, symbol_idx));
                    }
                }
                n_symbols = all_symbol_locations.len() / 2;
            } else {
                n_symbols /= 2;
            }
        }

        if current_productions == prods_at_pass_start {
            println!("[Minimizer] Pass {}: No further reductions found. Converged.", pass_num);
            break;
        }
    }

    println!("\n[Minimizer] Minimization phase complete.");
 
    // --- Final Simplification Pass: Inline A -> B rules ---
    println!("\n[Minimizer] Deterministic simplification phase starting.");
    loop {
        let productions_at_start_of_deterministic_iter = current_productions.clone();

        // Apply A -> B inlining (iterates to fixed point internally)
        let prev_len_unit_inline = current_productions.len();
        current_productions = simplify_and_inline_unit_nonterminal_rules(
            current_productions, // Takes ownership
            &augmented_start_rule_lhs,
            &predicate,
        );
        if current_productions.len() != prev_len_unit_inline {
             println!("[Minimizer-Determ] simplify_and_inline_unit_nonterminal_rules changed productions: {} -> {}", prev_len_unit_inline, current_productions.len());
        }

        // Apply A -> alpha (sole production) inlining (iterates to fixed point internally)
        let prev_len_sole_inline = current_productions.len();
        let (next_prods_after_sole_inline, _sole_inlined_overall) = inline_sole_productions_pass(
            current_productions, // Takes ownership
            &augmented_start_rule_lhs,
            &predicate,
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

    assert!(false, "[Minimizer] Minimization for '{}' finished. Review the MRE printed above and address the underlying bug.", test_name);
    Ok(())
}

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
            BTreeMap::new(), // No actions
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
fn simplify_and_inline_unit_nonterminal_rules<F>(
    mut productions: Vec<Production>, // Takes ownership, returns new Vec
    augmented_start_rule_lhs: &NonTerminal,
    predicate: &F,
) -> Vec<Production>
where
    F: Fn(&[Production], &NonTerminal) -> bool,
{
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
            } else if predicate(&temp_productions_after_inlining, augmented_start_rule_lhs) {
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

fn inline_sole_productions_pass<F>(
    mut productions: Vec<Production>, // Takes ownership
    augmented_start_rule_lhs: &NonTerminal,
    predicate: &F,
) -> (Vec<Production>, bool) 
where
    F: Fn(&[Production], &NonTerminal) -> bool,
{ // Returns new productions and bool indicating if a change was made
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
            } else if predicate(&temp_productions_after_inlining, augmented_start_rule_lhs) {
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

// #[ignore]
#[test]
fn test_minimize_grammar_for_goto_panic() -> Result<(), Box<dyn std::error::Error>> {
    // --- Initial Setup ---
    let serialized_definition_path = "src/serialized_grammar_definition.json";
    let json_string = match fs::read_to_string(serialized_definition_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[Minimizer] Failed to read file '{}': {}. Skipping test.", serialized_definition_path, e);
            return Ok(());
        }
    };
    let json_node = JSONNode::from_json_string(&json_string)?;
    let grammar_definition = GrammarDefinition::from_json(json_node)?;
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));

    let initial_productions = compiled_grammar.definition.productions.clone();
    let augmented_start_rule_lhs = compiled_grammar.definition.productions
        [compiled_grammar.definition.start_production_id].lhs.clone();
    
    let sequence_to_test_names = ["\"return\"", "\";\"", "IGNORE[0][0]", "\"[\"[0]"];

    let predicate = |prods: &[Production], start_lhs: &NonTerminal| -> bool {
        causes_specific_panic(prods, start_lhs, &sequence_to_test_names, PANIC_SUBSTRING_TO_FIND)
    };

    minimize_grammar_and_assert(
        "goto_panic",
        initial_productions,
        augmented_start_rule_lhs,
        predicate,
    )
}

#[ignore]
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
        BTreeMap::new(), // No actions
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
